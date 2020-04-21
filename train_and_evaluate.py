import matlab.engine
import os
import logging
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
import torch.nn.functional as F 
import torch
import utils
import scipy.io as io
import numpy as np
import math
from matplotlib import cm 
from matplotlib import axes
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def evaluate(generator, eng, numImgs, params):
    generator.eval()
    
    # generate images
    z = sample_z(numImgs, params)
    images = generator(z, params)
    logging.info('Generation is done. \n')

    # evaluate efficiencies
    images = torch.sign(images)
    effs = compute_effs(images, eng, params)

    # save images
    filename = 'imgs_w' + str(params.wavelength) +'_a' + str(params.angle) +'deg.mat'
    file_path = os.path.join(params.output_dir,'outputs',filename)
    io.savemat(file_path, mdict={'imgs': images.cpu().detach().numpy(), 
                                 'effs': effs.cpu().detach().numpy()})

    # plot histogram
    fig_path = params.output_dir + '/figures/Efficiency.png'
    utils.plot_histogram(effs.data.cpu().numpy().reshape(-1), params.numIter, fig_path,params)




def train(generator, optimizer, scheduler, eng, params, pca=None):

    generator.train()

    # initialization
    if params.restore_from is None:
        effs_mean_history = []
        binarization_history = []
        diversity_history = []
        iter0 = 0   
    else:
        effs_mean_history = params.checkpoint['effs_mean_history']
        binarization_history = params.checkpoint['binarization_history']
        diversity_history = params.checkpoint['diversity_history']
        iter0 = params.checkpoint['iter']
    
    nois=sample_z(1, params)
    params.writer.add_graph(generator,nois)

    # training loop
    with tqdm(total=params.numIter) as t:
        it = 0  
        while True:
            it +=1 
            params.iter = it + iter0

            # normalized iteration number
            normIter = params.iter / params.numIter

            # specify current batch size 
            params.batch_size = int(params.batch_size_start +  (params.batch_size_end - params.batch_size_start) * (1 - (1 - normIter)**params.batch_size_power))
            
            # sigma decay
            params.sigma = params.sigma_start + (params.sigma_end - params.sigma_start) * normIter

            # learning rate decay
            

            # binarization amplitude in the tanh function
            if params.iter < 1000:
                params.binary_amp = int(params.iter/100) + 1 
            else:
                params.binary_amp = 10

            # save model 
            if it % 5000 == 0 or it > params.numIter:
                model_dir = os.path.join(params.output_dir, 'model','iter{}'.format(it+iter0))
                os.makedirs(model_dir, exist_ok = True)
                utils.save_checkpoint({'iter': it + iter0 - 1,
                                       'gen_state_dict': generator.state_dict(),
                                       'optim_state_dict': optimizer.state_dict(),
                                       'scheduler_state_dict': scheduler.state_dict(),
                                       'effs_mean_history': effs_mean_history,
                                       'binarization_history': binarization_history,
                                       'diversity_history': diversity_history
                                       },
                                       checkpoint=model_dir)

            # terminate the loop
            if it > params.numIter:
                return 

            
            # sample  z
            z = sample_z(params.batch_size, params)

            # generate a batch of iamges
            gen_imgs = generator(z, params)


            # calculate efficiencies and gradients using EM solver
            effs, gradients = compute_effs_and_gradients(gen_imgs, eng, params)

            # free optimizer buffer 
            optimizer.zero_grad()

            # construct the loss function
            binary_penalty = params.binary_penalty_start if params.iter < params.binary_step_iter else params.binary_penalty_end
            g_loss = global_loss_function(gen_imgs, effs, gradients, params.sigma, binary_penalty,params.div_pen)
            save_the_max(gen_imgs,effs,params,pr=1)        
            params.writer.add_scalars('Loss-deg{}_wl{}_gen_ver{}'.format(params.angle,params.wavelength,params.gen_ver),{'Loss Curve':g_loss.detach()},it)

            # train the generator
            g_loss.backward()
            optimizer.step()
            scheduler.step()


            # evaluate 
            if it % params.plot_iter == 0:
                generator.eval()

                # vilualize generated images at various conditions
                visualize_generated_images(generator, params)

                # evaluate the performance of current generator
                effs_mean, binarization, diversity = evaluate_training_generator(generator, eng, params)

                # add to history 
                effs_mean_history.append(effs_mean)
                binarization_history.append(binarization)
                diversity_history.append(diversity)

                # plot current history
                utils.plot_loss_history((effs_mean_history, diversity_history, binarization_history), params)
                generator.train()

            t.update()



def sample_z(batch_size, params):
    '''
    smaple noise vector z
    '''
    intensor=np.random.randint(2,size=(batch_size, params.noise_dims))
    intensor= intensor*2.-1.
    return torch.from_numpy(intensor).type(Tensor)
    #return (torch.rand(batch_size, params.dime**2*params.noise_dims).type(Tensor)*2.-1.) * params.noise_amplitude


def compute_effs_and_gradients(gen_imgs, eng, params):
    '''
    Args:
        imgs: N x C x H
        labels: N x labels_dim 
        eng: matlab engine
        params: parameters 

    Returns:
        effs: N x 1
        gradients: N x C x H
    '''
    # convert from tensor to numpy array
    imgs = gen_imgs.clone().detach()
    N = imgs.size(0)
    img = matlab.double(imgs.cpu().numpy().tolist())
    wavelength = matlab.double([params.wavelength] * N)
    desired_angle = matlab.double([params.angle] * N)

    # call matlab function to compute efficiencies and gradients
    
    effs_and_gradients = eng.GradientFromSolver_1D_parallel(img, wavelength, desired_angle)  
    effs_and_gradients = Tensor(effs_and_gradients) 
    effs = effs_and_gradients[:, 0]             
    gradients = effs_and_gradients[:, 1:].unsqueeze(1)

    return (effs, gradients)


def compute_effs(imgs, eng, params):
    '''
    Args:
        imgs: N x C x H
        eng: matlab engine
        params: parameters 

    Returns:
        effs: N x 1
    '''
    # convert from tensor to numpy array
    N = imgs.size(0)
    img = matlab.double(imgs.data.cpu().numpy().tolist())
    wavelength = matlab.double([params.wavelength] * N)
    desired_angle = matlab.double([params.angle] * N)
   
    # call matlab function to compute efficiencies 
    effs = eng.Eval_Eff_1D_parallel(img, wavelength, desired_angle)
    
    
    return Tensor(effs)



def global_loss_function(gen_imgs, effs, gradients, sigma=0.5, binary_penalty=0,div_pen=0):
    '''
    Args:
        gen_imgs: N x C x H (x W)
        effs: N x 1
        gradients: N x C x H (x W)
        max_effs: N x 1
        sigma: scalar
        binary_penalty: scalar
    '''
    
    # efficiency loss
    eff_loss_tensor = - gen_imgs * gradients * (1./sigma) * (torch.exp(-effs/sigma)).view(-1, 1, 1)
    eff_loss = torch.sum(torch.mean(eff_loss_tensor, dim=0).view(-1))
    eff_loss_tensor_total = - gen_imgs * gradients * (1./sigma) * (torch.exp(effs/sigma*0)).view(-1, 1, 1)
    eff_loss_total = torch.sum(torch.mean(eff_loss_tensor_total, dim=0).view(-1))

    # binarization loss
    binary_loss = - torch.mean(torch.abs(gen_imgs.view(-1)) * (2.0 - torch.abs(gen_imgs.view(-1)))) 

    #diversity penalty
    var=torch.mean(torch.var(gen_imgs, dim=0))
    std=torch.mean(torch.std(gen_imgs, dim=0))
    div=0.5/div_pen
    div_loss=1/(0.02+(div*var)**6)
    print("var:",var)
    print("std:",std)
    print("div_loss:",div_loss)
    print("eff_loss_total:",eff_loss_total)
    print("eff_loss:",eff_loss)
    
    
    # total loss
    loss = eff_loss + binary_loss * binary_penalty + div_loss

    #/eff_loss_total
    return loss

def draw_heatmap(data,fig_path,nrow=4):
    cmap=cm.binary

    shape=data.shape
    numel=shape[0]  
    #cmap=cm.get_cmap('rainbow',1000)
    figure=plt.figure(facecolor='b')
    plt.subplots_adjust(left= 0.27 , bottom=0.02, right=1, top=0.97, wspace=0, hspace=0.04) 
    for pic in range(numel):
        
        ax=figure.add_subplot(nrow,nrow,pic+1)
        
        """
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels)
        """
        vmax=data[pic][0][0][0]
        vmin=data[pic][0][0][0]
        for i in data[pic][0]:
            for j in i:
                if j>vmax:
                    vmax=j
                if j<vmin:
                    vmin=j
        map=ax.imshow(data[pic][0],interpolation='None',cmap=cmap,aspect='equal',vmin=vmin,vmax=vmax)
        plt.axis('off')
                
    #cb=plt.colorbar(mappable=map,cax=None,ax=None,shrink=0.5)   
    
  
    figure.savefig(fig_path, facecolor=figure.get_facecolor(), edgecolor='none',bbox_inches = 'tight',
    pad_inches = 0)
    plt.close()

def visualize_generated_images(generator, params, n_row = 4, n_col = 4):
    # generate images and save
    fig_path = params.output_dir +  '/figures/deviceSamples/Iter{}.png'.format(params.iter) 
    
    z = sample_z(n_col * n_row, params)
    imgs = generator(z, params)
    #imgs_2D = imgs.unsqueeze(2).repeat(1, 1, 64, 1)
    #save_image(imgs_2D, fig_path, n_row, range=(-1, 1))
    dim=int(math.sqrt(imgs.size()[-1]))
    imgs_2D = imgs.unsqueeze(2).view(16, 1,dim,dim)
    draw_heatmap(imgs_2D.detach().numpy(), fig_path) 

def evaluate_training_generator(generator, eng, params, num_imgs = 100):

    # generate images
    z = sample_z(num_imgs, params)
    imgs = generator(z, params)

    # efficiencies of generated images
    effs = compute_effs(imgs, eng, params)
    effs_mean = torch.mean(effs.view(-1))
    
    # save the highest
    save_the_max(imgs,effs,params)

    # binarization of generated images
    binarization = torch.mean(torch.abs(imgs.view(-1))).cpu().detach().numpy()

    # diversity of generated images
    diversity = torch.mean(torch.std(imgs, dim=0)).cpu().detach().numpy()
    print("diversity:",diversity)
    #diversity penalty
    var = torch.mean(torch.var(imgs, dim=0)).cpu().detach().numpy()
    print("var:",var)

    # plot histogram
    fig_path = params.output_dir +  '/figures/histogram/Iter{}.png'.format(params.iter) 
    utils.plot_histogram(effs.data.cpu().numpy().reshape(-1), params.iter, fig_path,params)

    
    return effs_mean, binarization, diversity

def save_the_max(imgs,effs,params,pr=0):
    index=torch.max(effs.t(),0)[1]
    max_img=imgs[index]
    max_eff=torch.max(effs)
    if pr==0:
        params.recorder.update(max_eff,max_img)
        visualize_generated_max_images(max_eff,max_img,params)
    else:
        print("max:{}%".format(int(100*max_eff)))    

def visualize_generated_max_images(max_eff,max_img,params):
    # generate images and save
    fig_path = params.output_dir +  '/figures/deviceSamples_max/{}%Eff,{}Iter.png'.format(int(max_eff*100),params.iter) 

    dim=int(math.sqrt(max_img.size()[-1]))
    imgs_2D = max_img.unsqueeze(2).view(1, 1,dim,dim).repeat(1,1,4,4)
    draw_heatmap(imgs_2D.detach().numpy(), fig_path,nrow=1) 