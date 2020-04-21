import matlab.engine
import os,sys
import logging
import argparse
import numpy as np
from train_and_evaluate import evaluate, train
from net import Generator,Generator0
import utils
import torch
import shutil
from torch.utils.tensorboard import SummaryWriter




class Args:
    def __init__(self,wavelength,angle,gen_ver):
        self.wavelength=wavelength
        self.angle=angle
        self.gen_ver=gen_ver
        self.output_dir=r'./scan/results'+'-deg{}_wl{}_gen_ver{}'.format(self.angle,self.wavelength,self.gen_ver)
        self.restore_from=None

def copyfile(output_dir,srcfile=r'./results/Params.json'):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath=output_dir   #获取文件路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #没有就创建路径
        shutil.copyfile(srcfile,os.path.join(fpath,r'Params.json'))      #复制文件到默认路径
        print ("copy %s -> %s"%( srcfile,os.path.join(fpath,r'Params.json')))  



def autosim(args,eng):
   
    os.makedirs(args.output_dir, exist_ok = True)
     # Set the logger
    utils.set_logger(os.path.join(args.output_dir, 'train.log'))

    copyfile(args.output_dir)


     # Load parameters from json file
    json_path = os.path.join(args.output_dir,'Params.json')
    assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Add attributes to params
    params.output_dir = args.output_dir
    params.cuda = torch.cuda.is_available()
    params.restore_from = args.restore_from
    params.numIter = int(params.numIter)
    params.noise_dims = int(params.noise_dims)
    params.gkernlen = int(params.gkernlen)
    params.step_size = int(params.step_size)
    params.gen_ver = int(args.gen_ver)    
    params.dime = 1
    if args.wavelength is not None:
        params.wavelength = int(args.wavelength)
    if args.angle is not None:
        params.angle = int(args.angle)
         #build a recorder
    max_recorder=utils.max_recorder()
    params.recorder=max_recorder

    #build tools
    writer = SummaryWriter(log_dir=r'./scan/runs')
    max_recorder=utils.max_recorder()
    params.recorder=max_recorder
    params.writer= writer

    # make directory
    os.makedirs(args.output_dir + '/outputs', exist_ok = True)
    os.makedirs(args.output_dir + '/model', exist_ok = True)
    os.makedirs(args.output_dir + '/figures/histogram', exist_ok = True)
    os.makedirs(args.output_dir + '/figures/deviceSamples', exist_ok = True)
    os.makedirs(args.output_dir + '/figures/deviceSamples_max', exist_ok = True)
    os.makedirs(args.output_dir + '/deg{}_wl{}_gen_ver{}'.format(params.angle,params.wavelength,params.gen_ver), exist_ok = True)
    # Define the models
    if  params.gen_ver==0:
        generator = Generator0(params)
    else:     
        generator = Generator(params)

    # Move to gpu if possible
    if params.cuda:
        generator.cuda()


    # Define the optimizer
    optimizer = torch.optim.Adam(generator.parameters(), lr=params.lr, betas=(params.beta1, params.beta2))
    
    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma = params.gamma)


    # Load model data
    if args.restore_from is not None :
        params.checkpoint = utils.load_checkpoint(restore_from, generator, optimizer, scheduler)
        logging.info('Model data loaded')

    #set the timer
    timer=utils.timer()

    # Train the model and save 
    if params.numIter != 0 :
        logging.info('Start training')   
        train(generator, optimizer, scheduler, eng, params)

    # Generate images and save 
    logging.info('Start generating devices')
    evaluate(generator, eng, numImgs=500, params=params)

    timer.out()
    writer.close()

# start matlab engine
eng = matlab.engine.start_matlab()
# RCWA path
eng.addpath(eng.genpath('/home/users/jiangjq/Desktop/reticolo_allege'));
eng.addpath(eng.genpath('solvers'));

wlrange=range(900,1200,100)
degrange=range(40,80,10)
gen_vera=[0,1]
def paramscan(wlrange,degrange,gen_vera,eng):
    for wavelength in wlrange:
        for angle in degrange:
            for gen_ver in gen_vera: 
                args=Args(wavelength,angle,gen_ver)
                if os.path.exists(args.output_dir):
                    pass
                else:
                    print("start training",args.output_dir)
                    autosim(args,eng)

paramscan(wlrange,degrange,gen_vera,eng)
