import sys
from PIL import Image
from NewMain import Args
import os
def concate(wlrange,degrange,gen_vera):
    xlen=len(degrange)
    ylen=len(wlrange)*len(gen_vera)
    
    box=(200,70,1800,1440)
    path=r'D:\gradut\GLOnet-master\scan\Iter500.png' 
    img=Image.open(path).crop(box)
    #img.show()
    #print(img.size)
    #input()
    widths, heights=img.size
    total_width = xlen*(widths)
    total_height = ylen*(heights)

    new_im = Image.new('RGB', (total_width, total_height))

    for y,wavelength in enumerate(wlrange):
        for x,angle in enumerate(degrange):
            for y1,gen_ver in enumerate(gen_vera):
                args=Args(wavelength,angle,gen_ver)
                
                #path=os.path.join(args.output_dir+r'/',r'figures/Efficiency.png') 
                path=os.path.join(args.output_dir+r'/',r'figures/histogram/Iter{}.png'.format(300)) 
                offset=(x*widths,y*2*heights+y1*heights)
                with Image.open(path) as im:
                    im=im.crop(box)     
                    new_im.paste(im, offset)

    
    new_im.show()
    #save('test.jpg')
wlrange=list(reversed(range(900,1200,100)))
degrange=range(40,80,10)
gen_vera=[0,1]    
concate(wlrange,degrange,gen_vera)