import tensorflow as tf
import numpy as np
import random, cv2, operator, os

from PIL import Image
#print(tf.config.list_physical_devices('GPU'))

dir="/home/felipe/gans_enhancer/input/LPGAN/input/"
def rename_MIT_files(dir):

        # for file in os.listdir(dir):

        #         os.rename(dir+file,dir+file.split("-")[0]+".tif")
        subdirs = [os.path.join(dir,dI) for dI in os.listdir(dir) if os.path.isdir(os.path.join(dir,dI))]
        for subdir in subdirs:
                for file in os.listdir(subdir):

                
                        if file.split(".")[1] in ["jpg",'jpeg','dng']:

                                im = Image.open(subdir+'/'+file)
                                im.save(subdir+'/'+file.split(".")[0]+".tif")
                                os. remove(subdir+'/'+file)                
                        else: 
                                os.rename(subdir+'/'+file,subdir+'/'+file.split(".")[0]+".tif")
                        
                

def resize_image(dir,max_length = 512) :
        subdirs = [os.path.join(dir,dI) for dI in os.listdir(dir) if os.path.isdir(os.path.join(dir,dI))]
        for subdir in subdirs:
                for file in os.listdir(subdir):
                
                        image = Image.open(subdir+'/'+file)
                        print(image.size)
                        max_size = np.argmax(image.size )
                        width, height = image.size 
                        factor=max_length/image.size[max_size]
                        new_image = image.resize((int(factor*width), int(factor*height)), Image.ANTIALIAS)
                        image.close()
                        new_image.save(subdir+'/'+file)

if __name__ =="__main__":
        rename_MIT_files("/home/felipe/deep-photo-enhancer-master/images_LR/input/Training1")
       # resize_image("/home/felipe/deep-photo-enhancer-master/images_LR/input/Testing")
        resize_image("/home/felipe/deep-photo-enhancer-master/images_LR/input/Training1")
       # resize_image("/home/felipe/deep-photo-enhancer-master/images_LR/input/Training2")
       # resize_image("/home/felipe/deep-photo-enhancer-master/images_LR/Expert-C/Testing")
       # resize_image("/home/felipe/deep-photo-enhancer-master/images_LR/Expert-C/Training1")
        rename_MIT_files("/home/felipe/deep-photo-enhancer-master/images_LR/Expert-C/Training2")
        resize_image("/home/felipe/deep-photo-enhancer-master/images_LR/Expert-C/Training2")
       # resize_image("/home/felipe/deep-photo-enhancer-master/images_LR/Expert-C/Training2/hdrs")