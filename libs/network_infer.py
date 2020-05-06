import torch
import torch.nn as nn
from torch.nn import init
import torchvision

import numpy as np

import inspect
from PIL import Image

if torch.cuda.is_available() :   
    device = torch.device('cuda', 0)  # Default CUDA device
else:
    device = torch.device('cpu')
device_ids = [0, 1, 2, 3]  # CUDA ids
Tensor_gpu = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
Tensor = torch.FloatTensor
GPUS_NUM = torch.cuda.device_count()  # the GPUs' number
#torch.nn.InstanceNorm2d
# share all layers from the other model except batchnorms
#correct gadient penalty... aready ok ...ee the calculations for 2 way gan losses
#check order of conv definitions is the order activation batchnorm convolution ok?
#padding should be symmetric not implemented on main pytorch apparently https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/padding.py#L28
# GENERATOR NETWORK
class GeneratorWDilation(nn.Module):
    def __init__(self,dilation):
        super(GeneratorWDilation, self).__init__()
        #  Convolutional layers
        # input 512x512x3  output 512x512x16
        self.conv1 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(3, 16, 5, stride=1, padding=0,dilation = dilation),
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(16)
        )

        # input 512x512x16  output 256x256x32
        self.conv2 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(16, 32, 5, stride=2, padding=0,dilation = dilation),
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(32)
        )

        # input 256x256x32  output 128x128x64
        self.conv3 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(32, 64, 5, stride=2, padding=0,dilation = dilation),
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(64)
        )

        # input 128x128x64  output 64x64x128
        self.conv4 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(64, 128, 5, stride=2, padding=0,dilation = dilation),
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(128)
        )

        # input 64x64x128  output 32x32x128
        # the output of this layer we need layers for global features
        self.conv5 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(128, 128, 5, stride=2, padding=0,dilation = dilation),
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(128)
        )

        # convs for global features
        # input 32x32x128 output 16x16x128
        self.conv51 = nn.Sequential(
            nn.ReplicationPad2d(2), 
            nn.Conv2d(128, 128, 5, stride=2, padding=0,dilation = dilation))

        # input 16x16x128 output 8x8x128
        self.conv52 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(128, 128, 5, stride=2, padding=0,dilation = dilation))

        # input 8x8x128 output 1x1x128
        self.conv53 = nn.Conv2d(128, 128, 8, stride=1, padding=0,dilation = dilation)
        
        # revise this linear laye rin the orginal they appear to use a convoltion
        self.fc = nn.Sequential(
            nn.SELU(inplace=True),
            nn.Conv2d(128,128,1,1)
        )

        # input 32x32x128 output 32x32x128
        # the global features should be concatenated to the feature map after this layer

        #add resizing over here with mode neares torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=False)



        # the output after concat would be 32x32x256
        self.conv6 =  nn.Sequential(  
            nn.Conv2d(128, 128, 1, stride=1, padding=0,dilation = dilation) )

        # input 32x32x256 output 32x32x128
        self.conv7 = nn.Sequential(  nn.ReplicationPad2d(1), 
            nn.Conv2d(256, 128, 3, stride=1, padding=0,dilation = dilation) )

        # deconvolutional layers
        # input 32x32x128 output 64x64x128
        self.dconv1 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(128),
            nn.ReplicationPad2d(1),
            nn.Conv2d(128, 128, 3, stride=1, padding=0,dilation = dilation),
            nn.Upsample(scale_factor=2,  mode='nearest') 
        )

        # input 64x64x128 ouput 128x128x128
        self.dconv2 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(256),
           
            nn.ReplicationPad2d(1),

            nn.Conv2d(256, 128, 3, stride=1, padding=0,dilation = dilation),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # input 128x128x192 output 256x256x96
        self.dconv3 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(192),
            nn.ReplicationPad2d(1),
            nn.Conv2d(192, 64, 3, stride=1, padding=0,dilation = dilation),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # input 256x256x96 ouput 512x512x32
        self.dconv4 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(96),
            nn.ReplicationPad2d(1),
            nn.Conv2d(96, 32, 3, stride=1, padding=0,dilation = dilation),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # final convolutional layers
        # input 512x512x48 output 512x512x16
        self.conv8 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(48),
            #nn.ReplicationPad2d(1),
            nn.Conv2d(48, 16, 1, stride=1, padding=0,dilation = dilation)
        )

        # input 512x512x16 output 512x512x3
        self.conv9 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(16),
            #nn.ReplicationPad2d(1),
            nn.Conv2d(16, 3, 1, stride=1, padding=0,dilation = dilation)
        )

    def forward(self, x):
        # input 512x512x3 to output 512x512x16
        x0 = self.conv1(x)

        # input 512x512x16 to output 256x256x32
        x1 = self.conv2(x0)

        # input 256x256x32 to output 128x128x64
        x2 = self.conv3(x1)

        # input 128x128x64 to output 64x64x128
        x3 = self.conv4(x2)

        # input 64x64x128 to output 32x32x128
        x4 = self.conv5(x3)

        # convolutions for global features
        # input 32x32x128 to output 16x16x128
        x51 = self.conv51(x4)

        # input 16x16x128 to output 8x8x128
        x51 = self.conv52(x51)

        # input 8x8x128 to output 1x1x128
        x53 = self.conv53(x51)
        x53 = self.fc(x53)

        # input 32x32x128 to output 32x32x128
        x5 = self.conv6(x4)


        
        x53_temp = torch.cat([x53] * 32, dim=2)
        x53_temp = torch.cat([x53_temp] * 32, dim=3)


        #x53_temp = torch.cat([x53] * x5.shape[2], dim=2)
       # x53_temp = torch.cat([x53_temp] * x5.shape[3], dim=3)
        
       # x53_temp = torch.reshape(x53_temp, (x5.shape[0],-1,x5.shape[2],x5.shape[3]))
       

       
        # input 32x32x256 to output 32x32x128

        x6 = self.conv7(torch.cat([x5, x53_temp], dim=1))

        # input 32x32x128 to output 64x64x128
        xd = self.dconv1(x6)

        # input 64x64x256 to output 128x128x128
        xd = self.dconv2(torch.cat([xd, x3], dim=1))

        # input 128x128x192 to output 256x256x64
        xd = self.dconv3(torch.cat([xd, x2], dim=1))

        # input 256x256x96 to output 512x512x32
        xd = self.dconv4(torch.cat([xd, x1], dim=1))

        # input 512x512x48 to output 512x512x16
        xd = self.conv8(torch.cat([xd, x0], dim=1))

        # input 512x512x16 to output 512x512x3
        xd = self.conv9(xd)

        # Residuals
        xd = xd + x
        return xd



class GeneratorWDilationamp(nn.Module):

    def __init__(self,generator,dilation):
        super(GeneratorWDilationamp, self).__init__()
        attributes = inspect.getmembers(generator, lambda a:not(inspect.isroutine(a)))
        attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
        attributes = [a for a in attributes if not(a[0].startswith('_') or a[0].endswith('_'))]
        attributes = [a for a in attributes if (('conv' in a[0])  or ('fc' in a[0]))]
        for layer in attributes:
            #first item of layer is name, second is the object
            setattr(self,layer[0],build_conv_correction(generator,layer[0],dilation ))
        

   # def modify_conv(self):

      

    def forward(self, x):
        # input 512x512x3 to output 512x512x16
        x0 = self.conv1(x)

        # input 512x512x16 to output 256x256x32
        x1 = self.conv2(x0)

        # input 256x256x32 to output 128x128x64
        x2 = self.conv3(x1)

        # input 128x128x64 to output 64x64x128
        x3 = self.conv4(x2)

        # input 64x64x128 to output 32x32x128
        x4 = self.conv5(x3)

        # convolutions for global features
        # input 32x32x128 to output 16x16x128
        x51 = self.conv51(x4)

        # input 16x16x128 to output 8x8x128
        x51 = self.conv52(x51)

        # input 8x8x128 to output 1x1x128
        x53 = self.conv53(x51)
        x53 = self.fc(x53)

        # input 32x32x128 to output 32x32x128
        x5 = self.conv6(x4)


        
        #x53_temp = torch.cat([x53] * (32//x53.shape[2]), dim=2)
        #x53_temp = torch.cat([x53_temp] * (32//x53.shape[2]), dim=3)

        x53_temp = torch.cat([x53] * (x5.shape[2]//x53.shape[2]), dim=2)
        x53_temp = torch.cat([x53_temp] * (x5.shape[2]//x53.shape[2]), dim=3)


        #x53_temp = torch.cat([x53] * x5.shape[2], dim=2)
       # x53_temp = torch.cat([x53_temp] * x5.shape[3], dim=3)
        
       # x53_temp = torch.reshape(x53_temp, (x5.shape[0],-1,x5.shape[2],x5.shape[3]))
       

       
        # input 32x32x256 to output 32x32x128

        x6 = self.conv7(torch.cat([x5, x53_temp], dim=1))

        # input 32x32x128 to output 64x64x128
        xd = self.dconv1(x6)

        # input 64x64x256 to output 128x128x128
        xd = self.dconv2(torch.cat([xd, x3], dim=1))

        # input 128x128x192 to output 256x256x64
        xd = self.dconv3(torch.cat([xd, x2], dim=1))

        # input 256x256x96 to output 512x512x32
        xd = self.dconv4(torch.cat([xd, x1], dim=1))

        # input 512x512x48 to output 512x512x16
        xd = self.conv8(torch.cat([xd, x0], dim=1))

        # input 512x512x16 to output 512x512x3
        xd = self.conv9(xd)

        # Residuals
        xd = xd + x
        return xd

def build_conv_correction(generator,layer_name,rate):
    layer = getattr(generator,layer_name)
    #should check if it is conv and shod check for order, not all modules respect the same order
    #this as is only correct fo0r deconv

    #self.conv4._modules['1'].weight.shape
    #torch.Size([128, 64, 5, 5])



    #self.conv5._modules['1'].weight.shape
    #torch.Size([128, 128, 5, 5])

    #self.dconv2._modules['3'].weight.shape
    #torch.Size([128, 256, 3, 3])

    if  ('conv' in layer_name) or ('fc' in layer_name):
        new_modules = []
        for module in layer.children() :

            if isinstance(module,nn.Conv2d):
                conv_w_tmp = module.weight
                kernel = module.kernel_size[0]
                filters_in = conv_w_tmp.shape[1]
                filters_out = conv_w_tmp.shape[0]

                conv_w = torch.reshape(conv_w_tmp, [ 1,-1, kernel, kernel])
                
               # min_val = conv_w.detach().numpy()[0].min()
                
                #conv_w =  Image.fromarray( (conv_w - min_val).detach().numpy()[0] )
                
               # conv_w = torchvision.transforms.Resize( [rate*(kernel-1)+1, rate*(kernel-1)+1],interpolation=PIL.Image.ANTIALIAS)(conv_w)

                conv_w =  nn.UpsamplingNearest2d(rate*(kernel-1)+1)(conv_w)

                conv_w = torch.reshape(conv_w, [ filters_out,filters_in,rate*(kernel-1)+1, rate*(kernel-1)+1 ])

                conv_w = conv_w * kernel * kernel /( (rate*(kernel-1)+1) * (rate*(kernel-1)+1))

                module.weight = torch.nn.Parameter( conv_w)

                new_modules.append(module)


            else:
                new_modules.append(module)    
        final_modules=[]
        for module in new_modules :

            if  isinstance(module,nn.ReplicationPad2d) :
               
                
                pad_size = rate * (kernel - 1) // 2

                final_modules.append(nn.ReplicationPad2d(pad_size))
            
            else:
                final_modules.append(module)   

        return nn.Sequential(*final_modules)
    else:
        return layer




if __name__=="__main__":
    
    with torch.no_grad():
        device = torch.device('cpu')

        generator = GeneratorWDilation(1)

        #generator = nn.DataParallel(generator)
    
        module_dict=torch.load('/home/felipe/deep-photo-enhancer-master/models/train_checkpoint/2Way/gan2_train_28_40.pth', map_location=device)
    
    # module_dict=torch.load('/home/felipe/deep-photo-enhancer-master/models/train_checkpoint/2Way/gan2_train_92_60.pth')
        
        generator.load_state_dict(module_dict)

        generator = GeneratorWDilationamp(generator,2)



# conv_w = torch.reshape(conv_w_tmp, [conv_w.shape[0], -1, kernel, kernel,])

# conv_w = nn.Upsample(conv_w, [rate*(kernel-1)+1, rate*(kernel-1)+1], mode=tf.image.ResizeMethod.AREA, align_corners=False)

# #torchvision.transforms.Resize( mode=PIL.Image.ANTIALIAS)


# conv_w = torch.reshape(conv_w, [-1, filter,rate*(kernel-1)+1, rate*(kernel-1)+1 ])

# conv_w = conv_w * kernel * kernel / (rate*(kernel-1)+1) * (rate*(kernel-1)+1)