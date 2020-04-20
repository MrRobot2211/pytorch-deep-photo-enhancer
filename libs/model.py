import torch
import torch.nn as nn
import inspect

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

# GENERATOR NETWORK
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #  Convolutional layers
        # input 512x512x3  output 512x512x16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=1, padding=2),
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(16)
        )

        # input 512x512x16  output 256x256x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(32)
        )

        # input 256x256x32  output 128x128x64
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(64)
        )

        # input 128x128x64  output 64x64x128
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(128)
        )

        # input 64x64x128  output 32x32x128
        # the output of this layer we need layers for global features
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 5, stride=2, padding=2),
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(128)
        )

        # convs for global features
        # input 32x32x128 output 16x16x128
        self.conv51 = nn.Conv2d(128, 128, 5, stride=2, padding=2)

        # input 16x16x128 output 8x8x128
        self.conv52 = nn.Conv2d(128, 128, 5, stride=2, padding=2)

        # input 8x8x128 output 1x1x128
        self.conv53 = nn.Conv2d(128, 128, 8, stride=2, padding=0)

        self.fc = nn.Sequential(
            nn.Linear(1, 1),
            nn.SELU(inplace=True),
            nn.Linear(1, 1),
        )

        # input 32x32x128 output 32x32x128
        # the global features should be concatenated to the feature map after this layer
        # the output after concat would be 32x32x256
        self.conv6 = nn.Conv2d(128, 128, 5, stride=1, padding=2)

        # input 32x32x256 output 32x32x128
        self.conv7 = nn.Conv2d(256, 128, 5, stride=1, padding=2)

        # deconvolutional layers
        # input 32x32x128 output 64x64x128
        self.dconv1 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(128),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        )

        # input 64x64x128 ouput 128x128x128
        self.dconv2 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        )

        # input 128x128x192 output 256x256x64
        self.dconv3 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(192),
            nn.ConvTranspose2d(192, 64, 4, stride=2, padding=1)
        )

        # input 256x256x96 ouput 512x512x32
        self.dconv4 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(96),
            nn.ConvTranspose2d(96, 32, 4, stride=2, padding=1)
        )

        # final convolutional layers
        # input 512x512x48 output 512x512x16
        self.conv8 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(48),
            nn.Conv2d(48, 16, 5, stride=1, padding=2)
        )

        # input 512x512x16 output 512x512x3
        self.conv9 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.InstanceNorm2d(16),
            nn.Conv2d(16, 3, 5, stride=1, padding=2)
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
        x52 = self.conv52(x51)

        # input 8x8x128 to output 1x1x128
        x53 = self.conv53(x52)
        x53 = self.fc(x53)
        x53_temp = torch.cat([x53] * 32, dim=2)
        x53_temp = torch.cat([x53_temp] * 32, dim=3)

        # input 32x32x128 to output 32x32x128
        x5 = self.conv6(x4)

        # input 32x32x256 to output 32x32x128
        x5 = self.conv7(torch.cat([x5, x53_temp], dim=1))

        # input 32x32x128 to output 64x64x128
        xd = self.dconv1(x5)

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
def build_shared_layer_new_bn(generator,layer_name):
    layer = getattr(generator,layer_name)
    #should check if it is conv and shod check for order, not all modules respect the same order
    #this as is only correct fo0r deconv

    if (len(layer._modules) == 3 )  and ('conv' in layer_name):
        new_modules = []
        for module in layer.children() :
            if isinstance(module,nn.InstanceNorm2d):
                new_modules.append(nn.InstanceNorm2d(module.num_features))

            else:
              new_modules.append(module)    

        return nn.Sequential(*new_modules)
    else:
        return layer

  

class Generator_(nn.Module):
    def __init__(self,generator):
        super(Generator_, self).__init__()
        attributes = inspect.getmembers(generator, lambda a:not(inspect.isroutine(a)))
        attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
        attributes = [a for a in attributes if not(a[0].startswith('_') or a[0].endswith('_'))]
        attributes = [a for a in attributes if (('conv' in a[0])  or ('fc' in a[0]))]
        for layer in attributes:
            #first item of layer is name, second is the object
            setattr(self,layer[0],build_shared_layer_new_bn(generator,layer[0] ))
        
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
        x52 = self.conv52(x51)

        # input 8x8x128 to output 1x1x128
        x53 = self.conv53(x52)
        x53 = self.fc(x53)
        x53_temp = torch.cat([x53] * 32, dim=2)
        x53_temp = torch.cat([x53_temp] * 32, dim=3)

        # input 32x32x128 to output 32x32x128
        x5 = self.conv6(x4)

        # input 32x32x256 to output 32x32x128
        x5 = self.conv7(torch.cat([x5, x53_temp], dim=1))

        # input 32x32x128 to output 64x64x128
        xd = self.dconv1(x5)

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
       
# DISCRIMINATOR NETWORK
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        #  Convolutional layers
        # input 512x512x3  output 512x512x16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(16)
        )

        # input 512x512x16  output 256x256x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(32)
        )

        # input 256x256x32  output 128x128x64
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(64)
        )

        # input 128x128x64  output 64x64x128
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(128)
        )

        # input 64x64x128  output 32x32x128
        # the output of this layer we need layers for global features
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(128)
        )

        # input 32x32x128  output 16x16x128
        # the output of this layer we need layers for global features
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(128)
        )

        # input 16x16x128  output 1x1x128
        # the output of this layer we need layers for global features
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, 16),
            nn.LeakyReLU(inplace=True)
        )
        self.flat = nn.Flatten()
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # input 512x512x3 to output 512x512x16
        x = self.conv1(x)

        # input 512x512x16 to output 256x256x32
        x = self.conv2(x)

        # input 256x256x32 to output 128x128x64
        x = self.conv3(x)

        # input 128x128x64 to output 64x64x128
        x = self.conv4(x)

        # input 64x64x128 to output 32x32x128
        x = self.conv5(x)

        # input 32x32x128 to output 16x16x128
        x = self.conv6(x)

        # input 16x16x128 to output 1x1x1
        x = self.conv7(x)

        #x = self.flat(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


# BUILDING DATA LOADERS
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
