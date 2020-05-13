import numpy as np
import cv2
import torchvision
import torchvision.transforms as transforms
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torchvision.datasets import ImageFolder
import albumentations as albu
from albumentations import torch as AT
import pandas as pd
from libs.custom_transforms import PadDifferentlyIfNeeded
from libs.constant import *
from libs.model import *

#import segmentation_models_pytorch as smp

#smp.encoders.get_preprocessing_fn()


class ImageDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super(ImageDataset, self).__init__( root, transform=transform)

        # def __init__(self, df: pd.DataFrame = None, datatype: str = 'train', img_ids: np.array = None,
        #             transform = None,
        #             preprocessing=None):
        #     self.df = df
        #     if datatype != 'test':
        #     self.data_folder = f"{path}/train_images"
            
        #self.img_ids = img_ids
        #self.transforms = transforms
        #self.preprocessing = preprocessing
    def  _make_mask(self,img,output_size=(512,512)):
        image_width, image_height = output_size
        crop_height, crop_width = img.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        output = np.zeros(output_size)
        output[crop_left:crop_left + crop_width, crop_top:crop_top +crop_width ] = 1
        return output

    def __getitem__(self, index):
        
        path, target = self.samples[index]
        sample = self.loader(path)
        #mask = self._make_mask( sample)
        sample = np.array(sample)
        mask = np.ones(sample.shape[:-1])
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       # augmented = self.transform(image=sample, mask=mask)
        augmented = self.transform(image=sample, mask=mask)
        
        img = augmented['image']
        mask = augmented['mask']

        # if self.preprocessing:
        #     preprocessed = self.preprocessing(image=img, mask=mask)
        #     img = preprocessed['image']
        #     mask = preprocessed['mask']
            
        return img, mask

    # def __len__(self):
    #     return len(self.img_ids)



def data_loader_mask():
    """
    Converting the images for PILImage to tensor,
    so they can be accepted as the input to the network
    :return :
    """
    print("Loading Dataset")
    #transform = transforms.Compose([transforms.Resize((SIZE, SIZE), interpolation='PIL.Image.ANTIALIAS'), transforms.ToTensor()])
    #transform = transforms.Compose([
    # you can add other transformations in this list
   # transforms.CenterCrop(512),
   # transforms.ToTensor()  ])
    default_transform = albu.Compose([ PadDifferentlyIfNeeded(512,512,mask_value=0)
    , AT.ToTensor()])
  
    transform = albu.Compose([ albu.RandomRotate90(1.0)
    , albu.HorizontalFlip(0.5),PadDifferentlyIfNeeded(512,512,mask_value=0), AT.ToTensor()])
  
    testset_gt = ImageDataset(root=TEST_ENHANCED_IMG_DIR , transform=default_transform)
    trainset_1_gt = ImageDataset(root=ENHANCED_IMG_DIR, transform=transform)
    trainset_2_gt = ImageDataset(root=ENHANCED2_IMG_DIR, transform=transform)

    testset_inp = ImageDataset(root=TEST_INPUT_IMG_DIR , transform=default_transform)
    trainset_1_inp = ImageDataset(root=INPUT_IMG_DIR , transform=transform)
    trainset_2_inp = ImageDataset(root=INPUT2_IMG_DIR, transform=transform)

   
    train_loader_cross = torch.utils.data.DataLoader(
        ConcatDataset(
            trainset_1_inp,
            trainset_2_gt
        ),num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        ConcatDataset(
           
            testset_inp,
            testset_gt
        ),num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=False
    )
    print("Finished loading dataset")

    return  train_loader_cross, test_loader















def data_loader():
    """
    Converting the images for PILImage to tensor,
    so they can be accepted as the input to the network
    :return :
    """
    print("Loading Dataset")
    #transform = transforms.Compose([transforms.Resize((SIZE, SIZE), interpolation='PIL.Image.ANTIALIAS'), transforms.ToTensor()])
    transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.CenterCrop(516),
    transforms.ToTensor()  ])
    
    testset_gt = torchvision.datasets.ImageFolder(root='./images_LR/Expert-C/Testing/', transform=transform)
    trainset_1_gt = torchvision.datasets.ImageFolder(root='./images_LR/Expert-C/Training1/', transform=transform)
    trainset_2_gt = torchvision.datasets.ImageFolder(root='./images_LR/Expert-C/Training2/', transform=transform)

    testset_inp = torchvision.datasets.ImageFolder(root='./images_LR/input/Testing/', transform=transform)
    trainset_1_inp = torchvision.datasets.ImageFolder(root='./images_LR/input/Training1/', transform=transform)
    trainset_2_inp = torchvision.datasets.ImageFolder(root='./images_LR/input/Training2/', transform=transform)

    train_loader_1 = torch.utils.data.DataLoader(
        ConcatDataset(
            trainset_1_gt,
            trainset_1_inp
        ),
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=True,
    )

    train_loader_2 = torch.utils.data.DataLoader(
        ConcatDataset(
            trainset_2_gt,
            trainset_2_inp
        ),
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=True,
    )

    train_loader_cross = torch.utils.data.DataLoader(
        ConcatDataset(
            trainset_2_inp,
            trainset_1_gt
        ),
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        ConcatDataset(
            testset_inp,
            testset_gt
        ),
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=True,
    )
    print("Finished loading dataset")

    return train_loader_1, train_loader_2, train_loader_cross, test_loader


# Gradient Penalty
# def compute_gradient_penalty(discriminator, real_sample, fake_sample):
#     """
#     This function used to compute Gradient Penalty
#     The equation is Equation(4) in Chp5
#     :param discriminator: stands for D_Y
#     :param real_sample: stands for Y
#     :param fake_sample: stands for Y'
#     :return gradient_penalty: instead of the global parameter LAMBDA
#     """
#     alpha = Tensor_gpu(np.random.random(real_sample.shape))
#     interpolates = (alpha * real_sample + ((1 - alpha) * fake_sample)).requires_grad_(True)  # stands for y^
#     d_interpolation = discriminator(interpolates)  # stands for D_Y(y^)
#     fake_output = Variable(Tensor_gpu(real_sample.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
#
#     gradients = autograd.grad(
#         outputs=d_interpolation,
#         inputs=interpolates,
#         grad_outputs=fake_output,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True)[0]
#
#     # Use Adaptive weighting scheme
#     # The following codes stand for the Equation(4) in Chp5
#     gradients = gradients.view(gradients.size(0), -1)
#     max_vals = []
#     norm_gradients = gradients.norm(2, dim=1) - 1
#     for i in range(len(norm_gradients)):
#         if norm_gradients[i] > 0:
#             max_vals.append(Variable(norm_gradients[i].type(Tensor)).detach().numpy())
#         else:
#             max_vals.append(0)
#
#     tensor_max_vals = torch.as_tensor(max_vals, dtype=torch.float64, device=device)
#
#     # gradient_penalty = np.mean(max_vals)
#     gradient_penalty = torch.mean(tensor_max_vals)
#     return gradient_penalty


def computeGradientPenaltyFor1WayGAN(D, realSample, fakeSample):
    alpha = torch.rand(realSample.shape[0], 1, device=device)
    interpolates = (alpha * realSample + ((1 - alpha) * fakeSample)).requires_grad_(True)
    dInterpolation = D(interpolates)
    #fakeOutput = Variable(Tensor_gpu(realSample.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
    fakeOutput = Variable(Tensor_gpu(realSample.shape[0], 1).fill_(1.0), requires_grad=False)

    gradients = autograd.grad(
        outputs=dInterpolation,
        inputs=interpolates,
        grad_outputs=fakeOutput,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    ## Use Adadpative weighting scheme
    gradients = gradients.view(gradients.size(0), -1)
    maxVals = []
    normGradients = gradients.norm(2, dim=1) - 1
    for i in range(len(normGradients)):
        if (normGradients[i] > 0):
            maxVals.append(Variable(normGradients[i].type(Tensor)).detach().numpy())
        else:
            maxVals.append(0)

    gradientPenalty = np.mean(maxVals)
    return gradientPenalty
# elif type == 'mixed':
#             alpha = torch.rand(real_data.shape[0], 1, device=device)
#             alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
#             interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
#         else:
#             raise NotImplementedError('{} not implemented'.format(type))
#         interpolatesv.requires_grad_(True)
#         disc_interpolates = netD(interpolatesv)
#         gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
#                                         grad_outputs=torch.ones(disc_interpolates.size()).to(device),
#                                         create_graph=True, retain_graph=True, only_inputs=True)
#         gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
#         gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
#         return gradient_penalty, gradients

def compute_gradient_penalty(discriminator, real_sample, fake_sample):
    """
    This function used to compute Gradient Penalty
    The equation is Equation(4) in Chp5
    :param discriminator: stands for D_Y
    :param real_sample: stands for Y
    :param fake_sample: stands for Y'
    :return gradient_penalty: instead of the global parameter LAMBDA
    """
    alpha = Tensor_gpu(np.random.random(real_sample.shape))
    interpolates = (alpha * real_sample + ((1 - alpha) * fake_sample.detach())).requires_grad_(True)  # stands for y^
    d_interpolation = discriminator(interpolates)  # stands for D_Y(y^)
    #fake_output = Variable(Tensor_gpu(real_sample.shape[0], 1).fill_(1.0), requires_grad=False)
    fake_output = Variable(Tensor_gpu(real_sample.shape[0]).fill_(1.0), requires_grad=False)
    
    gradients = autograd.grad(
        outputs=d_interpolation,
        inputs=interpolates,
        grad_outputs=fake_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    # Use Adaptive weighting scheme
    # The following codes stand for the Equation(4) in Chp5
    gradients = gradients.view(gradients.size(0), -1)
    max_vals = []
    norm_gradients = gradients.norm(2, dim=1) - 1
    for i in range(len(norm_gradients)):
        if norm_gradients[i] > 0:
            # temp_data = Variable(norm_gradients[i].type(Tensor)).detach().item()
            temp_data = Variable(norm_gradients[i].type(Tensor)).item()
            max_vals.append(temp_data )
        else:
            max_vals.append(0)

    tensor_max_vals = torch.tensor(max_vals, dtype=torch.float64, device=device, requires_grad=True)

    # gradient_penalty = np.mean(max_vals)
    gradient_penalty = torch.mean(tensor_max_vals)
    # gradient_penalty.backward(retain_graph=True)
    return gradient_penalty

def _gradient_penalty(self, data, generated_data, gamma=10):
    batch_size = data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1)
    epsilon = epsilon.expand_as(data)


    if self.use_cuda:
        epsilon = epsilon.cuda()

    interpolation = epsilon * data.data + (1 - epsilon) * generated_data.data
    interpolation = Variable(interpolation, requires_grad=True)

    if self.use_cuda:
        interpolation = interpolation.cuda()

    interpolation_logits = self.D(interpolation)
    grad_outputs = torch.ones(interpolation_logits.size())

    if self.use_cuda:
        grad_outputs = grad_outputs.cuda()

    gradients = autograd.grad(outputs=interpolation_logits,
                                inputs=interpolation,
                                grad_outputs=grad_outputs,
                                create_graph=True,
                                retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return self.gamma * ((gradients_norm - 1) ** 2).mean()






def generatorAdversarialLoss(output_images, discriminator):
    """
    This function is used to compute Generator Adversarial Loss
    :param output_images:
    :param discriminator:
    :return: the value of Generator Adversarial Loss
    """
    validity = discriminator(output_images)
    gen_adv_loss = torch.mean(validity)
    return gen_adv_loss


def discriminatorLoss(d1Real, d1Fake, gradPenalty):
    """
    This function is used to compute Discriminator Loss E[D(x)]
    :param d1Real:
    :param d1Fake:
    :param gradPenalty:
    :return:
    """
    return (torch.mean(d1Fake) - torch.mean(d1Real)) + (LAMBDA * gradPenalty)


def computeGeneratorLoss(inputs, outputs_g1, discriminator, criterion):
    """
    This function is used to compute Generator Loss
    :param inputs:
    :param outputs_g1:
    :param discriminator:
    :param criterion:
    :return:
    """
    gen_adv_loss1 = generatorAdversarialLoss(outputs_g1, discriminator)
    i_loss = criterion(inputs, outputs_g1)
    gen_loss = -gen_adv_loss1 + ALPHA * i_loss

    return gen_loss
def computeIdentityMappingLoss(generatorX,generatorY,realEnhanced,realInput):
    
    criterion = nn.MSELoss()
    
    i_loss = criterion(realEnhanced, generatorX(realEnhanced)).mean() + criterion(realInput, generatorY(realInput)).mean()

    return i_loss


def computeIdentityMappingLoss_dpeversion(realInput, realEnhanced, fakeInput, fakeEnhanced):
    """
    This function is used to compute the identity mapping loss
    The equation is Equation(5) in Chp6
    :param x:
    :param x1:
    :param y:
    :param y1:
    :return:
    """
    # MSE Loss and Optimizer
    criterion = nn.MSELoss()
    i_loss = criterion(realInput, fakeEnhanced).mean() + criterion(realEnhanced, fakeInput).mean()

    return i_loss


def computeCycleConsistencyLoss(x, x2, y, y2):
    """
    This function is used to compute the cycle consistency loss
    The equation is Equation(6) in Chp6
    :param x:
    :param x2:
    :param y:
    :param y2:
    :return:
    """
    # MSE Loss and Optimizer
    criterion = nn.MSELoss()
    c_loss = criterion(x, x2).mean() + criterion(y, y2).mean()

    return c_loss


def computeAdversarialLosses(discriminator,discriminatorX, x, x1, y, y1):
    """
    This function is used to compute the adversarial losses
    for the discriminator and the generator
    The equations are Equation(7)(8)(9) in Chp6
    :param discriminator:
    :param x:
    :param x1:
    :param y:
    :param y1:
    :return:
    """

    dx = discriminatorX(x)
    dx1 = discriminatorX(x1)
    dy = discriminator(y)
    dy1 = discriminator(y1)

    ad = torch.mean(dx) - torch.mean(dx1) + \
         torch.mean(dy) - torch.mean(dy1)

    ag = torch.mean(dx1) + torch.mean(dy1)

    return ad, ag

def compute_d_adv_loss(discriminator,real,fake):

    # dx = discriminatorX(x)
    # dx1 = discriminatorX(x1)
    # dy = discriminator(y)
    # dy1 = discriminator(y1)

    # ad = torch.mean(dx) - torch.mean(dx1) + \
    #     + torch.mean(dy) - torch.mean(dy1) 
    ad = torch.mean(discriminator(real)) - torch.mean(discriminator(fake.detach()))

    return ad

def compute_g_adv_loss(discriminatorY,discriminatorX, fakeEnhanced,fakeInput):

   
    dx1 = discriminatorX(fakeInput)
    
    dy1 = discriminatorY(fakeEnhanced)

    ag = torch.mean(dx1) + torch.mean(dy1)

    return ag

   




def computeGradientPenaltyFor2Way(discriminator, x, x1, y, y1):
    """
    This function is used to compute the gradient penalty for 2-Way GAN
    The equations are Equation(10)(11) in Chp6
    :param generator:
    :param discriminator:
    :param x:
    :param x1:
    :param y:
    :param y1:
    :return:
    """
    gradient_penalty = computeGradientPenaltyFor1WayGAN(discriminator, y.data, y1.data) + \
                       computeGradientPenaltyFor1WayGAN(discriminator, x.data, x1.data)

    return gradient_penalty


def computeDiscriminatorLossFor2WayGan(ad, penalty):
    #return -ad + LAMBDA * penalty
    return - ad +  penalty


def computeGeneratorLossFor2WayGan(ag, i_loss, c_loss):
    return -ag + ALPHA * i_loss + 10 * ALPHA * c_loss


def adjustLearningRate( decay_rate, limit_epoch):
    """
    Adjust Learning rate to get better performance
    :param learning_rate:
    :param decay_rate:
    :param epoch_num:
    :return:
    """
    def get_decay(epoch_num):

        if epoch_num <= limit_epoch:
            return 1
        else:
            return 1 - ( 1/decay_rate ) *(epoch_num- limit_epoch)
    
    return get_decay

    

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

# FLAGS['loss_wgan_lambda'] = 10
# def compute_lambda_update(loss_wgan_lambda):
    
#     loss_wgan_lambda_grow = 2.0
#     FLAGS['loss_wgan_lambda_ignore'] = 1
#     FLAGS['loss_wgan_use_g_to_one'] = False
#     FLAGS['loss_wgan_gp_times'] = 1
#     FLAGS['loss_wgan_gp_use_all'] = False
#     loss_wgan_gp_bound = 5e-2
#     loss_wgan_gp_mv_decay = 0.99
#     netD_wgan_gp_mvavg_1 = 0
#     netD_wgan_gp_mvavg_2 = 0
#     netD_gp_weight_1 = FLAGS['loss_wgan_lambda']
#     netD_gp_weight_2 = FLAGS['loss_wgan_lambda']
#     netD_update_buffer_1 = 0
#     netD_change_times_1 = FLAGS['netD_times']
#     netD_update_buffer_2 = 0
#     netD_change_times_2 = FLAGS['netD_times']
#     netD_times = -FLAGS['netD_init_times']
#     FLAGS['netD_times'] = 50
#     FLAGS['netD_times_grow'] = 1
#     FLAGS['netD_buffer_times'] = 50 #it depends on batch size
#     FLAGS['netD_init_times'] = 0

#     if not (epoch * FLAGS['data_train_batch_count'] + iter < FLAGS['loss_wgan_lambda_ignore']):

#         #  gradient penalty 1 and 2   ___ -netD_train_s[-7]   -netD_train_s[-7]  
#             netD_wgan_gp_mvavg_1 = netD_wgan_gp_mvavg_1 * FLAGS['loss_wgan_gp_mv_decay'] + (-netD_train_s[-7] / netD_gp_weight_1) * (1 - FLAGS['loss_wgan_gp_mv_decay'])
#             netD_wgan_gp_mvavg_2 = netD_wgan_gp_mvavg_2 * FLAGS['loss_wgan_gp_mv_decay'] + (-netD_train_s[-6] / netD_gp_weight_2) * (1 - FLAGS['loss_wgan_gp_mv_decay'])

#         if netD_update_buffer_1 == 0 and netD_wgan_gp_mvavg_1 > FLAGS['loss_wgan_gp_bound']:
           
#             netD_gp_weight_1 = netD_gp_weight_1 * FLAGS['loss_wgan_lambda_grow']
#             netD_change_times_1 = netD_change_times_1 * FLAGS['netD_times_grow']
#             netD_update_buffer_1 = FLAGS['netD_buffer_times']
#             netD_wgan_gp_mvavg_1 = 0
        
#         netD_update_buffer_1 = 0 if netD_update_buffer_1 == 0 else netD_update_buffer_1 - 1

#         if netD_update_buffer_2 == 0 and netD_wgan_gp_mvavg_2 > loss_wgan_gp_bound :
           
#             netD_gp_weight_2 = netD_gp_weight_2 * loss_wgan_lambda_grow
#             netD_change_times_2 = netD_change_times_2 * netD_times_grow
#             netD_update_buffer_2 = FLAGS['netD_buffer_times']
#             netD_wgan_gp_mvavg_2 = 0
        
#         netD_update_buffer_2 = 0 if netD_update_buffer_2 == 0 else netD_update_buffer_2 - 1

class LambdaAdapter:
    def __init__(self, lambda_init,D_G_ratio):
        
        self.loss_wgan_gp_bound = 5e-2
        self.loss_wgan_gp_mv_decay = 0.99
        self.netD_wgan_gp_mvavg_1 = 0
        self.netD_wgan_gp_mvavg_2 = 0
        self.netD_update_buffer_1 = 0
        self.netD_update_buffer_2 = 0
        self.netD_gp_weight_1 = lambda_init
        self.netD_gp_weight_2 = lambda_init
        self.netD_times = D_G_ratio
        self.netD_times_grow = 1
        self.netD_buffer_times = 50  #should depend on batch size as the original
        self.netD_change_times_1 = D_G_ratio
        self.netD_change_times_2 = D_G_ratio
        
        self.loss_wgan_lambda_ignore = 1
        self.loss_wgan_lambda_grow = 2.0

    def update_penalty_weights(self,batches_done,gr_penalty1,gr_penalty2):
        # if not (epoch * batch_count + current_iter < 1):
        if not (batches_done < 1):
        #  gradient penalty 1 and 2   ___ -netD_train_s[-7]   -netD_train_s[-7]  
            self.netD_wgan_gp_mvavg_1 = self.netD_wgan_gp_mvavg_1 * self.loss_wgan_gp_mv_decay + (gr_penalty1 / self.netD_gp_weight_1) * (1 - self.loss_wgan_gp_mv_decay)
            self.netD_wgan_gp_mvavg_2 = self.netD_wgan_gp_mvavg_2 * self.loss_wgan_gp_mv_decay + (gr_penalty2 / self.netD_gp_weight_2) * (1 - self.loss_wgan_gp_mv_decay)

        if (self.netD_update_buffer_1 == 0) and (self.netD_wgan_gp_mvavg_1 > self.loss_wgan_gp_bound) :
           
            self.netD_gp_weight_1 = self.netD_gp_weight_1 * self.loss_wgan_lambda_grow
            self.netD_change_times_1 = self.netD_change_times_1 * self.netD_times_grow
            self.netD_update_buffer_1 = self.netD_buffer_times
            self.netD_wgan_gp_mvavg_1 = 0
        
        self.netD_update_buffer_1 = 0 if self.netD_update_buffer_1 == 0 else self.netD_update_buffer_1 - 1

        if (self.netD_update_buffer_2 == 0) and (self.netD_wgan_gp_mvavg_2 > self.loss_wgan_gp_bound) :
           
            self.netD_gp_weight_2 = self.netD_gp_weight_2 * self.loss_wgan_lambda_grow
            self.netD_change_times_2 = self.netD_change_times_2 * self.netD_times_grow
            self.netD_update_buffer_2 = self.netD_buffer_times
            self.netD_wgan_gp_mvavg_2 = 0
        
        self.netD_update_buffer_2 = 0 if self.netD_update_buffer_2 == 0 else self.netD_update_buffer_2 - 1
        
       
def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


   




