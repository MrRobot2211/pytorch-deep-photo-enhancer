import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch import autograd
from torch.autograd import Variable

from libs.constant import *
from libs.model import *


def data_loader():
    """
    Converting the images for PILImage to tensor,
    so they can be accepted as the input to the network
    :return :
    """
    print("Loading Dataset")
    transform = transforms.Compose([transforms.Resize((SIZE, SIZE), interpolation=2), transforms.ToTensor()])

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
            testset_gt,
            testset_inp
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
    alpha = Tensor_gpu(np.random.random((realSample.shape)))
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
    interpolates = (alpha * real_sample + ((1 - alpha) * fake_sample)).requires_grad_(True)  # stands for y^
    d_interpolation = discriminator(interpolates)  # stands for D_Y(y^)
    fake_output = Variable(Tensor_gpu(real_sample.shape[0], 1).fill_(1.0), requires_grad=False)

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
            max_vals.append(temp_data)
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


def computeIdentityMappingLoss(x, x1, y, y1):
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
    i_loss = criterion(x, y1).mean() + criterion(y, x1).mean()

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

def compute_d_adv_loss(discriminator,discriminatorX, x, x1, y, y1):

    dx = discriminatorX(x)
    dx1 = discriminatorX(x1)
    dy = discriminator(y)
    dy1 = discriminator(y1)

    ad = torch.mean(dx) - torch.mean(dx1) + \
         torch.mean(dy) - torch.mean(dy1)
    return ad

def compute_g_adv_loss(discriminator,discriminatorX, x, x1, y, y1):

    dx = discriminatorX(x)
    dx1 = discriminatorX(x1)
    dy = discriminator(y)
    dy1 = discriminator(y1)

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
    return ad - LAMBDA * penalty


def computeGeneratorLossFor2WayGan(ag, i_loss, c_loss):
    return -ag + ALPHA * i_loss + 10 * ALPHA * c_loss


def adjustLearningRate(learning_rate, decay_rate, epoch_num):
    """
    Adjust Learning rate to get better performance
    :param learning_rate:
    :param decay_rate:
    :param epoch_num:
    :return:
    """
    return learning_rate / (1 + decay_rate * epoch_num)

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
