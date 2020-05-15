import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.autograd import Variable

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[],original_network=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'dpe':
        
        net = DPEGenerator()
    elif  netG == 'dpe_':
        net = DPEGenerator_(original_network)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'dpe':
        net = DPEDiscriminator()
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


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
    elif type == 'DPEmax':
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps


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
        
        
        return gradient_penalty, gradients
    
    else:
        return 0.0, None






class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class DPEGenerator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #  Convolutional layers
        # input 512x512x3  output 512x512x16
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(3, 16, 5, stride=1, padding=0),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(16)
        )

        # input 512x512x16  output 256x256x32
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(16, 32, 5, stride=2, padding=0),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(32)
        )

        # input 256x256x32  output 128x128x64
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(32, 64, 5, stride=2, padding=0),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(64)
        )

        # input 128x128x64  output 64x64x128
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(64, 128, 5, stride=2, padding=0),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128)
        )

        # input 64x64x128  output 32x32x128
        # the output of this layer we need layers for global features
        self.conv5 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(128, 128, 5, stride=2, padding=0),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128)
        )

        # convs for global features
        # input 32x32x128 output 16x16x128
        self.conv51 = nn.Sequential(
            nn.ReflectionPad2d(2), 
            nn.Conv2d(128, 128, 5, stride=2, padding=0))

        # input 16x16x128 output 8x8x128
        self.conv52 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(128, 128, 5, stride=2, padding=0))

        # input 8x8x128 output 1x1x128
        self.conv53 = nn.Conv2d(128, 128, 8, stride=1, padding=0)
        
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
            nn.Conv2d(128, 128, 1, stride=1, padding=0) )

        # input 32x32x256 output 32x32x128
        self.conv7 = nn.Sequential(  nn.ReflectionPad2d(1), 
            nn.Conv2d(256, 128, 3, stride=1, padding=0) )

        # deconvolutional layers
        # input 32x32x128 output 64x64x128
        self.dconv1 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Upsample(scale_factor=2,  mode='nearest') 
        )

        # input 64x64x128 ouput 128x128x128
        self.dconv2 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.BatchNorm2d(256),
           
            nn.ReflectionPad2d(1),

            nn.Conv2d(256, 128, 3, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # input 128x128x192 output 256x256x96
        self.dconv3 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.BatchNorm2d(192),
            nn.ReflectionPad2d(1),
            nn.Conv2d(192, 64, 3, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # input 256x256x96 ouput 512x512x32
        self.dconv4 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.BatchNorm2d(96),
            nn.ReflectionPad2d(1),
            nn.Conv2d(96, 32, 3, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # final convolutional layers
        # input 512x512x48 output 512x512x16
        self.conv8 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.BatchNorm2d(48),
            #nn.ReflectionPad2d(1),
            nn.Conv2d(48, 16, 1, stride=1, padding=0)
        )

        # input 512x512x16 output 512x512x3
        self.conv9 = nn.Sequential(
            nn.SELU(inplace=True),
            nn.BatchNorm2d(16),
            #nn.ReflectionPad2d(1),
            nn.Conv2d(16, 3, 1, stride=1, padding=0)
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

def build_shared_layer_new_bn(generator,layer_name):
    layer = getattr(generator,layer_name)
    #should check if it is conv and shod check for order, not all modules respect the same order
    #this as is only correct fo0r deconv

    if (len(layer._modules) == 3 )  and ('conv' in layer_name):
        new_modules = []
        for module in layer.children() :
            if isinstance(module,nn.BatchNorm2d):
                new_modules.append(nn.BatchNorm2d(module.num_features))

            else:
              new_modules.append(module)    

        return nn.Sequential(*new_modules)
    else:
        return layer

  

class DPEGenerator_(nn.Module):
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

        #check that the dims are the correct one in the original it is width and height reshaped  by 32 , it is ususal for pytorch to have batch,channel,width,height
        # batch = tf.tile(batch, [h * w])
        # batch = tf.reshape(batch, [h, w, -1])
        
        
        x53_temp = torch.cat([x53] * 32, dim=2)
        x53_temp = torch.cat([x53_temp] * 32, dim=3)


        
        # input 32x32x128 to output 32x32x128
        x5 = self.conv6(x4)

        # input 32x32x256 to output 32x32x128

        #concat_t.set_shape(concat_t.get_shape().as_list()[:3] + [dims])
        #tensor = tf.concat([tensor, concat_t], 3)

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
class DPEDiscriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        #  Convolutional layers
        # input 512x512x3  output 512x512x16
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(3, 16,kernel_size= 3, stride=1, padding=0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.InstanceNorm2d(16,affine=True)
        )

        # input 512x512x16  output 256x256x32
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(16, 32, 5, stride=2, padding=0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.InstanceNorm2d(32,affine=True)
        )

        # input 256x256x32  output 128x128x64
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(32, 64, 5, stride=2, padding=0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.InstanceNorm2d(64,affine=True)
        )

        # input 128x128x64  output 64x64x128
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(64, 128, 5, stride=2, padding=0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.InstanceNorm2d(128,affine=True)
        )

        # input 64x64x128  output 32x32x128
        # the output of this layer we need layers for global features
        self.conv5 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(128, 128, 5, stride=2, padding=0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.InstanceNorm2d(128,affine=True)
        )

        # input 32x32x128  output 16x16x128
        # the output of this layer we need layers for global features
        self.conv6 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(128, 128, 5, stride=2, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(128,affine=True)
        )

        # input 16x16x128  output 1x1x128
        # the output of this layer we need layers for global features
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, 16,stride=1),
            nn.LeakyReLU(0.2,inplace=True)
        )
       # self.flat = nn.Flatten()
        #self.fc = nn.Linear(128, 1)

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
        #x = x.reshape(x.size(0), -1)
        #x = self.fc(x)
        x = x.mean((1,2,3))
        return x



