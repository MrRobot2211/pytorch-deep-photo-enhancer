import torch.optim as optim
from torchvision.utils import save_image
from _datetime import datetime
from libs.compute import *
from libs.constant import *
from libs.model import *

if __name__ == "__main__":

    start_time = datetime.now()

    # Creating generator and discriminator
    generator = Generator()

    #generator = nn.DataParallel(generator)

    generator.load_state_dict(torch.load('./gan1_pretrain_100_14.pth', map_location=device))

    if torch.cuda.is_available():
        generator.cuda(device=device)




    # Loading Training and Test Set Data
    trainLoader1, trainLoader2, trainLoader_cross, testLoader = data_loader()

    for i, (target, input) in enumerate(testLoader, 0):
        unenhanced_image = input[0]
        enhanced_image = target[0]
        enhanced = Variable(enhanced_image.type(Tensor_gpu))

        
        generated_enhanced_image = generator(enhanced)
        

        for k in range(0, generated_enhanced_image.data.shape[0]):
                save_image(generated_enhanced_image.data[k], "./models/pretrain_images/1Way/test_%d_%d.png" % (i + 1, k + 1),
                           nrow=1,
                           normalize=True)
        
            

        