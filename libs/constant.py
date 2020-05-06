# PARAMETERS
SIZE = 512
BETA1 = 0.9
BETA2 = 0.999
LAMBDA = 10
ALPHA = 1e3 # original FLAGS['loss_constant_term_weight'] or  FLAGS['loss_data_term_weight']
BATCH_SIZE = 3# 5
NUM_EPOCHS_PRETRAIN = 100  # 50
NUM_EPOCHS_TRAIN = 30
LATENT_DIM = 100
LEARNING_RATE = 1e-5
DECAY_RATE = 0.95
INPUT_IMG_DIR = './images_LR/input/Training1/'
INPUT2_IMG_DIR = './images_LR/input/Training2/'
ENHANCED_IMG_DIR ='./images_LR/Expert-C/Training1/'
ENHANCED2_IMG_DIR ='./images_LR/Expert-C/Training2/'
TEST_INPUT_IMG_DIR = './images_LR/input/Testing/'
TEST_ENHANCED_IMG_DIR ='./images_LR/Expert-C/Testing/'