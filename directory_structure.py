from libs.directory_tools import *


level_list=[['images_LR'],['Expert-C','input'],['Testing','Training1','Training2']]

create_tree(level_list)

level_list=[['models'],[ 'gt_images',
      'input_images',
      'pretrain_checkpoint',
       'pretrain_images',
       'test_images',
       'train_checkpoint',
       'train_images',
       'train_test_images'],['1Way','2Way']]

create_tree(level_list)

level_list = ['model']

create_tree(level_list)