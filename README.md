## A Pytorch Implementation of Deep Photo Enhancer

This project is based on the thesis《Deep Photo Enhancer: Unpaired Learning for Image Enhancement from Photographs with GANs》。

The author's project address is：[nothinglo/Deep-Photo-Enhancer](https://github.com/nothinglo/Deep-Photo-Enhancer)

My code is based on https://github.com/mtics/deep-photo-enhancer

[hyerania\Deep-Learning-Project](https://github.com/hyerania/Deep-Learning-Project)

中文文档说明请看[这里](https://github.com/mtics/deep-photo-enhancer/blob/master/README_zh_cn.md)

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix


The latest code is on the `dev` branch

## Requirements

- Python 3.6
- CUDA 10.0
- To install requirements：
  `pip install -r requirements.txt`

## Prerequisites

### Data

Expert-C on [MIT-Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/)

### Folders

1. All hyperparameters are in `libs\constant.py`


All this extra foders are created by calling

```python directory_strcture.py```


2. There are some folders need to be created, to do that just call python directory_structure.py:
   1. `images_LR`：Used to store datasets
      1. `Expert-C`
      2. `input`
      3. In each of the above two folders, the following three new folders need to be created:
         1. `Testing`
         2. `Training1`
         3. `Training2`
   2. `models`：Used to store all the training generated files：
      1. `gt_images`
      2. `input_images`
      3. `pretrain_checkpoint`
      4. `pretrain_images`
      5. `test_images`
      6. `train_checkpoint`
      7. `train_images`
      8. `train_test_images`
      9. In each of the above folders, the following two new folders need to be created:
         1. `1Way`
         2. `2Way`
   3. `model`: Used to store `log_PreTraining.txt`

## Training

1. 1. `images_LR/input/Training1 ` should contain folders containing images you want to correct.
   2. `images_LR/Expert-C/Training2 ` should contain folders containing  the images of the type you want to obtain.
   3. `images_LR/input/Testing ` `images_LR/Expert-C/Testing ` should contain folders containing the images to get the test scores. In the former you should put the 'bad' sample and on the latter the 'good' sample of the same image 

2. Run the following line to get all the images to max_size 512 ...change the commented directoies as needed
```python reize.py ```

3. Run for training   

```python 2WayGAN_Train_v3.py ```

## Evaluation

For now, the evaluation and training are simultaneous. So there is no need to run anything.

To evaluate my model, I use PSNR in “XWayGAN_Train.py”

## Inference
This is inference for general ized images, it may be improved by using a loader.

```python unbounded_inference.py --generator_model 'gan2_train_28_40.pth' --input_image 'input_image' --output_image  'output_image' ```


## Problem

1. There may be a problem in computing the value of PSNR or not. It needs to be  proved.
2. The compute functions in `libs\compute.py` are wrong, which cause the discriminator loss cannot converge and the output is awful.

## License

This repo is released under  the [MIT License](LICENSE.md)

## Contributor

For now, This repo is maintained by Felipe Bivort Haiek.

Welcome to join me to maintain it together.




