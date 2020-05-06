import os, sys, gc
import libs.network_infer
import torch.optim as optim
from torchvision.utils import save_image
from _datetime import datetime
from libs.compute import *
from libs.constant import *
from libs.model import *
from libs.network_infer import *



data_max_image_size=16 * 64 * 2
data_image_size=512
data_patch_size=16*64
#data_patch_size=16*32
data_padrf_size=64

# def normalizeImage(img, max_length):
#     #print(current_time() + ', [normalizeImage]')
#     [height, width, channels] = img.shape
#     #print(current_time() + ', original shape = [%d, %d, %d]' % (height, width, channels))
#     max_l = max(height, width)

#     is_need_resize = max_l != data_image_size
#     if is_need_resize:
#         use_gpu = False 
#         if use_gpu and is_downsample:
#             # gpu
#             new_h, new_w, is_normalize = get_normalize_size_shape_method(img, max_length)
#             # if not is_normalize:
#             #     dict_d = [img, new_h, new_w]
#             #     dict_t = [tf_input_img_ori, tf_img_new_h, tf_img_new_w]
#             #     img = sess.run(tf_resize_img, feed_dict={t:d for t, d in zip(dict_t, dict_d)})
#         else:
#             # cpu
#             img = cpu_normalize_image(img, max_length)
#     return img
from PIL import Image
from numpy import array

def get_normalize_size_shape_method(img, max_length):
    [ height, width, channels ] = img.shape
    if height >= width:
        longerSize = height
        shorterSize = width
    else:
        longerSize = width
        shorterSize = height

    scale = float(max_length) / float(longerSize)

    outputHeight = int(round(height*scale))
    outputWidth = int(round(width*scale))
    return outputHeight, outputWidth

def getInputPhoto(file_name):
    print(current_time() + ', [getInputPhoto]: file_name = %s' % (FLAGS['folder_input'] + file_name))
    file_name_without_ext = os.path.splitext(file_name)[0]
    input_img = cv2.imread(FLAGS['folder_input'] + file_name, 1)
    os.remove(FLAGS['folder_input'] + file_name)
    
    if checkValidImg(input_img):
        h, w, _ = input_img.shape
        resize_input_img = normalizeImage(input_img, data_max_image_size) if max(h, w) > data_max_image_size else input_img
        #file_name = file_name_without_ext + FLAGS['data_output_ext']
        
        #cv2.imwrite(FLAGS['folder_input'] + file_name, resize_input_img)

        cv2.imwrite(FLAGS['folder_input'] + file_name_without_ext + '.jpg', resize_input_img)
        os.rename(FLAGS['folder_input'] + file_name_without_ext + '.jpg', FLAGS['folder_input'] + file_name)
        return file_name
    else:
        return None

def processImg(file_in_name, file_out_name):
    #print(current_time() + ', [processImg]: file_name = %s' % (FLAGS['folder_input'] + file_in_name))
    file_out_name_without_ext = os.path.splitext(file_out_name)[0]
    
    input_img = np.array( Image.open(file_in_name))
    
    #resize_input_img = normalizeImage(input_img, data_image_size)
    #resize_input_img, _, _ = random_pad_to_size(resize_input_img, data_image_size, None, True, False)
    #resize_input_img = resize_input_img[None, :, :, :]

    # dict_d = [resize_input_img, 1]
    # dict_t = [test_df.input1_src, test_df.rate]
    # gfeature = sess.run(netG_test_gfeature1, feed_dict={t:d for t, d in zip(dict_t, dict_d)})

    h, w, c = input_img.shape
    rate = int(round(max(h, w) / data_image_size))
    if rate == 0:
        rate = 1


    generator = GeneratorWDilation(1)

    #generator = nn.DataParallel(generator)

    module_dict=torch.load('/home/felipe/deep-photo-enhancer-master/models/train_checkpoint/2Way/gan2_train_28_40.pth', map_location=device)
   # module_dict=torch.load('/home/felipe/deep-photo-enhancer-master/models/train_checkpoint/2Way/gan2_train_92_60.pth')
    generator.load_state_dict(module_dict)
    
    generator = GeneratorWDilationamp(generator,rate)
    generator = nn.DataParallel(generator)

    if torch.cuda.is_available():
        generator.cuda(device=device)

    generator.eval()

    padrf = rate * data_padrf_size
    patch = data_patch_size

    pad_h = 0 if h % patch == 0 else patch - (h % patch)
    pad_w = 0 if w % patch == 0 else patch - (w % patch)
    pad_h = pad_h + padrf if pad_h < padrf else pad_h
    pad_w = pad_w + padrf if pad_w < padrf else pad_w

    input_img = np.pad(input_img, [ (padrf, pad_h),(padrf, pad_w), (0, 0)], 'reflect')
   



    input_img=input_img.transpose((2,0,1))


    input_img =  input_img / 255



    y_list = []
    
    #process for each chunk
    for y in range(padrf, h+padrf, patch):
        x_list = []
        for x in range(padrf, w+padrf, patch):
            
            crop_img = input_img[None,:,y-padrf:y+padrf+patch,  x-padrf:x+padrf+patch]

            
           
            # dict_d = [crop_img, gfeature, rate]
            # dict_t = [test_df.input1_src, test_df.input2, test_df.rate]
           #pad to full image here 
            #enhance_test_img = sess.run(netG_test_dilation_list[min(9, rate-1)], feed_dict={t:d for t, d in zip(dict_t, dict_d)})
            #crop_img = torch.Tensor(crop_img)

            crop_img = Variable(torch.Tensor(crop_img).type(Tensor_gpu)) 
            
            enhance_test_img = generator(crop_img)

            enhance_test_img = enhance_test_img[:,:, padrf:-padrf, padrf:-padrf]
           
            x_list.append(enhance_test_img.detach().cpu())
           # x_list.append(enhance_test_img.detach().cpu().numpy() )

        y_list.append(torch.cat(x_list, axis=3))
        #y_list.append(np.concatenate(x_list, axis=2))
    
    enhance_test_img = torch.cat(y_list, axis=2)
    #enhance_test_img = np.concatenate(y_list, axis=3)



    enhance_test_img = enhance_test_img[:,:,:h, :w]

    #enhance_test_img = safe_casting(enhance_test_img * tf.as_dtype(FLAGS['data_input_dtype']).max, FLAGS['data_input_dtype'])
    
    #enhanced_img_file_name = file_out_name_without_ext + FLAGS['data_output_ext']
    #enhance_img_file_path = folder_test_img + enhanced_img_file_name
    #try:
    #    print(current_time() + ', try remove file path = %s' % enhance_img_file_path)
    #    os.remove(enhance_img_file_path)
    #except OSError as e:
    #    print(current_time() + ', remove fail, error = %s' % e.strerror)
    #cv2.imwrite(enhance_img_file_path, enhance_test_img)




    save_image( enhance_test_img, file_out_name_without_ext + '.jpg')

    #rescaled = (255.0 / enhance_test_img.max() * (enhance_test_img - enhance_test_img.min())).astype(np.uint8)

   # arr = 255 * np.ascontiguousarray(enhance_test_img[0].transpose(1,2,0))
    #img = Image.fromarray(arr, 'RGB')

    #im = Image.fromarray(rescaled)
    #im.save( file_out_name_without_ext + '.jpg')



   # cv2.imwrite( file_out_name_without_ext + '.jpg', enhance_test_img[0])

    #os.rename( file_out_name_without_ext + '.jpg', enhance_img_file_path)

    #return enhanced_img_file_name



if __name__ == "__main__":

   
    processImg('/home/felipe/deep-photo-enhancer-master/images_LR/images-2.jpg', '/home/felipe/deep-photo-enhancer-master/images_LR/images-2inferred.jpg')



        
            