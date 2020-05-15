"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        #visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(total_iters)   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

        if batches_done % 150 == 0:
               
                psnr=0
                d_test_loss=0
                generatorX.eval()
                discriminatorY.eval()
                with torch.no_grad():
                    for j, (data_t, gt1_t) in enumerate(testLoader, 0):

                        input_test, maskInput_test = data_t
                        Testgt, maskEnhanced_test = gt1_t

                        maskInput_test = Variable(maskInput_test.type(Tensor_gpu)) 
                        maskEnhanced_test = Variable(maskEnhanced_test.type(Tensor_gpu)) 

                        realInput_test = Variable(input_test.type(Tensor_gpu))
                        realEnhanced_test = Variable(Testgt.type(Tensor_gpu))

                        
                        
                        fakeEnhanced_test = generatorX(realInput_test)
                        test_loss = criterion( realEnhanced_test*maskEnhanced_test,torch.clamp(fakeEnhanced_test,0,1)*maskInput_test  )
                        
                        #psnr is okey because answers are from zero to one...we should check clamping in between (?)
                        
                        psnr = psnr + 10 * torch.log10(1 / (test_loss))
                        d_test_loss = d_test_loss + torch.mean(discriminatorY(fakeEnhanced_test))-torch.mean(discriminatorY(realEnhanced_test))
                    
                    psnrAvg = psnr/(j+1)
                    d_test_lossAvg = d_test_loss /(j+1)

                    print("Loss loss: %f" % test_loss)
                    print("DLoss loss: %f" % d_test_lossAvg)
                    print("PSNR Avg: %f" % (psnrAvg ))
                    f = open("./models/dtest_loss_trailing.txt", "a+")
                    f.write("dtest_loss_Avg: %f" % ( d_test_lossAvg ))
                    f.close()
                    f = open("./models/psnr_Score_trailing.txt", "a+")
                    f.write("PSNR Avg: %f" % (psnrAvg ))
                    f.close()
                    writer1.add_scalar('PSNR test',psnrAvg,batches_done)

                #save exampes of geneted images ion test

                if batches_done % 1200 == 0:
                       
                     
                        for k in range(0, fakeEnhanced_test.data.shape[0]):
                            save_image((fakeEnhanced_test*maskInput_test).data[k],
                                        "./models/train_test_images/2Way/2Way_Train_Test_%d_%d_%d.png" % (epoch, batches_done, k),
                                        nrow=1, normalize=True)
                    
                del fakeEnhanced_test ,realEnhanced_test , realInput_test,  gt1_t, data_t,maskInput_test,maskEnhanced_test ,Testgt, input_test, test_loss
                
                if torch.cuda.is_available() :   
                    torch.cuda.empty_cache()
                else:
                    gc.collect()

                generatorX.train()
                discriminatorY.train()


            if batches_done % 1200 == 0:
                       
                        if GPUS_NUM >1:
                            
                            torch.save({'generatorX':generatorX.module.state_dict(),'generatorX_':generatorX_.module.state_dict(),
                            'generatorY':generatorY.module.state_dict(),'generatorY_':generatorY_.module.state_dict(),
                            'discriminatorY':discriminatorY.module.state_dict(),'discriminatorX':discriminatorX.module.state_dict(),
                            'optimizer_g':optimizer_g.state_dict(),'optimizer_d':optimizer_d.state_dict(),
                            'adapter':LambdaAdapt, 
                            'g_loss':g_loss,'d_loss':d_loss,
                            #'summary_writer': writer1,
                                'epoch':epoch,'batches_done':batches_done},'./models/train_checkpoint/2Way/full_train_' + str(epoch) + '_' + str(i) + '.pth')
                        else:
                            torch.save({'generatorX':generatorX.state_dict(),'generatorX_':generatorX_.state_dict(),
                            'generatorY':generatorY.state_dict(),'generatorY_':generatorY_.state_dict(),
                            'discriminatorY':discriminatorY.state_dict(),'discriminatorX':discriminatorX.state_dict(),
                            'optimizer_g':optimizer_g.state_dict(),'optimizer_d':optimizer_d.state_dict(),
                            'adapter':LambdaAdapt, 
                            'g_loss':g_loss,'d_loss':d_loss,
                            #'summary_writer': writer1,
                                'epoch':epoch,'batches_done':batches_done},'./models/train_checkpoint/2Way/full_train_' + str(epoch) + '_' + str(i) + '.pth')
                    
            batches_done += 1
            LambdaAdapt.netD_times += 1
        
            print("Done training discriminator on iteration: %d" % i)

            print("[Epoch %d/%d] [Batch %d/%d] [lr: %f] [D loss: %f] [G loss: %f] [ad loss: %f]  [gp1 loss: %f] [gp2 loss: %f][wp1 loss: %f] [wp2 loss: %f] " % (
                epoch + 1, NUM_EPOCHS_TRAIN, i + 1,len(trainLoader_cross),scheduler_g.get_last_lr()[0] , d_loss.item(), g_loss.item(),
                 ad,gradient_penalty1,gradient_penalty2,LambdaAdapt.netD_gp_weight_1,LambdaAdapt.netD_gp_weight_2 ))

            writer1.add_scalars("losses", {'D loss':d_loss.item(), 'G loss':g_loss.item(),
                'ad loss':ad, 'gp1 loss':gradient_penalty1,'gp2 loss':gradient_penalty2,'weight pen 1':LambdaAdapt.netD_gp_weight_1,'weight pen 2':LambdaAdapt.netD_gp_weight_2 }, batches_done)
            

            f = open("./models/log_Train.txt", "a+")
            f.write("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n" % (
                epoch + 1, NUM_EPOCHS_TRAIN, i + 1, len(trainLoader_cross), d_loss.item(), g_loss.item()))
            f.close()
