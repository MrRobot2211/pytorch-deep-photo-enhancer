import torch
import torch.optim as optim
from torchvision.utils import save_image
from datetime import datetime
import itertools
from libs.compute import *
from libs.constant import *
from libs.model import *
import gc

import tensorboardX  

# we are missing weight decayed specified in the original as regularization loss
# add cipping the equivalent to tf.clip_by_value  to  torch.clamp(input, 0 , 1 ) !!!!!!verify that we only clamp when applying the inverse!!!!!!!
#add gradient clipping FLAGS['net_gradient_clip_value'] = 1e8    torch.nn.utils.clip_grad_value_
#add he ( keras variance scaling ) init with the apropriate weight debug that this is actually initializing all layers ( because of the use of apply)
#complete loss_wgan_lambda and lambda_grow ...  seting process
#analize tf_crop_rect
# add the missing rounding
clip_value = 1e8 
D_G_ratio = 50

if __name__ == "__main__":

    start_time = datetime.now()
    if continue_checkpoint:
        checkpoint = torch.load(continue_checkpoint)
        #temporary
        #writer1 = checkpoint["summary_writer"] 
        writer1 = tensorboardX.SummaryWriter('./runs/exp-1')
    else:

        writer1 = tensorboardX.SummaryWriter('./runs/exp-1')

    # Creating generator and discriminator
    generatorX = Generator()
  
    

    #generatorX.load_state_dict(torch.load('./gan1_pretrain_100_14.pth', map_location=device))
    
    

    generatorX_ = Generator_(generatorX)

    if continue_checkpoint:
        generatorX.load_state_dict( checkpoint['generatorX'])
        generatorX_.load_state_dict( checkpoint['generatorX_'])
    
    else:
        init_net(generatorX)
        init_net(generatorX_)

    generatorX = nn.DataParallel(generatorX)

    

    generatorX_ = nn.DataParallel(generatorX_)
   
    generatorY = Generator()
    
    
    #generatorY.load_state_dict(torch.load('./gan1_pretrain_100_14.pth', map_location=device))
    generatorY_ = Generator_(generatorY)
    

    if continue_checkpoint:
        generatorY.load_state_dict( checkpoint['generatorY'])
        generatorY_.load_state_dict( checkpoint['generatorY_'])
    else:
        init_net(generatorY)
        init_net(generatorY_)

    generatorY = nn.DataParallel(generatorY)
   
    generatorY_ = nn.DataParallel(generatorY_)


    
    

    discriminatorY = Discriminator()
    
   
    discriminatorX = Discriminator()

    if continue_checkpoint:
        discriminatorX.load_state_dict( checkpoint['discriminatorX'])
        discriminatorY.load_state_dict( checkpoint['discriminatorY'])
    else:
        init_net(discriminatorY)
        init_net(discriminatorX)
        
    
    
    discriminatorY = nn.DataParallel(discriminatorY)
    discriminatorX = nn.DataParallel(discriminatorX)

    if torch.cuda.is_available():
        generatorX.cuda(device=device)
        generatorX_.cuda(device=device)
        generatorY.cuda(device=device)
        generatorY_.cuda(device=device)

        discriminatorY.cuda(device=device)
        discriminatorX.cuda(device=device)

    # Loading Training and Test Set Data
    trainLoader_cross, testLoader = data_loader_mask()

    # MSE Loss and Optimizer
    criterion = nn.MSELoss()

    optimizer_g = optim.Adam(itertools.chain(generatorX.parameters(), generatorY.parameters(),generatorX_.parameters(),generatorY_.parameters()), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_d = optim.Adam(itertools.chain(discriminatorY.parameters(),discriminatorX.parameters()), lr=LEARNING_RATE, betas=(BETA1, BETA2))


    #optimizer_g = optim.SGD(itertools.chain(generatorX.parameters(), generatorY.parameters(),generatorX_.parameters(),generatorY_.parameters()), lr=LEARNING_RATE)
    #optimizer_d = optim.SGD(itertools.chain(discriminatorY.parameters(),discriminatorX.parameters()), lr=LEARNING_RATE)

    if continue_checkpoint:
        optimizer_g.load_state_dict( checkpoint['optimizer_g'])
        optimizer_d.load_state_dict( checkpoint['optimizer_d'])
        scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, adjustLearningRate( 150, 150),last_epoch=checkpoint["epoch"])
        scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, adjustLearningRate( 150, 150),last_epoch=checkpoint["epoch"])
    else:
        scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, adjustLearningRate( 150, 150))
        scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, adjustLearningRate( 150, 150))



    if continue_checkpoint:
        LambdaAdapt = checkpoint["adapter"]
        LambdaAdapt.netD_times=50
    else:
        LambdaAdapt = LambdaAdapter(LAMBDA,D_G_ratio)

    generatorX.train()
    generatorX_.train()
    generatorY.train()
    generatorY_.train()
    discriminatorX.train()
    discriminatorY.train()

    
    generator_loss = []
    discriminator_loss = []
    
    if continue_checkpoint:
        start_epoch = checkpoint['epoch']
        batches_done = checkpoint['batches_done'] 
        #batches_done = checkpoint['batches_done'] 
        #g_loss = checkpoint['g_loss'] 
        #d_loss = checkpoint['d_loss'] 
    else:
        start_epoch=0
        batches_done = 0

    for epoch in range(start_epoch,NUM_EPOCHS_TRAIN):
        for i, (data, gt1) in enumerate(trainLoader_cross, 0):

            input, maskInput= data
            groundTruth, maskEnhanced = gt1
            
            maskInput = Variable(maskInput.type(Tensor_gpu)) 
            maskEnhanced = Variable(maskEnhanced.type(Tensor_gpu)) 

            realInput = Variable(input.type(Tensor_gpu))   # stands for X
            

            realEnhanced = Variable(groundTruth.type(Tensor_gpu))   # stands for Y
            


           
            fakeEnhanced = generatorX(realInput)   # stands for Y'

          

         
            fakeInput = generatorY(realEnhanced)           # stands for x'
          
            
            
         
          
          
           # 
            
           # y2 = generatorX_(x1)          # stands for y''

          
            
        

            if batches_done % 150 == 0:
                # Training Network
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
            
            set_requires_grad([discriminatorY,discriminatorX], True)

            # TRAIN DISCRIMINATOR
           # discriminatorX.zero_grad()
           # discriminatorY.zero_grad()
            optimizer_d.zero_grad()
           
            
            #computing losses
            #ad, ag = computeAdversarialLosses(discriminatorY,discriminatorX, trainInput, x1, realImgs, fake_imgs)
            
            ad = compute_d_adv_loss(discriminatorY,realEnhanced,fakeEnhanced ) + compute_d_adv_loss(discriminatorX,realInput,fakeInput)


            # ad.backward(retain_graph=True)

           

            # gradient_penalty.backward(retain_graph=True)

            gradient_penalty1 =  compute_gradient_penalty(discriminatorY, realEnhanced, fakeEnhanced) 
            gradient_penalty2 =  compute_gradient_penalty(discriminatorX, realInput,fakeInput)

            LambdaAdapt.update_penalty_weights(batches_done ,gradient_penalty1,gradient_penalty2)



            d_loss = computeDiscriminatorLossFor2WayGan(ad, LambdaAdapt.netD_gp_weight_1*gradient_penalty1 + LambdaAdapt.netD_gp_weight_2 * gradient_penalty2)
            
            #d_loss = ad
            d_loss.backward()

            torch.nn.utils.clip_grad_value_(itertools.chain(discriminatorY.parameters(),discriminatorX.parameters()),clip_value)

            optimizer_d.step()

            

             
            if (LambdaAdapt.netD_change_times_1 > 0 and LambdaAdapt.netD_times >= 0 and LambdaAdapt.netD_times % LambdaAdapt.netD_change_times_1 == 0): # or (batches_done % 50 == 0): 
                LambdaAdapt.netD_times = 0
           # if batches_done % 20 == 0:
                recInput = generatorY_(torch.clamp(fakeEnhanced,0,1))     # stands for x''
                recEnhanced = generatorX_(torch.clamp(fakeInput,0,1))   # stands for y''

                set_requires_grad([discriminatorY,discriminatorX], False)
                
                
                # TRAIN GENERATOR
                #generatorX.zero_grad()

                #generatorY.zero_grad()

                optimizer_g.zero_grad()
               

                ag = compute_g_adv_loss(discriminatorY,discriminatorX, fakeEnhanced,fakeInput)
                
                i_loss = computeIdentityMappingLoss_dpeversion(realInput* maskInput, realEnhanced*maskEnhanced, fakeInput*maskEnhanced, fakeEnhanced* maskInput)
                
                #i_loss = computeIdentityMappingLoss(generatorX, generatorY, realEnhanced *maskEnhanced,realInput * maskInput)

                c_loss = computeCycleConsistencyLoss(realInput * maskInput , recInput* maskInput  , realEnhanced*maskEnhanced, recEnhanced*maskEnhanced)

                g_loss = computeGeneratorLossFor2WayGan(ag, i_loss, c_loss)

                #set_requires_grad([discriminatorY,discriminatorX], False)
                # ag.backward(retain_graph=True)
                # i_loss.backward(retain_graph=True)
                # c_loss.backward(retain_graph=True)
                g_loss.backward()

                torch.nn.utils.clip_grad_value_(itertools.chain(generatorX.parameters(), generatorY.parameters()),clip_value)


                optimizer_g.step()
                
                del ag,i_loss,c_loss,recEnhanced,recInput#x2,y2 #,g_loss
                if torch.cuda.is_available() :   
                    torch.cuda.empty_cache()
                else:
                    gc.collect()

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
        scheduler_g.step()
        scheduler_d.step()
           
        
            
    # TEST NETWORK
    batches_done = 0
     # Training Network
    dataiter = iter(testLoader)
    gt_test, data_test = dataiter.next()
    input_test, dummy = data_test
    Testgt, dummy = gt_test
    with torch.no_grad():
        psnrAvg = 0.0
        for j, (data, gt) in enumerate(testLoader, 0):
            input, dummy = data
            groundTruth, dummy = gt
            trainInput = Variable(input.type(Tensor_gpu))
            realImgs = Variable(groundTruth.type(Tensor_gpu))

            output = generatorX(trainInput)
            
            loss = criterion(output, realImgs)
            
            psnr = 10 * torch.log10(1 / loss)
            psnrAvg += psnr

            for k in range(0, output.data.shape[0]):
                save_image(output.data[k],
                           "./models/test_images/2Way/test_%d_%d_%d.png" % (batches_done + 1, j + 1, k + 1),
                           nrow=1,
                           normalize=True)
            for k in range(0, realImgs.data.shape[0]):
                save_image(realImgs.data[k],
                           "./models/gt_images/2Way/gt_%d_%d_%d.png" % (batches_done + 1, j + 1, k + 1),
                           nrow=1,
                           normalize=True)
            for k in range(0, trainInput.data.shape[0]):
                save_image(trainInput.data[k],
                           "./models/input_images/2Way/input_%d_%d_%d.png" % (batches_done + 1, j + 1, k + 1), nrow=1,
                           normalize=True)

            batches_done += 5
            print("Loss loss: %f" % loss)
            print("PSNR Avg: %f" % (psnrAvg / (j + 1)))
            f = open("./models/psnr_Score.txt", "a+")
            f.write("PSNR Avg: %f" % (psnrAvg / (j + 1)))
        f = open("./models/psnr_Score.txt", "a+")
        f.write("Final PSNR Avg: %f" % (psnrAvg / len(testLoader)))
        print("Final PSNR Avg: %f" % (psnrAvg / len(testLoader)))

    end_time = datetime.now()
    print(end_time - start_time)
