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

    writer1 = tensorboardX.SummaryWriter('./runs/exp-1')

    # Creating generator and discriminator
    generatorX = Generator()
  
    

    #generatorX.load_state_dict(torch.load('./gan1_pretrain_100_14.pth', map_location=device))
    init_net(generatorX,'xavier')
    

    generatorX_ = Generator_(generatorX)

    generatorX = nn.DataParallel(generatorX)

    

    generatorX_ = nn.DataParallel(generatorX_)
    generatorX.train()

    generatorY = Generator()
    init_net(generatorY,'xavier')
    
    #generatorY.load_state_dict(torch.load('./gan1_pretrain_100_14.pth', map_location=device))
    generatorY_ = Generator_(generatorY)
    
    generatorY = nn.DataParallel(generatorY)
   
    generatorY_ = nn.DataParallel(generatorY_)



    generatorY.train()

    

    discriminatorY = Discriminator()
    init_net(discriminatorY)
    discriminatorY = nn.DataParallel(discriminatorY)

    discriminatorX = Discriminator()
    init_net(discriminatorX)
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

    optimizer_g = optim.Adam(itertools.chain(generatorX.parameters(), generatorY.parameters()), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_d = optim.Adam(itertools.chain(discriminatorY.parameters(),discriminatorX.parameters()), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g,70,1e-2)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d,70,1e-2)

    LambdaAdapt = LambdaAdapter(LAMBDA,D_G_ratio)

    


    batches_done = 0
    generator_loss = []
    discriminator_loss = []
    for epoch in range(NUM_EPOCHS_TRAIN):
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

           
            if (LambdaAdapt.netD_change_times_1 > 0 and LambdaAdapt.netD_times >= 0 and LambdaAdapt.netD_times % LambdaAdapt.netD_change_times_1 == 0) or batches_done % 50 == 0: 
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
                
                # i_loss = computeIdentityMappingLoss(realInput, fakeInput, realEnhanced, fakeEnhanced)
                
                i_loss = computeIdentityMappingLoss(generatorX, generatorY, realEnhanced *maskEnhanced,realInput * maskInput)

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

           
            
        

            if batches_done % 500 == 0:
                # Training Network
                psnr=0
                generatorX.eval()
                with torch.no_grad():
                    for j, (data_t, gt1_t) in enumerate(testLoader, 0):

                       
                        input_test, maskInput_test = data_t
                        Testgt, maskEnhanced_test = gt1_t

                        maskInput = Variable(maskInput_test.type(Tensor_gpu)) 
                        maskEnhanced = Variable(maskEnhanced_test.type(Tensor_gpu)) 

                        
                        realInput_test = Variable(input_test.type(Tensor_gpu))
                        realEnhanced_test = Variable(Testgt.type(Tensor_gpu))

                        
                        
                        fakeEnhanced_test = generatorX(realInput_test)
                        test_loss = criterion( realEnhanced_test*maskEnhanced,fakeEnhanced_test*maskInput  )
                        #psnr is okey because answers are from zero to one...we should check clamping in between (?)
                        psnr = psnr + 10 * torch.log10(1 / (test_loss))
                    psnrAvg = psnr/(j+1)

                    print("Loss loss: %f" % test_loss)
                    print("PSNR Avg: %f" % (psnrAvg ))
                    f = open("./models/psnr_Score_trailing.txt", "a+")
                    f.write("PSNR Avg: %f" % (psnrAvg ))
                    f.close()
                    writer1.add_scalar('PSNR test',psnrAvg,batches_done)

                    for k in range(0, realInput_test.data.shape[0]):
                        save_image(realInput_test.data[k], "./models/train_images/2Way/2Way_Train_%d_%d_%d.png" % (epoch+1, batches_done+1, k+1),
                                    nrow=1,
                                    normalize=True)
                    if GPUS_NUM >1:
                        torch.save(generatorX.module.state_dict(),
                                    './models/train_checkpoint/2Way/gan2_train_' + str(epoch) + '_' + str(i) + '.pth')


                        torch.save(discriminatorY.module.state_dict(),
                                    './models/train_checkpoint/2Way/discriminator2_train_' + str(epoch) + '_' + str(i) + '.pth')

                    
                    else:
                        torch.save(generatorX.state_dict(),
                                    './models/train_checkpoint/2Way/gan2_train_' + str(epoch) + '_' + str(i) + '.pth')


                        torch.save(discriminatorY.state_dict(),
                                    './models/train_checkpoint/2Way/discriminator2_train_' + str(epoch) + '_' + str(i) + '.pth')

                    for k in range(0, fakeEnhanced_test.data.shape[0]):
                        save_image(fakeEnhanced_test.data[k],
                                    "./models/train_test_images/2Way/2Way_Train_Test_%d_%d_%d.png" % (epoch, batches_done, k),
                                    nrow=1, normalize=True)
                    
                del fakeEnhanced_test ,realEnhanced_test , realInput_test,  gt1_t, data_t,maskInput_test,maskEnhanced_test ,Testgt, input_test, test_loss
                
                if torch.cuda.is_available() :   
                    torch.cuda.empty_cache()
                else:
                    gc.collect()

                generatorX.train()
            
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
            
            d_loss.backward()

            torch.nn.utils.clip_grad_value_(itertools.chain(discriminatorY.parameters(),discriminatorX.parameters()),clip_value)

            optimizer_d.step()

            batches_done += 1
            LambdaAdapt.netD_times += 1
            scheduler_g.step()
            scheduler_d.step()
            print("Done training discriminator on iteration: %d" % i)

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [ad loss: %f]  [gp1 loss: %f] [gp2 loss: %f][wp1 loss: %f] [wp2 loss: %f]" % (
                epoch + 1, NUM_EPOCHS_TRAIN, i + 1, len(trainLoader_cross), d_loss.item(), g_loss.item(),
                 ad,gradient_penalty1,gradient_penalty2,LambdaAdapt.netD_gp_weight_1,LambdaAdapt.netD_gp_weight_2 ))

            writer1.add_scalars("losses", {'D loss':d_loss.item(), 'G loss':g_loss.item(),
                'ad loss':ad, 'gp1 loss':gradient_penalty1,'gp2 loss':gradient_penalty2,'weight pen 1':LambdaAdapt.netD_gp_weight_1,'weight pen 2':LambdaAdapt.netD_gp_weight_2 }, batches_done)
            

            f = open("./models/log_Train.txt", "a+")
            f.write("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n" % (
                epoch + 1, NUM_EPOCHS_TRAIN, i + 1, len(trainLoader_cross), d_loss.item(), g_loss.item()))
            f.close()
        
            
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

# G_AB = Generator()
# G_BA = Generator()
# D_A = Discriminator()
# D_B = Discriminator()

# batches_done = 0
# prev_time = time.time()
# for epoch in range(opt.n_epochs):
#     for i, batch in enumerate(dataloader):

#         # Configure input
#         imgs_A = Variable(batch["A"].type(FloatTensor))
#         imgs_B = Variable(batch["B"].type(FloatTensor))

#         # ----------------------
#         #  Train Discriminators
#         # ----------------------

#         optimizer_D_A.zero_grad()
#         optimizer_D_B.zero_grad()

#         # Generate a batch of images
#         fake_A = G_BA(imgs_B).detach()
#         fake_B = G_AB(imgs_A).detach()

#         # ----------
#         # Domain A
#         # ----------

#         # Compute gradient penalty for improved wasserstein training
#         gp_A = compute_gradient_penalty(D_A, imgs_A.data, fake_A.data)
#         # Adversarial loss
#         D_A_loss = -torch.mean(D_A(imgs_A)) + torch.mean(D_A(fake_A)) + lambda_gp * gp_A

#         # ----------
#         # Domain B
#         # ----------

#         # Compute gradient penalty for improved wasserstein training
#         gp_B = compute_gradient_penalty(D_B, imgs_B.data, fake_B.data)
#         # Adversarial loss
#         D_B_loss = -torch.mean(D_B(imgs_B)) + torch.mean(D_B(fake_B)) + lambda_gp * gp_B

#         # Total loss
#         D_loss = D_A_loss + D_B_loss

#         D_loss.backward()
#         optimizer_D_A.step()
#         optimizer_D_B.step()

#         if i % opt.n_critic == 0:

#             # ------------------
#             #  Train Generators
#             # ------------------

#             optimizer_G.zero_grad()

#             # Translate images to opposite domain
#             fake_A = G_BA(imgs_B)
#             fake_B = G_AB(imgs_A)

#             # Reconstruct images
#             recov_A = G_BA(fake_B)
#             recov_B = G_AB(fake_A)

#             # Adversarial loss
#             G_adv = -torch.mean(D_A(fake_A)) - torch.mean(D_B(fake_B))
#             # Cycle loss
#             G_cycle = cycle_loss(recov_A, imgs_A) + cycle_loss(recov_B, imgs_B)
#             # Total loss
#             G_loss = lambda_adv * G_adv + lambda_cycle * G_cycle

#             G_loss.backward()
#             optimizer_G.step()

#             # --------------
#             # Log Progress
#             # --------------

#             # Determine approximate time left
#             batches_left = opt.n_epochs * len(dataloader) - batches_done
#             time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / opt.n_critic)
#             prev_time = time.time()

#             sys.stdout.write(
#                 "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, cycle: %f] ETA: %s"
#                 % (
#                     epoch,
#                     opt.n_epochs,
#                     i,
#                     len(dataloader),
#                     D_loss.item(),
#                     G_adv.data.item(),
#                     G_cycle.item(),
#                     time_left,
#                 )
#             )

#         # Check sample interval => save sample if there
#         if batches_done % opt.sample_interval == 0:
#             sample_images(batches_done)

#         batches_done += 1

# def backward_G(self):
#         """Calculate the loss for generators G_A and G_B"""
#         lambda_idt = self.opt.lambda_identity
#         lambda_A = self.opt.lambda_A
#         lambda_B = self.opt.lambda_B
#         # Identity loss
#         if lambda_idt > 0:
#             # G_A should be identity if real_B is fed: ||G_A(B) - B||
#             self.idt_A = self.netG_A(self.real_B)
#             self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
#             # G_B should be identity if real_A is fed: ||G_B(A) - A||
#             self.idt_B = self.netG_B(self.real_A)
#             self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
#         else:
#             self.loss_idt_A = 0
#             self.loss_idt_B = 0

#         # GAN loss D_A(G_A(A))
#         self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
#         # GAN loss D_B(G_B(B))
#         self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
#         # Forward cycle loss || G_B(G_A(A)) - A||
#         self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
#         # Backward cycle loss || G_A(G_B(B)) - B||
#         self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
#         # combined loss and calculate gradients
#         self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
#         self.loss_G.backward()