import torch
import networks
from geo_process_layer import geo_projection
from utils import ImagePool
from scipy.ndimage.morphology import binary_closing, binary_erosion
import cv2
import numpy

from torch.autograd import Variable
import torchvision
from torchvision import transforms, utils


class R2DModel:
    def name(self):
        return 'R2DModel'

    def initialize(self, opt):
        self.direction = opt.direction
        self.is_train = opt.is_train
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.checkpoints_dir
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')   
        self.lambda_L1 = opt.lambda_L1

        # load/define networks
        #self.netG = networks.define_X(1, 3, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG = networks.define_G(3, 2, opt.ngf, opt.netG, opt.norm_G_D, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.is_train:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)


    def set_input(self, input):
        self.real_R = input['R'].to(self.device)
        if self.is_train:
            self.real_D = input['D'].to(self.device)
            self.real_L = input['L'].to(self.device)
        self.img_id = input['img_id']

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        self.fake_DL = self.netG(self.real_R)
        self.fake_L = self.fake_DL[:,0:1,:,:]
        self.fake_D = self.fake_DL[:,1:2,:,:]

    def backward_G(self):
        # Second, G(A) = B
        self.loss_G_Loss1 = self.criterionL1(self.fake_D, self.real_D)
        self.loss_G_Loss2 = self.criterionL1(self.fake_L, self.real_L)

        self.loss_G = self.loss_G_Loss1 + self.loss_G_Loss2 * 0.1 

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def save_networks(self, epoch):
        torch.save(self.netG.state_dict(), self.save_dir +'/model_G_'+str(epoch)+'.pt')
        torch.save(self.netG.state_dict(), self.save_dir +'/model_G_latest.pt') 

    # load models from the disk
    def load_networks(self, epoch):
        if epoch >= 0:
            self.netG.load_state_dict(torch.load(self.save_dir +'/model_G_'+str(epoch)+'.pt',
            map_location=lambda storage, loc: storage.cuda(0)))       
        else:
            self.netG.load_state_dict(torch.load(self.save_dir +'/model_G_latest.pt',
            map_location=lambda storage, loc: storage.cuda(0)))


class D2LModel:
    def name(self):
        return 'D2LModel'

    def initialize(self, opt):
        self.direction = opt.direction
        self.is_train = opt.is_train
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.checkpoints_dir
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')   
        self.lambda_L1 = opt.lambda_L1
        self.fine_tune_sidewalk = opt.fine_tune_sidewalk

        # load/define networks
        self.netG = networks.define_G(2, 3, opt.ngf, opt.netG, opt.norm_G_D, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.is_train:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input, id_iter):
        self.real_D = input['D'].to(self.device)
        self.real_L = input['L'].to(self.device)
        if self.is_train:
            self.real_S = input['S'].to(self.device)
            self.mask = input['M'].to(self.device)
            if self.fine_tune_sidewalk:
                self.mask_swalk = input['W'].to(self.device)
        self.img_id = input['img_id']

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        self.fake_S = self.netG(torch.cat((self.real_D, self.real_L), 1))
        
    def backward_G(self):
        # self.dev = torch.abs(self.fake_S - self.real_S)
        # self.loss_G_L1 = self.dev + torch.mul(self.dev, self.mask)*10
        # self.loss_G_L1 = torch.mean(self.loss_G_L1)
        # self.loss_G = self.loss_G_L1
        self.loss_G = self.criterionL1(self.fake_S, self.real_S)

        if self.fine_tune_sidewalk:
            print("1111111")
            self.loss_G_L2 = torch.mul(self.dev, self.mask_swalk)
            self.loss_G_L2 = torch.mean(self.loss_G_L2)
            self.loss_G = self.loss_G_L1 + self.loss_G_L2
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def save_networks(self, epoch):
        torch.save(self.netG.state_dict(), self.save_dir +'/model_G_'+str(epoch)+'.pt')
        torch.save(self.netG.state_dict(), self.save_dir +'/model_G_latest.pt') 

    # load models from the disk
    def load_networks(self, epoch):
        if epoch >= 0:
            self.netG.load_state_dict(torch.load(self.save_dir +'/model_G_'+str(epoch)+'.pt',
            map_location=lambda storage, loc: storage.cuda(0)))        
        else:
            self.netG.load_state_dict(torch.load(self.save_dir +'/model_G_latest.pt',
            map_location=lambda storage, loc: storage.cuda(0)))


class L2RModel:
    def name(self):
        return 'L2RModel'

    def initialize(self, opt):
        self.direction = opt.direction
        self.is_train = opt.is_train
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.checkpoints_dir
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')   
        self.lambda_L1 = opt.lambda_L1

        # load/define networks
        self.netG = networks.define_G(3, 3, opt.ngf, opt.netG, opt.norm_G_D, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.is_train:
            self.netD = networks.define_D(6, opt.ndf, opt.netD,
                                          opt.n_layers_d, opt.norm_G_D, opt.no_lsgan, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.is_train:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_L = input['L'].to(self.device)
        self.proj_R = input['Proj_R'].to(self.device)
        if self.is_train:
            self.real_R = input['R'].to(self.device)
        self.img_id = input['img_id']

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        #self.fake_R = self.netG(torch.cat((self.real_L, self.proj_R), 1))
        self.fake_R = self.netG(self.real_L)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_LR = torch.cat((self.real_L, self.fake_R), 1)
        pred_fake = self.netD(fake_LR.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_LR = torch.cat((self.real_L, self.real_R), 1)
        pred_real = self.netD(real_LR)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_LR = torch.cat((self.real_L, self.fake_R), 1)
        pred_fake = self.netD(fake_LR)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_Loss = self.criterionL1(self.real_R, self.fake_R) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_Loss
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad([self.netD, self.netG], False)
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.netD, self.netG], False)
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def save_networks(self, epoch):
        torch.save(self.netG.state_dict(), self.save_dir +'/model_G_'+str(epoch)+'.pt')
        torch.save(self.netG.state_dict(), self.save_dir +'/model_G_latest.pt') 
        torch.save(self.netD.state_dict(), self.save_dir +'/model_D_'+str(epoch)+'.pt')
        torch.save(self.netD.state_dict(), self.save_dir +'/model_D_latest.pt') 

    # load models from the disk
    def load_networks(self, epoch):
        if epoch >= 0:
            self.netG.load_state_dict(torch.load(self.save_dir +'/model_G_'+str(epoch)+'.pt',
            map_location=lambda storage, loc: storage.cuda(0)))
            if self.is_train:
                self.netD.load_state_dict(torch.load(self.save_dir +'/model_D_'+str(epoch)+'.pt',
                map_location=lambda storage, loc: storage.cuda(0)))            
        else:
            self.netG.load_state_dict(torch.load(self.save_dir +'/model_G_latest.pt',
            map_location=lambda storage, loc: storage.cuda(0)))
            if self.is_train:
                self.netD.load_state_dict(torch.load(self.save_dir +'/model_D_latest.pt',
                map_location=lambda storage, loc: storage.cuda(0)))  




















class L2RAllModel:
    def name(self):
        return 'L2RAllModel'

    def initialize(self, opt):
        # Added
        self.num_classes = opt.num_classes
        self.use_multiple_G = opt.use_multiple_G
        self.use_sate = opt.use_sate
        self.sate_encoder_nc = opt.sate_encoder_nc

        self.direction = opt.direction
        self.is_train = opt.is_train
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.checkpoints_dir
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')   
        self.lambda_L1 = opt.lambda_L1

        if self.use_sate:
            input_nc = 3 + 1 + self.num_classes + self.sate_encoder_nc
        else:
            input_nc = 3 + 1 + self.num_classes

        # load/define networks
        if self.use_multiple_G:
            self.netGs = []
            for i in range(self.num_classes):
                self.netGs.append(networks.define_G(input_nc, 3, opt.ngf, opt.netG,
                    opt.norm_G_D, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids))
        else:
            self.netG = networks.define_G(input_nc, 3, opt.ngf, opt.netG, opt.norm_G_D, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.use_sate:
            self.netE = networks.define_E(3, self.sate_encoder_nc, n_downsampling=5, ngf=32, net_type='resnet_6blocks')

        if self.is_train:
            self.netDs = []
            for i in range(self.num_classes):
                self.netDs.append(networks.define_D(4, opt.ndf, opt.netD,
                                          opt.n_layers_d, opt.norm_G_D, opt.no_lsgan, opt.init_type, opt.init_gain, self.gpu_ids))

        if self.is_train:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss(reduction='sum')

            # initialize optimizers
            self.optimizers = []
            if self.use_multiple_G:
                self.optimizer_Gs = []
                for i in range(self.num_classes):
                    self.optimizer_Gs.append(torch.optim.Adam(self.netGs[i].parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999)))
                self.optimizers.extend(self.optimizer_Gs)
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_G)

            self.optimizer_Ds = []
            for i in range(self.num_classes):
                self.optimizer_Ds.append(torch.optim.Adam(self.netDs[i].parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999)))
            self.optimizers.extend(self.optimizer_Ds)

    def set_input(self, input):
        self.real_semantic = input['street_label']
        self.g_input_rgbd = torch.cat([input['proj_rgb'], input['proj_depth']], 1).to(self.device)
        self.g_input_label = torch.nn.functional.one_hot(input['street_label'], num_classes=self.num_classes).float().to(self.device)
        self.g_masks = [(input['street_label'] == i).float().to(self.device) for i in range(self.num_classes)]
        if self.use_sate:
            self.e_input_rgb = input['sate_rgb'].to(self.device)
        # print('Mask:')
        # print('  ', self.real_semantic.shape, self.real_semantic.dtype)
        # print('  ', torch.min(self.real_semantic).item(), torch.max(self.real_semantic).item())
        # for i in range(self.g_masks[0].shape[0]):
        #     for mask in self.g_masks:
        #         print(int(torch.sum(mask[i]).item()), end=' ')
        #     print()

        self.img_id = input['img_id']
        if self.is_train:
            self.g_output_gt = input['street_rgb'].to(self.device)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        if self.use_sate:
            h, w = self.g_input_label.shape[2:4]
            sate_ft = self.E(self.e_input_rgb)
            self.sate_ft = torch.mean(sate_ft, dim=(2,3), keepdim=True)
            self.sate_ft = self.sate_ft.repeat(1, 1, h, w)
            self.g_input = torch.cat([self.g_input_rgbd, self.g_input_label, self.sate_ft], 1)
        else:
            self.g_input = torch.cat([self.g_input_rgbd, self.g_input_label], 1)

        if self.use_multiple_G:
            self.g_outputs = []
            for i in range(self.num_classes):
                self.g_outputs.append(self.netGs[i](self.g_input))
        else:
            self.g_output = self.netG(self.g_input)

    def backward_D(self):
        self.loss_Ds = []
        for i in range(self.num_classes):
            mask = self.g_masks[i]
            mask_3 = torch.cat([mask, mask, mask], 1)

            # Fake
            # stop backprop to the generator by detaching fake_B
            if self.use_multiple_G:
                masked_fake = mask_3 * self.g_outputs[i]
            else:
                masked_fake = mask_3 * self.g_output

            fake = torch.cat([mask, masked_fake], 1)
            pred_fake = self.netDs[i](fake.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)

            # Real
            masked_real = mask_3 * self.g_output_gt
            real = torch.cat([mask, masked_real], 1)
            pred_real = self.netDs[i](real)
            loss_D_real = self.criterionGAN(pred_real, True)

            # Combined loss
            loss = (loss_D_fake + loss_D_real) * 0.5
            self.loss_Ds.append(loss)

        self.loss_D = torch.sum(torch.stack(self.loss_Ds))
        self.loss_D.backward()


    def backward_G(self):
        self.loss_Gs = []
        for i in range(self.num_classes):
            # First, G(A) should fake the discriminator
            mask = self.g_masks[i]
            mask_sum = torch.sum(mask)
            mask_3 = torch.cat([mask, mask, mask], 1)

            if self.use_multiple_G:
                masked_fake = mask_3 * self.g_outputs[i]
            else:
                masked_fake = mask_3 * self.g_output

            fake = torch.cat([mask, masked_fake], 1)
            pred_fake = self.netDs[i](fake)
            loss_G_GAN = self.criterionGAN(pred_fake, True)

            # Second, G(A) = B
            masked_real = mask_3 * self.g_output_gt
            loss_G_Loss = self.criterionL1(masked_real, masked_fake) * self.lambda_L1
            loss_G_Loss = loss_G_Loss / torch.max(mask_sum, torch.ones_like(mask_sum))
            # print('Sum of mask', mask_sum.item(), mask_sum.dtype)

            loss_G = loss_G_GAN + loss_G_Loss
            self.loss_Gs.append(loss_G)

        self.loss_G = torch.sum(torch.stack(self.loss_Gs))
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        if self.use_multiple_G:
            self.set_requires_grad(self.netGs, False)
        else:
            self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netDs, True)
        for i in range(self.num_classes):
            self.optimizer_Ds[i].zero_grad()
        self.backward_D()
        for i in range(self.num_classes):
            self.optimizer_Ds[i].step()

        # update G
        self.set_requires_grad(self.netDs, False)
        if self.use_multiple_G:
            self.set_requires_grad(self.netGs, True)
            for i in range(self.num_classes):
                self.optimizer_Gs[i].zero_grad()
            self.backward_G()
            for i in range(self.num_classes):
                self.optimizer_Gs[i].step()
        else:
            self.set_requires_grad(self.netG, True)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

    def save_networks(self, epoch):
        if self.use_multiple_G:
            for i in range(self.num_classes):
                torch.save(self.netGs[i].state_dict(), self.save_dir +'/model_G%d_'%i+str(epoch)+'.pt')
                torch.save(self.netGs[i].state_dict(), self.save_dir +'/model_G%d_latest.pt'%i)
        else:
            torch.save(self.netG.state_dict(), self.save_dir +'/model_G_'+str(epoch)+'.pt')
            torch.save(self.netG.state_dict(), self.save_dir +'/model_G_latest.pt')
        for i in range(self.num_classes):
            torch.save(self.netDs[i].state_dict(), self.save_dir +'/model_D%d_'%i+str(epoch)+'.pt')
            torch.save(self.netDs[i].state_dict(), self.save_dir +'/model_D%d_latest.pt'%i) 

    # load models from the disk
    def load_networks(self, epoch):
        if epoch >= 0:
            if self.use_multiple_G:
                for i in range(self.num_classes):
                    self.netGs[i].load_state_dict(torch.load(self.save_dir +'/model_G%s_'%i+str(epoch)+'.pt',
                    map_location=lambda storage, loc: storage.cuda(0)))
            else:
                self.netG.load_state_dict(torch.load(self.save_dir +'/model_G_'+str(epoch)+'.pt',
                map_location=lambda storage, loc: storage.cuda(0)))
            if self.is_train:
                for i in range(self.num_classes):
                    self.netDs[i].load_state_dict(torch.load(self.save_dir +'/model_D%d_'%i+str(epoch)+'.pt',
                    map_location=lambda storage, loc: storage.cuda(0)))            
        else:
            if self.use_multiple_G:
                for i in range(self.num_classes):
                    self.netGs[i].load_state_dict(torch.load(self.save_dir +'/model_G%s_latest.pt'%i,
                    map_location=lambda storage, loc: storage.cuda(0)))
            else:
                self.netG.load_state_dict(torch.load(self.save_dir +'/model_G_latest.pt',
                map_location=lambda storage, loc: storage.cuda(0)))
            if self.is_train:
                for i in range(self.num_classes):
                    self.netDs[i].load_state_dict(torch.load(self.save_dir +'/model_D%d_latest.pt'%i,
                    map_location=lambda storage, loc: storage.cuda(0)))  

























###########
class DLRModel:
    def name(self):
        return 'DLRModel'

    def initialize(self, opt):
        self.direction = opt.direction
        self.sate_gsd = opt.sate_gsd
        self.pano_size = opt.pano_size
        self.is_train = opt.is_train
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.checkpoints_dir
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.lambda_L1 = opt.lambda_L1

        # the GAN input projected depth+label, output streetview label
        self.netG1 = networks.define_G(2, 3, opt.ngf, opt.netG, opt.norm_G_D,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # the GAN input streetview label, output streetview rgb
        self.netG2 = networks.define_G(3, 3, opt.ngf, opt.netG, opt.norm_G_D,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.is_train:
            # the discriminor for streetview rgb 
            self.netD = networks.define_D(3+3, opt.ndf, opt.netD,
                                          opt.n_layers_d, opt.norm_G_D, opt.no_lsgan, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.is_train:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))          
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))            
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input, id_iter):
        self.proj_D = input['proj_D'].to(self.device)
        self.proj_L = input['proj_L'].to(self.device)
        self.proj_R = input['proj_R'].to(self.device)
        if self.is_train:
            self.real_R = input['real_R'].to(self.device)
            self.real_L = input['real_L'].to(self.device)
            self.mask = input['M'].to(self.device)
            self.mask_swalk = input['W'].to(self.device)

        self.img_id = input['img_id']        
        self.id_iter = id_iter

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        self.fake_L = self.netG1(torch.cat((self.proj_D, self.proj_L), 1))
        #self.fake_R = self.netG2(torch.cat((self.fake_L, self.proj_R), 1))
        self.fake_R = self.netG2(self.fake_L)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_LR = torch.cat((self.real_L, self.fake_R), 1)
        pred_fake = self.netD(fake_LR.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_LR = torch.cat((self.real_L, self.real_R), 1)
        pred_real = self.netD(real_LR)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # First, G(D) should fake the discriminator
        fake_LR = torch.cat((self.real_L, self.fake_R), 1)
        pred_fake = self.netD(fake_LR)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G1(DL) = L
        self.loss_G_L1 = self.criterionL1(self.fake_L, self.real_L) * self.lambda_L1
        self.dev = torch.abs(self.fake_L - self.real_L)
        self.loss_G_L11 = torch.mul(self.dev, self.mask)
        self.loss_G_L11 = torch.mean(self.loss_G_L11)
        #self.loss_G_L12 = torch.mul(self.dev, self.mask_swalk)
        #self.loss_G_L12 = torch.mean(self.loss_G_L12)
        #self.loss_G_L1 = (self.loss_G_L11 + self.loss_G_L12) * self.lambda_L1
        self.loss_G_L1 = (10*self.loss_G_L11 + torch.mean(self.dev) )

        # third, G2(L) = R
        self.dev2 = torch.abs(self.fake_R - self.real_R)
        self.loss_G_L21 = torch.mul(self.dev2, self.mask)
        self.loss_G_L21 = torch.mean(self.loss_G_L21) * self.lambda_L1
        self.loss_G_L2 = (10*self.loss_G_L21 + torch.mean(self.dev2) )

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_L2
        #self.loss_G.backward(retain_graph=True)       
        self.loss_G.backward()        

    def optimize_parameters(self):
        self.forward()

        # update D
        self.set_requires_grad([self.netG1, self.netG2, self.netD], False)
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.netG1, self.netG2, self.netD], False)
        #self.set_requires_grad([self.netG1, self.netG2], True)
        self.set_requires_grad(self.netG2, True)
        #self.optimizer_G1.zero_grad()        
        self.optimizer_G2.zero_grad()        
        self.backward_G()            
        #self.optimizer_G1.step()      
        self.optimizer_G2.step()              

    def save_networks(self, epoch):
        torch.save(self.netG1.state_dict(), self.save_dir +'/model_G1_'+str(epoch)+'.pt')
        torch.save(self.netG1.state_dict(), self.save_dir +'/model_G1_latest.pt')         
        torch.save(self.netG2.state_dict(), self.save_dir +'/model_G2_'+str(epoch)+'.pt')
        torch.save(self.netG2.state_dict(), self.save_dir +'/model_G2_latest.pt') 
        torch.save(self.netD.state_dict(), self.save_dir +'/model_D_'+str(epoch)+'.pt')
        torch.save(self.netD.state_dict(), self.save_dir +'/model_D_latest.pt') 

    # load models from the disk
    def load_networks(self, epoch):
        if epoch >= 0:
            self.netG1.load_state_dict(torch.load(self.save_dir +'/model_G1_'+str(epoch)+'.pt',
            map_location=lambda storage, loc: storage.cuda(0)))            
            self.netG2.load_state_dict(torch.load(self.save_dir +'/model_G2_'+str(epoch)+'.pt',
            map_location=lambda storage, loc: storage.cuda(0)))                     
            if self.is_train:
                self.netD.load_state_dict(torch.load(self.save_dir +'/model_D_'+str(epoch)+'.pt',
                map_location=lambda storage, loc: storage.cuda(0)))  
                      
        else:
            self.netG1.load_state_dict(torch.load(self.save_dir +'/model_G1_latest.pt',
            map_location=lambda storage, loc: storage.cuda(0)))            
            self.netG2.load_state_dict(torch.load(self.save_dir +'/model_G2_latest.pt',
            map_location=lambda storage, loc: storage.cuda(0)))
            if self.is_train:
                self.netD.load_state_dict(torch.load(self.save_dir +'/model_D_latest.pt',
                map_location=lambda storage, loc: storage.cuda(0)))   


class RDLRModel:
    def name(self):
        return 'RDLRModel'

    def initialize(self, opt):
        self.direction = opt.direction
        self.sate_gsd = opt.sate_gsd
        self.pano_size = opt.pano_size
        self.is_train = opt.is_train
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.checkpoints_dir
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.lambda_L1 = opt.lambda_L1

        self.netG0 = networks.define_G(3, 2, opt.ngf, opt.netG, opt.norm_G_D,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # the GAN input projected depth+label, output streetview label
        self.netG1 = networks.define_G(2, 3, opt.ngf, opt.netG, opt.norm_G_D,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # the GAN input streetview label, output streetview rgb
        self.netG2 = networks.define_G(3, 3, opt.ngf, opt.netG, opt.norm_G_D,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.is_train:
            # the discriminor for streetview rgb 
            self.netD = networks.define_D(3+3, opt.ndf, opt.netD,
                                          opt.n_layers_d, opt.norm_G_D, opt.no_lsgan, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.is_train:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G0 = torch.optim.Adam(self.netG1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))            
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))          
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))            
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G0)
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input, id_iter):
        self.sate_R = input['sate_R'].to(self.device)
        if self.is_train:
            self.sate_D = input['sate_D'].to(self.device)
            self.sate_L = input['sate_L'].to(self.device)      
            self.street_R = input['street_R'].to(self.device)
            self.street_L = input['street_L'].to(self.device)
        self.img_id = input['img_id']        
        self.ori = input['ori']
        self.id_iter = id_iter

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def shape_fill(self):
        sem = self.fake_street_L.detach().cpu().numpy()
        sem = (sem*0.5+0.5)*255
        [n,d,w,h] = sem.shape
        sem_rgb = sem[:,0,:,:] + sem[:,1,:,:] + sem[:,2,:,:]

        sem_filled = numpy.zeros([n,d,w,h]).astype(numpy.float)

        # filling
        nsize = 1
        th_pixle = 30
        for i in range(0, n):
            # sky
            mask_sky = abs(sem_rgb[i,:,:]-380)<th_pixle
            mask_sky = binary_closing(mask_sky)
            mask_sky = numpy.asarray(mask_sky).astype(numpy.uint8)
            [h,w] = mask_sky.shape
            mask_sky[int(h/2):int(h),:] = 0
            mask_sky = binary_erosion(mask_sky, structure=numpy.ones((nsize,nsize))).astype(mask_sky.dtype)

            sem_filled[i,0,:,:] = sem[i,0,:,:]*(1-mask_sky) + mask_sky*70
            sem_filled[i,1,:,:] = sem[i,1,:,:]*(1-mask_sky) + mask_sky*130
            sem_filled[i,2,:,:] = sem[i,2,:,:]*(1-mask_sky) + mask_sky*180

            # building
            mask_building = abs(sem_rgb[i,:,:]-210)<th_pixle
            mask_building = binary_closing(mask_building)
            mask_building = numpy.asarray(mask_building).astype(numpy.uint8)
            [h,w] = mask_building.shape
            #mask_building[int(h/2):int(h),:] = 0
            mask_building = binary_erosion(mask_building, structure=numpy.ones((nsize,nsize))).astype(mask_building.dtype)

            sem_filled[i,0,:,:] = sem[i,0,:,:]*(1-mask_building) + mask_building*70
            sem_filled[i,1,:,:] = sem[i,1,:,:]*(1-mask_building) + mask_building*70
            sem_filled[i,2,:,:] = sem[i,2,:,:]*(1-mask_building) + mask_building*70


        sem_filled = (sem_filled/255.0-0.5)/0.5
        sem_filled = torch.tensor(sem_filled).cuda().float()

        return sem_filled

    def forward(self):
        # sate rgb to depth
        self.fake_sate_DL = self.netG0(self.sate_R)
        self.fake_sate_L = self.fake_sate_DL[:,0:1,:,:]
        self.fake_sate_D = self.fake_sate_DL[:,1:2,:,:]

        # projection layer
        self.proj_D, self.proj_L = geo_projection(self.fake_sate_D, self.fake_sate_L, self.ori, self.sate_gsd, self.pano_size, is_normalized=True)

        # projected depth + label to street label
        self.fake_street_L = self.netG1(torch.cat((self.proj_D, self.proj_L), 1))
        if self.is_train:
            self.fake_street_L_filled = self.shape_fill()

        # street label to street rgb
        self.fake_street_R = self.netG2(self.fake_street_L)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_LR = torch.cat((self.street_L, self.fake_street_R), 1)
        pred_fake = self.netD(fake_LR.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_LR = torch.cat((self.street_L, self.street_R), 1)
        pred_real = self.netD(real_LR)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # First, G(D) should fake the discriminator
        fake_LR = torch.cat((self.street_L, self.fake_street_R), 1)
        pred_fake = self.netD(fake_LR)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # G0
        self.loss_G_L0 = (self.criterionL1(self.fake_sate_L, self.sate_L) + self.criterionL1(self.fake_sate_D, self.sate_D))

        # G1
        self.loss_G_L1 = self.criterionL1(self.fake_street_L, self.street_L)
        self.loss_G_L1_shape = self.criterionL1(self.fake_street_L, self.fake_street_L_filled)
        self.loss_G_L1 = self.loss_G_L1 + self.loss_G_L1_shape*0.1

        # G2
        self.loss_G_L2 = self.criterionL1(self.fake_street_R, self.street_R)


        self.loss_G = self.loss_G_GAN + (self.loss_G_L0 + self.loss_G_L1 + self.loss_G_L2) * self.lambda_L1
        self.loss_G.backward()        

    def optimize_parameters(self):
        self.forward()

        # update D
        self.set_requires_grad([self.netG0, self.netG1, self.netG2, self.netD], False)
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.netG0, self.netG1, self.netG2, self.netD], False)
        self.set_requires_grad([self.netG0, self.netG1, self.netG2], True)  # no propagate for G0
        self.optimizer_G0.zero_grad()        
        self.optimizer_G1.zero_grad()        
        self.optimizer_G2.zero_grad()        
        self.backward_G()            
        self.optimizer_G0.step()      
        self.optimizer_G1.step()      
        self.optimizer_G2.step()              

    def save_networks(self, epoch):
        torch.save(self.netG0.state_dict(), self.save_dir +'/model_G0_'+str(epoch)+'.pt')
        torch.save(self.netG0.state_dict(), self.save_dir +'/model_G0_latest.pt')           
        torch.save(self.netG1.state_dict(), self.save_dir +'/model_G1_'+str(epoch)+'.pt')
        torch.save(self.netG1.state_dict(), self.save_dir +'/model_G1_latest.pt')         
        torch.save(self.netG2.state_dict(), self.save_dir +'/model_G2_'+str(epoch)+'.pt')
        torch.save(self.netG2.state_dict(), self.save_dir +'/model_G2_latest.pt') 
        torch.save(self.netD.state_dict(), self.save_dir +'/model_D_'+str(epoch)+'.pt')
        torch.save(self.netD.state_dict(), self.save_dir +'/model_D_latest.pt') 

    # load models from the disk
    def load_networks(self, epoch):
        if epoch >= 0:
            self.netG0.load_state_dict(torch.load(self.save_dir +'/model_G0_'+str(epoch)+'.pt',
            map_location=lambda storage, loc: storage.cuda(0)))                    
            self.netG1.load_state_dict(torch.load(self.save_dir +'/model_G1_'+str(epoch)+'.pt',
            map_location=lambda storage, loc: storage.cuda(0)))            
            self.netG2.load_state_dict(torch.load(self.save_dir +'/model_G2_'+str(epoch)+'.pt',
            map_location=lambda storage, loc: storage.cuda(0)))                     
            if self.is_train:
                self.netD.load_state_dict(torch.load(self.save_dir +'/model_D_'+str(epoch)+'.pt',
                map_location=lambda storage, loc: storage.cuda(0)))  
                      
        else:
            self.netG0.load_state_dict(torch.load(self.save_dir +'/model_G0_latest.pt',
            map_location=lambda storage, loc: storage.cuda(0)))              
            self.netG1.load_state_dict(torch.load(self.save_dir +'/model_G1_latest.pt',
            map_location=lambda storage, loc: storage.cuda(0)))            
            self.netG2.load_state_dict(torch.load(self.save_dir +'/model_G2_latest.pt',
            map_location=lambda storage, loc: storage.cuda(0)))
            if self.is_train:
                self.netD.load_state_dict(torch.load(self.save_dir +'/model_D_latest.pt',
                map_location=lambda storage, loc: storage.cuda(0)))   



# abolition study
###########
class DLLModel:
    def name(self):
        return 'DLLModel'

    def initialize(self, opt):
        self.direction = opt.direction
        self.is_train = opt.is_train
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.checkpoints_dir
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')   
        self.lambda_L1 = opt.lambda_L1

        # load/define networks
        #self.netG = networks.define_X(1, 3, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG = networks.define_G(2, 3, opt.ngf, opt.netG, opt.norm_G_D, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.is_train:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)


    def set_input(self, input):
        self.real_D = input['sate_D'].to(self.device)
        self.real_L = input['sate_L'].to(self.device)
        if self.is_train:
            self.real_S = input['street_L'].to(self.device)
        self.img_id = input['img_id']

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        self.fake_S = self.netG(torch.cat((self.real_D, self.real_L), 1))
        a = 0

    def backward_G(self):
        # Second, G(A) = B
        self.loss_G = self.criterionL1(self.fake_S, self.real_S)

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def save_networks(self, epoch):
        torch.save(self.netG.state_dict(), self.save_dir +'/model_G_'+str(epoch)+'.pt')
        torch.save(self.netG.state_dict(), self.save_dir +'/model_G_latest.pt') 

    # load models from the disk
    def load_networks(self, epoch):
        if epoch >= 0:
            self.netG.load_state_dict(torch.load(self.save_dir +'/model_G_'+str(epoch)+'.pt',
            map_location=lambda storage, loc: storage.cuda(0)))       
        else:
            self.netG.load_state_dict(torch.load(self.save_dir +'/model_G_latest.pt',
            map_location=lambda storage, loc: storage.cuda(0)))