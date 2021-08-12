import os, argparse, sys, shutil, warnings, glob
from datetime import datetime
import matplotlib.pyplot as plt
from math import log2, log10
import pandas as pd
import numpy as np
from collections import OrderedDict

from torchvision import transforms, utils
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler

from skimage import exposure, color, io, img_as_float, img_as_ubyte
from skimage.util import view_as_windows, pad, montage
from PIL import Image, ImageFilter
import imagej

import data_loader as data
import models

import pytorch_fid.fid_score as fid_score


def new_compress_curriculum(args, cur_factor, csv='train', stc=False):
    transformed_dataset = data.Compress_Dataset(csv_file=data.compress_csv_path(csv, args.dataset),
                                               transform=data.Compose([
                                                   transforms.RandomCrop((args.patch_size, args.patch_size)),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomVerticalFlip(),
                                                   data.Rescale((args.patch_size, args.patch_size), up_factor=cur_factor, stc=stc), 
                                                   data.ToTensor()
                                           ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return dataloader

def train(args, epoch, run, dataloader, generator, discriminator, optimizer_G, optimizer_D, criterionL, criterionMSE, Tensor=None, device='cuda:0', patch=None):
    l = args.percep_weight
    if args.gan == 0:
        gan = False
    else:
        gan = True
    epoch_loss = 0
    gan_loss = 0
    total_loss = 0
    dis_loss = 0
    generator.train()
    for iteration, batch in enumerate(dataloader):
        real_mid = Variable(batch['input'].type(Tensor).to(device), requires_grad=False)
        real_high = Variable(batch['output'].type(Tensor).to(device), requires_grad=False)        
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_mid.size(0), *patch))).to(device), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_mid.size(0), *patch))).to(device), requires_grad=False)       
        #---------------
        #  Train Generator
        #---------------        
        optimizer_G.zero_grad()   
        # GAN loss
        fake_high = generator(real_mid)
        if gan:
            pred_fake = discriminator(fake_high, real_mid)
            loss_GAN = criterionMSE(pred_fake, valid)
        
        # Identity
        lossL1 = criterionL(fake_high, real_high)               
        loss_pixel = lossL1        
        # Total loss
        if gan:
            loss_G = l * loss_GAN + (1-l) * loss_pixel   
            loss_G.backward()
            total_loss = total_loss + loss_G.item()
            gan_loss = gan_loss + loss_GAN.item()
        else:
            loss_pixel.backward()
        optimizer_G.step()        
        #---------------
        #  Train Discriminator
        #---------------         
        if gan and iteration % args.num_critic == 0:
            optimizer_D.zero_grad() 
            # Real loss
            pred_real = discriminator(real_high, real_mid)
            loss_real = criterionMSE(pred_real, valid)        
            # Fake loss
            pred_fake = discriminator(fake_high.detach(), real_mid)
            loss_fake = criterionMSE(pred_fake, fake)
            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()
            dis_loss = dis_loss + loss_D.item()        
        epoch_loss = epoch_loss + loss_pixel.item()       
        if gan:
            sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Identity/Advers/Total): %.4f/%.4f/%.4f' 
                             % (epoch, args.num_epochs, iteration, len(dataloader), loss_D.item(), 
                                loss_pixel.item(), loss_GAN.item(), loss_G.item()))
        else:
            sys.stdout.write('\r[%d/%d][%d/%d] Generator_L1_Loss: %.4f' 
                             % (epoch, args.num_epochs, iteration, len(dataloader), loss_pixel.item()))
    print("\n ===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(dataloader)))    
    g_path = os.path.join('weights', run, 'generator.pth')
    d_path = os.path.join('weights', run, 'discriminator.pth')
    os.makedirs(os.path.join('weights', run), exist_ok=True)
    torch.save(generator.state_dict(), g_path)
    if gan:
        os.makedirs(os.path.join('weights', run), exist_ok=True)
        torch.save(discriminator.state_dict(), d_path)

def test(args, generator, test_csv, stitching=False):
    try:
        shutil.rmtree('output')
    except:
        pass
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/lr', exist_ok=True)
    os.makedirs('output/hr', exist_ok=True)
    os.makedirs('output/sr', exist_ok=True)
    os.makedirs('output/temp_patch', exist_ok=True)
    os.makedirs('output/temp_patch_target', exist_ok=True)
    os.makedirs('output/temp_channel', exist_ok=True)
    step = 192
    test_files = pd.read_csv(test_csv)
    avg_fid = 0
    avg_psnr = 0
    for k in range(len(test_files)):
        img = Image.open(test_files.iloc[k, 0])
        img_hr_array = img_as_float(np.array(img))
        img_lr = img.resize((int(img.size[1]/args.up_scale), int(img.size[0]/args.up_scale)))
        img_lr = img_lr.resize(img.size, Image.BILINEAR)
        img_lr = img_lr.filter(ImageFilter.GaussianBlur(radius=((args.up_scale-1)/2)))
        img_lr_array = img_as_float(np.array(img_lr))
        pad_h = int((np.floor(img_lr_array.shape[0]/step) * step + args.patch_size) - img_lr_array.shape[0])
        pad_w = int((np.floor(img_lr_array.shape[1]/step) * step + args.patch_size) - img_lr_array.shape[1])
        img_lr_array_padded = pad(img_lr_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        img_lr_wd = view_as_windows(img_lr_array_padded, (args.patch_size, args.patch_size, 3), step=step)
        img_lr_wd = np.squeeze(img_lr_wd)
        img_hr_array_padded = pad(img_hr_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        img_hr_wd = view_as_windows(img_hr_array_padded, (args.patch_size, args.patch_size, 3), step=step)
        img_hr_wd = np.squeeze(img_hr_wd) 
        with open('output/temp_patch/TileConfiguration.txt', 'w') as text_file:
            print('dim = {}'.format(2), file=text_file)
            with torch.no_grad():
                generator.eval()
                for i in range (0, img_lr_wd.shape[1]):
                    for j in range (0, img_lr_wd.shape[0]):
                        target = img_hr_wd[j, i]
                        patch = img_lr_wd[j, i].transpose((2, 0, 1))[None, :]
                        patch_tensor = torch.from_numpy(patch).float().cuda()
                        prediction = generator(patch_tensor)
                        io.imsave('output/temp_patch/{}_{}.tiff'.format(j, i), img_as_ubyte(np.clip(prediction.cpu().numpy()[0], 0, 1)))
                        io.imsave('output/temp_patch_target/{}_{}.tiff'.format(j, i), img_as_ubyte(target))
                        print('{}_{}.tiff; ; ({}, {})'.format(j, i, i*step, j*step), file=text_file)
        fid = fid_score.calculate_fid_given_paths(('output/temp_patch', 'output/temp_patch_target'), 8, 'cuda:0', 2048)
        avg_fid = avg_fid + fid
        if stitching:
            sys.stdout.write('\r{}/{} stitching, please wait...'.format(k+1, len(test_files)))                
            params = {'type': 'Positions from file', 'order': 'Defined by TileConfiguration', 
                    'directory':'output/temp_patch', 'ayout_file': 'TileConfiguration.txt', 
                    'fusion_method': 'Linear Blending', 'regression_threshold': '0.30', 
                    'max/avg_displacement_threshold':'2.50', 'absolute_displacement_threshold': '3.50', 
                    'compute_overlap':False, 'computation_parameters': 'Save computation time (but use more RAM)', 
                    'image_output': 'Write to disk', 'output_directory': 'output/temp_channel'}
            plugin = "Grid/Collection stitching"
            ij.py.run_plugin(plugin, params)
            list_channels = [f for f in os.listdir('output/temp_channel')]
            c1 = io.imread(os.path.join('output/temp_channel', list_channels[0]))
            c2 = io.imread(os.path.join('output/temp_channel', list_channels[1]))
            c3 = io.imread(os.path.join('output/temp_channel', list_channels[2]))
            c1 = c1[:img.size[1], :img.size[0]]
            c2 = c2[:img.size[1], :img.size[0]]
            c3 = c3[:img.size[1], :img.size[0]]
            img_to_save = np.clip(np.stack((c1, c2, c3)).transpose((1, 2, 0)), 0, 1)
            io.imsave(os.path.join('output/sr', os.path.basename(test_files.iloc[k, 0]).replace('.jpg', '.tiff')), img_as_ubyte(img_to_save))
            io.imsave(os.path.join('output/lr', os.path.basename(test_files.iloc[k, 0]).replace('.jpg', '.tiff')), img_as_ubyte(img_lr_array))
            io.imsave(os.path.join('output/hr', os.path.basename(test_files.iloc[k, 0]).replace('.jpg', '.tiff')), img_as_ubyte(img))
        else:
            psnr = p_snr('output/temp_patch', 'output/temp_patch_target')
            avg_psnr = avg_psnr + psnr
        if stitching:
            psnr = p_snr('output/sr', 'output/hr')
        else:
            psnr = avg_psnr / len(test_files)
    fid = avg_fid / len(test_files)
    return fid, psnr

def p_snr(path_input, path_ref):
    MSE = nn.MSELoss()
    imgs_input = glob.glob(os.path.join(path_input, '*.tiff'))
    imgs_ref = glob.glob(os.path.join(path_ref, '*.tiff'))
    ave_psnr = 0
    for i in range(len(imgs_input)):
        img_input = torch.from_numpy(img_as_float(io.imread(imgs_input[i]).transpose(2, 1, 0)))               
        img_ref = torch.from_numpy(img_as_float(io.imread(imgs_ref[i]).transpose(2, 1, 0)))
        img_input = img_input[None, :]
        img_ref = img_ref[None, :]             
        mse = MSE(img_input, img_ref)               
        psnr = 10 * log10(1 / mse.item())
        ave_psnr += psnr
    ave_psnr = ave_psnr / len(imgs_input)
    return ave_psnr

def print_output(generator, dataloader_valid, device='cuda:0'):
    os.makedirs('output/print', exist_ok=True)
    os.makedirs('output/print/lr', exist_ok=True)
    os.makedirs('output/print/hr', exist_ok=True)
    os.makedirs('output/print/sr', exist_ok=True)
    with torch.no_grad(): 
        generator.eval()
        print("===> 8x: printing sampled patches")
        for iteration, batch in enumerate(dataloader_valid):     
            input, target = batch['input'].to(device), batch['output'].to(device)
            imgs_input =input.float().to(device)
            prediction = generator(imgs_input)
            target = target.float()
            for i in range(target.shape[0]):
                utils.save_image(imgs_input[i], 'output/print/lr/{}.tiff'.format(i))
                utils.save_image(target[i], 'output/print/hr/{}.tiff'.format(i))
                utils.save_image(prediction[i], 'output/print/sr/{}.tiff'.format(i))
            break 

def main():
    parser = argparse.ArgumentParser(description='Train WSISR on compressed TMA dataset')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--patch-size', default=224, type=int, help='Patch size')
    parser.add_argument('--up-scale', default=5, type=float, help='Targeted upscale factor')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--num-epochs', default=900, type=int, help='Number of epochs, more epochs are desired for GAN training')
    parser.add_argument('--g-lr', default=0.0001, type=float, help='Learning rate of the generator')
    parser.add_argument('--d-lr', default=0.00001, type=float, help='Learning rate of the descriminator')
    parser.add_argument('--percep-weight', default=0.01, type=float, help='GAN loss weight')
    parser.add_argument('--run-from', default=None, type=str, help='Load weights from a previous run, use folder name in [weights] folder')
    parser.add_argument('--start-epoch', default=1, type=int, help='Starting epoch for the curriculum, start at 1/2 of the epochs to skip the curriculum')
    parser.add_argument('--gan', default=1, type=int, help='Use GAN')
    parser.add_argument('--num-critic', default=1, type=int, help='Iteration interval for training the descriminator') 
    parser.add_argument('--test-interval', default=50, type=int, help='Epoch interval for FID score testing')
    parser.add_argument('--print-interval', default=10, type=int, help='Epoch interval for output printing')     
    parser.add_argument('--dataset', default='TMA', type=str, help='Dataset folder name')
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    device = torch.device('cuda:0')
    tensor = torch.cuda.FloatTensor
    data.generate_compress_csv()
    valid_dataset = new_compress_curriculum(args, args.up_scale, 'valid')
    generator = models.Generator()
    generator.to(device);
    discriminator = models.Discriminator()
    discriminator.to(device);
    criterionL = nn.L1Loss().cuda()
    criterionMSE = nn.MSELoss().cuda()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.g_lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr)
    patch = (1, args.patch_size // 2 ** 4, args.patch_size // 2 ** 4)
    if args.run_from is not None:
        generator.load_state_dict(torch.load(os.path.join('weights', args.run_from, 'generator.pth')))
        try:
            discriminator.load_state_dict(torch.load(os.path.join('weights', args.run_from, 'discriminator.pth')))
        except:
            print('Discriminator weights not found!')
            pass
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.g_lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, args.num_epochs, args.g_lr*0.05)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, args.num_epochs, args.d_lr*0.05)
    run = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    cur_length = int(0.5*args.num_epochs)
    init_scale = 2**2
    step_size = (2**args.up_scale-init_scale) / cur_length
    print('loading ImageJ, please wait')
    ij = imagej.init('fiji/fiji/Fiji.app/')
    for epoch in range(args.start_epoch, args.num_epochs):
        factor = min(log2(init_scale+(epoch-1)*step_size), args.up_scale)
        print('curriculum updated: {} '.format(factor))
        train_dataset = new_compress_curriculum(args, factor, 'train', stc=True)
        train(args, epoch, run, train_dataset, generator, discriminator, optimizer_G, optimizer_D, criterionL, criterionMSE, tensor, device, patch)
        scheduler_G.step()
        scheduler_D.step()
        if epoch % args.test_interval == 0:
            fid, psnr = test(args, generator, data.compress_csv_path('valid', args.dataset))
            print('\r>>>> PSNR: {}, FID: {}'.format(psnr, fid))
        if epoch % args.print_interval == 0:
            print_output(generator, valid_dataset, device)
    test(args, generator, data.compress_csv_path('test', args.dataset), stitching=True)
    
if __name__ == '__main__':
    main()

