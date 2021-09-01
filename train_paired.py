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


def paired_dataloader(args, csv='train'):
    transformed_dataset = data.Paired_Dataset(csv_file=data.paired_csv_path(csv, dataset=args.dataset),
                                              img_size=args.patch_size,
                                              transform=data.Compose([data.ToTensor()])
                                              )
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return dataloader

def train(args, epoch, run, dataloader, generator, feature_extractor, discriminator, optimizer_G, optimizer_D, criterion_pixel, criterion_percep, criterionMSE, Tensor=None, device='cuda:0', patch=None):
    l = args.gan_weight
    p = args.pixel_weight
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
        real_low = Variable(batch['input'].type(Tensor).to(device), requires_grad=False)
        real_high = Variable(batch['output'].type(Tensor).to(device), requires_grad=False)        
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_low.size(0), *patch))).to(device), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_low.size(0), *patch))).to(device), requires_grad=False)       
        #---------------
        #  GAN training
        #---------------        
        if gan:
            optimizer_G.zero_grad()
            fake_high = generator(real_low)
            
            # GAN loss
            pred_real = discriminator(real_high.detach(), real_low)
            pred_fake = discriminator(fake_high, real_low)
            loss_GAN = criterionMSE(pred_fake-pred_real.mean(0, keepdim=True), valid)
        
            # Identity loss
            loss_pixel = criterion_pixel(fake_high, real_high)   
            
            # Perceptual loss
            fake_features = feature_extractor(fake_high)
            real_features = feature_extractor(real_high).detach()
            loss_percep = criterion_percep(fake_features, real_features)
            
            # Total loss
            loss_G = l*loss_GAN + p*loss_pixel + (1-l-p)*loss_percep 
            loss_G.backward()
            optimizer_G.step()
            total_loss = total_loss + loss_G.item()
            
            # Discriminator training
            if iteration % args.num_critic == 0:
                optimizer_D.zero_grad() 
                pred_real = discriminator(real_high, real_low)
                pred_fake = discriminator(fake_high.detach(), real_low)
                loss_real = criterionMSE(pred_real-pred_fake.mean(0, keepdim=True), valid)
                loss_fake = criterionMSE(pred_fake-pred_real.mean(0, keepdim=True), fake)
                loss_D = 0.5 * (loss_real + loss_fake)
                loss_D.backward()
                optimizer_D.step()
                dis_loss = dis_loss + loss_D.item()        
                epoch_loss = epoch_loss + dis_loss + total_loss
                
            sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Identity/Percep/Advers/Total): %.4f/%.4f/%.4f/%.4f' 
                             % (epoch, args.num_epochs, iteration, len(dataloader), loss_D.item(), 
                                loss_pixel.item(), loss_percep.item(), loss_GAN.item(), loss_G.item()))
            
        else:
            optimizer_G.zero_grad()
            fake_high = generator(real_low)
            
            # Identity loss
            loss_pixel = criterion_pixel(fake_high, real_high)   
            
            # Perceptual loss
            fake_features = feature_extractor(fake_high)
            real_features = feature_extractor(real_low).detach()
            loss_percep = criterion_percep(fake_features, real_features)
            
            loss_G = p*loss_pixel + (1-p)*loss_percep 
            loss_G.backward()
            total_loss = total_loss + loss_G.item()
            optimizer_G.step()        
            epoch_loss = epoch_loss + total_loss
            sys.stdout.write('\r[%d/%d][%d/%d] Generator_Loss (Identity/Percep): %.4f/%.4f' 
                             % (epoch, args.num_epochs, iteration, len(dataloader), loss_pixel.item(), loss_percep.item()))
                             
    print("\n ===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(dataloader)))    
    g_path = os.path.join('weights', run, 'generator.pth')
    d_path = os.path.join('weights', run, 'discriminator.pth')
    os.makedirs(os.path.join('weights', run), exist_ok=True)
    torch.save(generator.state_dict(), g_path)
    if gan:
        os.makedirs(os.path.join('weights', run), exist_ok=True)
        torch.save(discriminator.state_dict(), d_path)

def compute_p_snr(path_input, path_ref):
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
    shutil.rmtree('output/print')
    os.makedirs('output/print', exist_ok=True)
    os.makedirs('output/print/lr', exist_ok=True)
    os.makedirs('output/print/hr', exist_ok=True)
    os.makedirs('output/print/sr', exist_ok=True)
    with torch.no_grad(): 
        generator.eval()
        print("=> Printing sampled patches")
        for k, batch in enumerate(dataloader_valid):     
            input, target = batch['input'].to(device), batch['output'].to(device)
            imgs_input =input.float().to(device)
            prediction = generator(imgs_input)
            target = target.float()
            for i in range(target.shape[0]):
                utils.save_image(imgs_input[i], 'output/print/lr/{}_{}.tiff'.format(k, i))
                utils.save_image(target[i], 'output/print/hr/{}_{}.tiff'.format(k, i))
                utils.save_image(prediction[i], 'output/print/sr/{}_{}.tiff'.format(k, i))
            sys.stdout.write("\r ==> Batch {}/{}".format(k+1, len(dataloader_valid)))
        print("\n Computing FID score")
        fid = fid_score.calculate_fid_given_paths(('output/print/sr', 'output/print/hr'), 8, 'cuda:0', 2048)
        print("\n Computing PSNR")
        psnr = compute_p_snr('output/print/sr', 'output/print/hr')
        print("FID score: {}, PSNR: {}".format(fid, psnr))
    return fid, psnr

def main():
    parser = argparse.ArgumentParser(description='Train WSISR on compressed TMA dataset')
    parser.add_argument('--batch-size', default=16, type=int, help='Batch size')
    parser.add_argument('--patch-size', default=256, type=int, help='Patch size')
    parser.add_argument('--num-workers', default=8, type=int, help='Number of workers')
    parser.add_argument('--num-epochs', default=900, type=int, help='Number of epochs, more epochs are desired for GAN training')
    parser.add_argument('--g-lr', default=0.0001, type=float, help='Learning rate of the generator')
    parser.add_argument('--d-lr', default=0.00001, type=float, help='Learning rate of the descriminator')
    parser.add_argument('--gan-weight', default=5e-3, type=float, help='GAN loss weight')
    parser.add_argument('--pixel-weight', default=0.5, type=float, help='Identity loss weight')
    parser.add_argument('--run-from', default=None, type=str, help='Load weights from a previous run, use folder name in [weights] folder')
    parser.add_argument('--gan', default=1, type=int, help='Use GAN')
    parser.add_argument('--norm-layer', default='none', type=str, help='Normalization layer type [none|batch|instance]')   
    parser.add_argument('--num-critic', default=1, type=int, help='Iteration interval for training the descriminator') 
    parser.add_argument('--test-interval', default=50, type=int, help='Epoch interval for FID score testing')
    parser.add_argument('--print-interval', default=10, type=int, help='Epoch interval for output printing')
    parser.add_argument('--dataset', default='TMA', type=str, help='Dataset folder name')
    parser.add_argument('--in-folder', default='low', type=str, help='Low NA image folder name')
    parser.add_argument('--out-folder', default='high', type=str, help='High NA image folder name')      
    parser.add_argument('--extension', default='jpg', type=str, help='Training image extension')   
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    device = torch.device('cuda:0')
    tensor = torch.cuda.FloatTensor
    data.generate_paired_csv(dataset=args.dataset, in_folder=args.in_folder, out_folder=args.out_folder, ext=args.extension)
    valid_dataset = paired_dataloader(args, 'valid')
    train_dataset = paired_dataloader(args, 'train')
    test_dataset = paired_dataloader(args, 'test')
    if args.norm_layer == 'none':
        generator = models.Generator(norm=None)
    else:
        generator = models.Generator(norm=args.norm_layer)
    generator.to(device)
    discriminator = models.Discriminator()
    discriminator.to(device)
    feature_extractor = models.FeatureExtractor().to(device)
    feature_extractor.eval()
    criterion_pixel = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)
    criterion_percep = nn.L1Loss().to(device)
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
    # scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, args.num_epochs, args.g_lr*0.05)
    # scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, args.num_epochs, args.d_lr*0.05)
    run = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    for epoch in range(0, args.num_epochs):
        train(args, epoch, run, train_dataset, generator, feature_extractor, discriminator, optimizer_G, optimizer_D, criterion_pixel, criterion_percep, criterionMSE, tensor, device, patch)
        # scheduler_G.step()
        # scheduler_D.step()
        if epoch % args.print_interval == 0:
            print_output(generator, valid_dataset, device)
    print_output(generator, test_dataset, device)
    
if __name__ == '__main__':
    main()

