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
from skimage import io, img_as_float

import data_loader as data
from models import stylegan, backbones, models
import pytorch_fid.fid_score as fid_score


def paired_dataloader(args, csv='train'):
    transformed_dataset = data.Paired_Dataset(csv_file=data.paired_csv_path(csv, dataset=args.dataset),
                                              img_size=args.patch_size,
                                              transform=data.Compose([data.Resize(int(args.patch_size/8)), data.ToTensor()])
                                              )
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return dataloader

def compress_dataloader(args, csv='train'):
    transformed_dataset = data.Compress_Dataset(csv_file=data.compress_csv_path(csv, dataset=args.dataset),
                                              transform=data.Compose([
                                                  transforms.RandomCrop((args.patch_size, args.patch_size)),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomVerticalFlip(),
                                                  data.Duplicate(),
                                                  data.Resize(int(args.patch_size/8)), 
                                                  data.ToTensor()])
                                              )
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return dataloader

def train_forward(args, epoch, run, dataloader, generator, feature_extractor, optimizer_G, criterion_pixel, criterion_percep, Tensor=None, device='cuda:0'):
    p = args.pixel_weight
    total_epoch_loss = 0
    total_percep_loss = 0
    total_pixel_loss = 0
    generator.train()
    for iteration, batch in enumerate(dataloader):
        optimizer_G.zero_grad()
        
        real_low = Variable(batch['input'].type(Tensor).to(device), requires_grad=False)
        real_high = Variable(batch['output'].type(Tensor).to(device), requires_grad=False)           
        fake_high = generator(real_low)

        # Identity loss
        loss_pixel = criterion_pixel(fake_high, real_high)   

        # Perceptual loss
        fake_features = feature_extractor(fake_high)
        real_features = feature_extractor(real_high).detach()
        loss_percep = criterion_percep(fake_features, real_features)

        loss_G = p*loss_pixel + (1-p)*loss_percep 
        total_epoch_loss = total_epoch_loss + loss_G.item()
        total_percep_loss = total_percep_loss + loss_percep.item()
        total_pixel_loss = total_pixel_loss + loss_pixel.item()
        loss_G.backward()
        optimizer_G.step()        
        sys.stdout.write('\r[%d/%d][%d/%d] Generator_Loss (Identity/Percep/Total): %.4f/%.4f/%.4f' 
                         % (epoch, args.num_epochs, iteration+1, len(dataloader), loss_pixel.item(), loss_percep.item(), loss_G.item()))
                             
    print("\n ============> Epoch {} Complete: Avg. Loss (Identity/Percep/Total): {:.4f}/{:.4f}/{:.4f}".format(epoch, total_pixel_loss/len(dataloader), 
                                                                                                               total_percep_loss/len(dataloader), total_epoch_loss/len(dataloader)))    
    g_path = os.path.join('weights', run, 'generator.pth')
    os.makedirs(os.path.join('weights', run), exist_ok=True)
    torch.save(generator.state_dict(), g_path)
    
def train_gan(args, epoch, run, dataloader, generator, feature_extractor, discriminator, optimizer_G, optimizer_D, criterion_pixel, criterion_percep, criterionMSE, Tensor=None, device='cuda:0', patch=None):
    p = args.pixel_weight
    l = args.gan_weight
    total_epoch_loss = 0
    total_percep_loss = 0
    total_pixel_loss = 0
    total_adv_loss = 0
    generator.train()
    for iteration, batch in enumerate(dataloader):
        optimizer_G.zero_grad()
        
        real_low = Variable(batch['input'].type(Tensor).to(device), requires_grad=False)
        real_high = Variable(batch['output'].type(Tensor).to(device), requires_grad=False)
        
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_low.size(0), *patch))).to(device), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_low.size(0), *patch))).to(device), requires_grad=False)
        
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
        total_epoch_loss = total_epoch_loss + loss_G.item()
        total_percep_loss = total_percep_loss + loss_percep.item()
        total_pixel_loss = total_pixel_loss + loss_pixel.item()
        total_adv_loss = total_adv_loss + loss_GAN.item()
        loss_G.backward()
        optimizer_G.step()
        
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
        
        sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f, Generator_Loss (Identity/Percep/Advers/Total): %.4f/%.4f/%.4f/%.4f' 
                         % (epoch, args.num_epochs, iteration+1, len(dataloader), loss_D.item(), loss_pixel.item(), loss_percep.item(), loss_GAN.item(), loss_G.item()))
                             
    print("\n ============> Epoch {} Complete: Avg. Loss (Identity/Percep/Advers/Total): {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(epoch,
                                                                                                                             total_pixel_loss/len(dataloader),
                                                                                                                             total_percep_loss/len(dataloader),
                                                                                                                             total_adv_loss/len(dataloader),
                                                                                                                             total_epoch_loss/len(dataloader)))    
    g_path = os.path.join('weights', run, 'generator.pth')
    os.makedirs(os.path.join('weights', run), exist_ok=True)
    torch.save(generator.state_dict(), g_path)

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

def print_output(epoch, generator, dataloader_valid, metric='FID'):
    print_path = os.path.join('output/print', str(epoch))
    lr_path = os.path.join(print_path, 'lr')
    hr_path = os.path.join(print_path, 'hr')
    sr_path = os.path.join(print_path, 'sr')
    os.makedirs(print_path, exist_ok=True)
    os.makedirs(lr_path, exist_ok=True)
    os.makedirs(hr_path, exist_ok=True)
    os.makedirs(sr_path, exist_ok=True)
    device = next(generator.parameters()).device
    with torch.no_grad(): 
        generator.eval()
        print("=> Printing sampled patches")
        for k, batch in enumerate(dataloader_valid):     
            input, target = batch['input'].to(device), batch['output'].to(device)
            imgs_input =input.float().to(device)
            prediction = generator(imgs_input)
            target = target.float()
            for i in range(target.shape[0]):
                utils.save_image(imgs_input[i], os.path.join(lr_path, '{}_{}.tiff'.format(k, i)))
                utils.save_image(target[i], os.path.join(hr_path, '{}_{}.tiff'.format(k, i)))
                utils.save_image(prediction[i], os.path.join(sr_path, '{}_{}.tiff'.format(k, i)))
            sys.stdout.write("\r ==> Batch {}/{}".format(k+1, len(dataloader_valid)))
        if metric=='FID':
            print("\n Computing FID score")
            fid = fid_score.calculate_fid_given_paths((sr_path, hr_path), 8, 'cuda:0', 2048)
            psnr = np.nan
        elif metric=='PSNR':
            print("\n Computing PSNR")
            psnr = compute_p_snr('output/print/sr', 'output/print/hr')
            fid = np.nan
        print("FID score: {}, PSNR: {}".format(fid, psnr))
    return fid, psnr

def main():
    parser = argparse.ArgumentParser(description='Train WSISR on compressed TMA dataset')
    parser.add_argument('--batch-size', default=4, type=int, help='Batch size')
    parser.add_argument('--patch-size', default=256, type=int, help='Patch size')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--num-epochs', default=900, type=int, help='Number of epochs, more epochs are desired for GAN training')
    parser.add_argument('--g-lr', default=0.0001, type=float, help='Learning rate of the generator')
    parser.add_argument('--d-lr', default=0.00001, type=float, help='Learning rate of the descriminator')
    parser.add_argument('--gan-weight', default=5e-2, type=float, help='GAN loss weight')
    parser.add_argument('--pixel-weight', default=0.8, type=float, help='Identity loss weight')
    parser.add_argument('--run-from', default=None, type=str, help='Load weights from a previous run, use folder name in [weights] folder')
    parser.add_argument('--compress-type', default='compress', type=str, help='Input image compression method [compress|NA]')
    parser.add_argument('--gan', default=0, type=int, help='Use GAN')
    parser.add_argument('--num-critic', default=10, type=int, help='Iteration interval for training the descriminator') 
    parser.add_argument('--print-interval', default=10, type=int, help='Epoch interval for output printing and evaluation')
    parser.add_argument('--dataset', default='TMA', type=str, help='Dataset folder name')
    parser.add_argument('--in-folder', default='low', type=str, help='Low NA image folder name')
    parser.add_argument('--out-folder', default='high', type=str, help='High NA image folder name')      
    parser.add_argument('--extension', default='jpg', type=str, help='Training image extension')   
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    device = torch.device('cuda:0')
    tensor = torch.cuda.FloatTensor
    if args.compress_type=='NA':
        data.generate_paired_csv(dataset=args.dataset, in_folder=args.in_folder, out_folder=args.out_folder, ext=args.extension)
        valid_dataset = paired_dataloader(args, 'valid')
        train_dataset = paired_dataloader(args, 'train')
        test_dataset = paired_dataloader(args, 'test')
    elif args.compress_type=='compress':
        data.generate_compress_csv(dataset=args.dataset, ext=args.extension)
        valid_dataset = compress_dataloader(args, 'valid')
        train_dataset = compress_dataloader(args, 'train')
        test_dataset = compress_dataloader(args, 'test')
        
    latent_bank = stylegan.StyledGenerator().to(device)
    encoder = backbones.Encoder().to(device)
    decoder = backbones.Decoder().to(device)
    latent_weights = torch.load('latent_bank_weights/130000.model')
    latent_bank.load_state_dict(latent_weights, strict=False)
    for p in latent_bank.parameters():
        p.requires_grad=False
    for p in latent_bank.generator.fusion.parameters():
        p.requires_grad=True
    trainable_params = list(latent_bank.generator.fusion.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    generator = models.EDLatentBank(encoder, decoder, latent_bank).to(device)
    
    feature_extractor = backbones.FeatureExtractor().to(device)
    feature_extractor.eval()
    criterion_pixel = nn.L1Loss().to(device)
    criterion_percep = nn.L1Loss().to(device)
    optimizer_G = torch.optim.Adam(trainable_params, lr=args.g_lr)
    if args.gan==1:
        discriminator = models.Discriminator()
        discriminator.to(device)
        criterionMSE = nn.MSELoss().to(device)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr)
        patch = (1, args.patch_size // 2 ** 4, args.patch_size // 2 ** 4)
    if args.run_from is not None:
        generator.load_state_dict(torch.load(os.path.join('weights', args.run_from, 'generator.pth')))
        if args.gan==1:
            try:
                discriminator.load_state_dict(torch.load(os.path.join('weights', args.run_from, 'discriminator.pth')))
            except:
                print('Discriminator weights not found!')
                pass
        
    run = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    for epoch in range(0, args.num_epochs):
        if args.gan:
            train_gan(args, epoch, run, train_dataset, generator, feature_extractor, discriminator, optimizer_G, optimizer_D, criterion_pixel, criterion_percep, criterionMSE, tensor, device, patch)
        else:
            train_forward(args, epoch, run, train_dataset, generator, feature_extractor, optimizer_G, criterion_pixel, criterion_percep, tensor, device)
        
        if epoch % args.print_interval == 0:
            print_output(epoch, generator, valid_dataset)
    print_output(epoch, generator, test_dataset)
    
if __name__ == '__main__':
    main()

