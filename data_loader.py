from __future__ import print_function, division
import os, glob, random
import torch
import pandas as pd
from skimage import io, transform, img_as_float, color, img_as_ubyte, exposure
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
from sklearn.utils import shuffle
import copy

plt.ion()   # interactive mode

class Paired_Dataset(Dataset):

    def __init__(self, csv_file, img_size=256, transform=None):
     
        self.files_list = pd.read_csv(csv_file)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):      
        
        low_name = os.path.join(self.files_list.iloc[idx, 0])
        high_name = os.path.join(self.files_list.iloc[idx, 1])
        low_image = Image.open(low_name)
        high_image = Image.open(high_name)
        if low_image.size[0] != self.img_size or low_image.size[1] != self.img_size:
            low_image = low_image.resize((self.img_size, self.img_size), Image.BILINEAR)
        if high_image.size[0] != self.img_size or high_image.size[1] != self.img_size:
            high_image = high_image.resize((self.img_size, self.img_size), Image.BILINEAR)

        sample = {'input': low_image, 'output': high_image}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Compress_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):   
        self.files_list = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):             
        img_name = self.files_list.iloc[idx, 0] # image path
        img = Image.open(img_name)
        
        if self.transform:
            sample = self.transform(img)
        return sample
    
class Duplicate(object):
    def __call__(self, img):
        return {'input': img, 'output': img}
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string 
    
class Resize(object):
    def __init__(self, input_size, output_size=None):
        assert isinstance(input_size, (int, tuple))
        self.input_size = input_size
        self.output_size = output_size

    def __call__(self, sample):
        img_low, img_high = sample['input'], sample['output']         
        img_low = img_low.resize((self.input_size, self.input_size), Image.BILINEAR)
        if self.output_size is not None:
            img_low = img_low.resize((self.output_size, self.output_size), Image.BILINEAR)
        return {'input': img_low, 'output': img_high}
    
class Rescale(object):
    def __init__(self, output_size, up_factor=5, stc=False):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.up_factor = up_factor
        self.stc = stc

    def __call__(self, img):
        img_low = copy.deepcopy(img)
        img_high = copy.deepcopy(img)
        if self.stc == True:
            factor = max(1, np.random.normal(self.up_factor, 0.5))
        else:
            factor = self.up_factor           
        img_low = img_high.resize((int(img.size[1]/factor), int(img.size[0]/factor)))
        img_low = img_low.resize(self.output_size, Image.BILINEAR)
#         img_low = img_low.filter(ImageFilter.GaussianBlur(radius=((factor-1)/2)))
        
        return {'input': img_low, 'output': img_high}

class ToHSV(object):
    def __call__(self, sample):
        img_low, img_high = sample['input'], sample['output']
        img_low = img_low.convert('HSV')
        img_high = img_high.convert('HSV')     
        return {'input': img_low, 'output': img_high}

class ToTensor(object):
    def __call__(self, sample):
        img_low, img_high = sample['input'], sample['output']
        return {'input': transforms.functional.to_tensor(img_low), 'output': transforms.functional.to_tensor(img_high)}
    

def show_patch(dataloader, index = 0, is_hsv = False):
    for i_batch, sample_batched in enumerate(dataloader):
        if i_batch == index:         
            input_batch, output_batch = sample_batched['input'], sample_batched['output']
            if is_hsv:
                input_img = input_batch.numpy().transpose((0, 2, 3, 1))
                output_img = output_batch.numpy().transpose((0, 2, 3, 1))
                for i in range(0, input_batch.shape[0]):
                    input_img[i] = color.hsv2rgb(input_img[i])
                    output_img[i] = color.hsv2rgb(output_img[i])
                input_batch = torch.from_numpy(input_img.transpose(((0, 3, 1, 2))))
                output_batch = torch.from_numpy(output_img.transpose(((0, 3, 1, 2))))
            batch_size = len(input_batch)
            im_size = input_batch.size(2)
            plt.figure(figsize=(20, 10))
            grid = utils.make_grid(input_batch)
            plt.imshow(grid.numpy().transpose((1, 2, 0)), interpolation='bicubic')     
            plt.axis('off')
            plt.figure(figsize=(20, 10))
            grid = utils.make_grid(output_batch)
            plt.imshow(grid.numpy().transpose((1, 2, 0)), interpolation='bicubic')     
            plt.axis('off')
            break
            
def generate_compress_csv(dataset='TMA', ext='jpg'):
    train_imgs = glob.glob(os.path.join('dataset', dataset, '*.'+ext)) + glob.glob(os.path.join('dataset', dataset, '*', '*.'+ext))
    random.shuffle(train_imgs)
    train_df = pd.DataFrame(train_imgs[0:int(0.8*len(train_imgs))])
    valid_df = pd.DataFrame(train_imgs[int(0.8*len(train_imgs)):int(0.9*len(train_imgs))])
    test_df = pd.DataFrame(train_imgs[int(0.9*len(train_imgs)):])
    train_df.to_csv(os.path.join('dataset', dataset, 'train-compress.csv'), index=False)
    valid_df.to_csv(os.path.join('dataset', dataset, 'valid-compress.csv'), index=False)
    test_df.to_csv(os.path.join('dataset', dataset, 'test-compress.csv'), index=False)
    
def compress_csv_path(csv='train', dataset=None):
    if csv =='train':
        return os.path.join('dataset', dataset, 'train-compress.csv')
    if csv =='test':
        return os.path.join('dataset', dataset, 'valid-compress.csv')
    if csv =='valid':
        return os.path.join('dataset', dataset, 'test-compress.csv')
        
def generate_paired_csv(dataset='TMA', in_folder=None, out_folder=None, ext='jpg'):
    train_imgs_in = glob.glob(os.path.join('dataset', dataset, in_folder, '*.'+ext)) + glob.glob(os.path.join('dataset', dataset, in_folder, '*', '*.'+ext))
    train_imgs_out = glob.glob(os.path.join('dataset', dataset, out_folder, '*.'+ext)) + glob.glob(os.path.join('dataset', dataset, out_folder, '*', '*.'+ext))
    df = pd.DataFrame(train_imgs_in)
    df = df.assign(e=pd.DataFrame(train_imgs_out).values)
    df = shuffle(df)
    train_df = pd.DataFrame(df.iloc[0:int(0.8*len(train_imgs_in)), :])
    valid_df = pd.DataFrame(df.iloc[int(0.8*len(train_imgs_in)):int(0.9*len(train_imgs_in)), :])
    test_df = pd.DataFrame(df.iloc[int(0.9*len(train_imgs_in)):, :])
    train_df.to_csv(os.path.join('dataset', dataset, 'train-paired.csv'), index=False)
    valid_df.to_csv(os.path.join('dataset', dataset, 'valid-paired.csv'), index=False)
    test_df.to_csv(os.path.join('dataset', dataset, 'test-paired.csv'), index=False)
    
def paired_csv_path(csv='train', dataset=None):
    if csv =='train':
        return os.path.join('dataset', dataset, 'train-paired.csv')
    if csv =='test':
        return os.path.join('dataset', dataset, 'valid-paired.csv')
    if csv =='valid':
        return os.path.join('dataset', dataset, 'test-paired.csv')
        