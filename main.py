from skimage import io, img_as_ubyte, morphology, img_as_bool, img_as_float, exposure, color
from skimage.util.shape import view_as_windows
from skimage.util import crop, pad
from skimage.transform import resize, rescale
from PIL import Image
import imagej


import numpy as np
import os, glob, sys
from collections import OrderedDict
import shutil
import argparse
import pandas as pd
import csv
import warnings
warnings.simplefilter("ignore", UserWarning)


from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torch.functional as F
import torch

import model_multi_fcn as md

def generate_csv(img_dir, csv_dir):
    file_list= [name for name in os.listdir(img_dir) if 
                os.path.isfile(os.path.join(img_dir, name))]
    num_img = len(file_list)
    csv_file_path = os.path.join(csv_dir, 'image_files.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, num_img):           
            img = os.path.join(img_dir, str(file_list[i]))
            filewriter.writerow([img])
    return csv_file_path

class SynthDataset(Dataset):

    def __init__(self, csv_file, transform=None):
     
        self.files_list = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        img = os.path.join(self.files_list.iloc[idx, 0])

        img = io.imread(img)
        img = img_as_float(img)
        img = img.transpose((2, 0, 1))
        
        img = torch.from_numpy(img)

        sample = {'image':img}

        if self.transform:
            sample = self.transform(sample)
        return sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/sisr.pth', help='path to model weights')
    parser.add_argument('--use-cuda', type=int, default=1, help='1: use cuda, 0: use cpu')
    parser.add_argument('--which-gpu', type=int, default=0, help='index of gpu')
    parser.add_argument('--load-multiple-gpu-weights', type=int, default=0, help='1: multiple gpu weights, 0: single gpu weghts')
    parser.add_argument('--input-folder', type=str, default='default', help='input_test + _FOLDERNAME')
    parser.add_argument('--intensity', type=tuple, default=(0, 230), help='output intensity rescale')
    parser.add_argument('--pilot', type=int, default=0, help='1: only process the first image, 0: process all images')
    
    args = parser.parse_args()
    
    demo(args)

def demo(args):
    
    available_cuda = torch.cuda.is_available()
    if not available_cuda:
        print('gpu not avaibable')
    if args.use_cuda and available_cuda:
        device = 'cuda:' + str(args.which_gpu)
        torch.cuda.set_device(args.which_gpu)
        print('use GPU')
    else:
        device = 'cpu'

    model = md.ResUNet(3, 3)
    DICT_DIR = args.weights
    state_dict = torch.load(DICT_DIR, map_location=torch.device(device))

    if args.load_multiple_gpu_weights:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    
    print('loading ImageJ, please wait')
    ij = imagej.init('fiji/fiji/Fiji.app/')
    
    # use for SHG
    TASK = args.input_folder
    INPUT_DIR = 'input_test_' + TASK
    INPUT_PATCH_DIR = 'input_patch_temp/'
    OUTPUT_DIR = 'output_test_' + TASK
    OUTPUT_PATCH_DIR = 'output_patch_temp/'
    CHANNEL_DIR = 'channel_temp/'
    if os.path.exists(CHANNEL_DIR): shutil.rmtree(CHANNEL_DIR) 
    os.mkdir(CHANNEL_DIR)
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR) 
    os.mkdir(OUTPUT_DIR)
    files = [f for f in os.listdir(INPUT_DIR)]
    files.sort()
    window_shape = (256, 256, 3)
    step_size = 224
    is_crf = False
    for k, fn in enumerate(files):
        print('forward pass, please wait...')  
        if os.path.exists(INPUT_PATCH_DIR): shutil.rmtree(INPUT_PATCH_DIR)
        if os.path.exists(OUTPUT_PATCH_DIR): shutil.rmtree(OUTPUT_PATCH_DIR) 
        os.mkdir(INPUT_PATCH_DIR)
        os.mkdir(OUTPUT_PATCH_DIR)
        
        fs = os.path.join(INPUT_DIR, fn)
        img = io.imread(fs)
        
        shape_0_factor = np.ceil(img.shape[0] / window_shape[0]) # height
        shape_1_factor = np.ceil(img.shape[1] / window_shape[1]) # width
        canvas_0 = int(window_shape[0] * shape_0_factor)
        canvas_1 = int(window_shape[1] * shape_1_factor)
        pad_0 = canvas_0 - img.shape[0]
        pad_1 = canvas_1 - img.shape[1]
        canvas = pad(img, ((0, pad_0), (0, pad_1), (0, 0)), mode='reflect')
        windows = view_as_windows(canvas, window_shape, step_size)
        with open(OUTPUT_PATCH_DIR+'TileConfiguration.txt', 'w') as text_file:
            print('dim = {}'.format(2), file=text_file)
            for i in range (0, windows.shape[1]):
                for j in range (0, windows.shape[0]):
                    patch = windows[j, i, ]
                    patch = np.squeeze(patch)           
                    io.imsave(INPUT_PATCH_DIR+'/{}_{}.tiff'.format(j, i), patch)
                    print('{}_{}.tiff; ; ({}, {})'.format(j, i, i*step_size, j*step_size), file=text_file)
        csv_file_path = generate_csv(INPUT_PATCH_DIR, os.getcwd())
        dataset = SynthDataset(csv_file=csv_file_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        with open(csv_file_path) as f:
            reader = csv.reader(f)
            name_list = list(reader)
            with torch.no_grad():
                for iteration, batch in enumerate(dataloader):
                    input = batch['image'].float().to(device).view(1, 3, window_shape[0], window_shape[1])
                    prediction = model(input)
                    for i in range(prediction.size(0)):
                        img_name = str(name_list[iteration]).split('/')[-1].split('\'')[0]
                        result_patch = prediction[i, :, :, :,].cpu().detach().numpy().transpose((1, 2, 0))*255
                        result_patch = result_patch.astype(np.uint8)
                        io.imsave(OUTPUT_PATCH_DIR+img_name, result_patch)
        
        print('stitching, please wait...')                
        params = {'type': 'Positions from file', 'order': 'Defined by TileConfiguration', 
                'directory':OUTPUT_PATCH_DIR, 'ayout_file': 'TileConfiguration.txt', 
                'fusion_method': 'Linear Blending', 'regression_threshold': '0.30', 
                'max/avg_displacement_threshold':'2.50', 'absolute_displacement_threshold': '3.50', 
                'compute_overlap':False, 'computation_parameters': 'Save computation time (but use more RAM)', 
                'image_output': 'Write to disk', 'output_directory': CHANNEL_DIR}
        plugin = "Grid/Collection stitching"
        ij.py.run_plugin(plugin, params)     

        output_name = os.path.join(OUTPUT_DIR, fn)
        listOfChannels = [f for f in os.listdir(CHANNEL_DIR)]
        c1 = io.imread(os.path.join(CHANNEL_DIR, listOfChannels[0]))
        c2 = io.imread(os.path.join(CHANNEL_DIR, listOfChannels[1]))
        c3 = io.imread(os.path.join(CHANNEL_DIR, listOfChannels[2]))
        c1 = c1[:img.shape[0], :img.shape[1]]
        c2 = c2[:img.shape[0], :img.shape[1]]
        c3 = c3[:img.shape[0], :img.shape[1]]
        img_to_save = np.stack((c1, c2, c3))
        img_to_save = exposure.rescale_intensity(img_to_save, in_range=args.intensity, out_range=(0, 255))
        print(str(k+1)+"/" + str(len(files)) + " output saved as: " + output_name)
        io.imsave(output_name, img_as_ubyte(img_to_save))
        if args.pilot:
            break

if __name__ == '__main__':
    main()
