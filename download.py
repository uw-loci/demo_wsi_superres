import urllib.request
import argparse
from tqdm import tqdm
import zipfile
import shutil
import os


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        
def unzip_data(zip_path, data_path):
    if os.path.exists(data_path): shutil.rmtree(data_path) 
    os.mkdir(data_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, 
                        default='https://uwmadison.box.com/shared/static/2gt7wqgdkyxv6pos0i18ni4q9sx36dnr.zip', 
                        help='url to dataset')
    parser.add_argument('--path', type=str, default='data.zip', help='path to store data')
    parser.add_argument('--url-weights', type=str, 
                        default='https://uwmadison.box.com/shared/static/t6zwylady7dwqdv7pq5u6xnz7vk9pd5i.pth', 
                        help='url to model weights')
    parser.add_argument('--path-weights', type=str, default='weights/sisr.pth', help='path to store weights')
    
    args = parser.parse_args()
    
    print('downloading data')
    download_url(args.url, args.path)
    unzip_data(args.path, 'input_test_default')
    os.remove(args.path)
    print('downloading model weights')
    if os.path.exists('weights'): shutil.rmtree('weights') 
    os.mkdir('weights')
    download_url(args.url_weights, args.path_weights)
    print('downloading fiji')
    download_url('https://uwmadison.box.com/shared/static/mysrf3cvyhlk4orvhuwi53t5pm0b53k8.zip',
                 'fiji.zip')
    unzip_data('fiji.zip', 'fiji')
    os.remove('fiji.zip')
    
if __name__ == '__main__':
    main()