import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F     
    
class UpSampleBlock(nn.Module):
    """
    Upsample using pixelshuffle
    """
    
    def __init__(self, in_filters, out_filters, bias=False, non_linearity=True):
        super(UpSampleBlock, self).__init__()

        def block(in_filters, out_filters, bias, non_linearity):
            layers = [nn.Conv2d(in_filters, out_filters*4, 3, 1, 1, bias=bias)]
            if non_linearity:
                layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            layers += [nn.PixelShuffle(upscale_factor=2)]
            return nn.Sequential(*layers)
        
        self.layer = block(in_filters, out_filters, bias, non_linearity)

    def forward(self, x):
        return self.layer(x)
    
    
class DownSampleBlock(nn.Module):
    """
    Downsample using stride-2 convolution
    """
    
    def __init__(self, filters, bias=True, non_linearity=False):
        super(DownSampleBlock, self).__init__()
        
        def block(filters, bias, non_linearity):
            layers = [nn.Conv2d(filters, filters, 4, 2, 1, bias=bias), nn.Conv2d(filters, filters, 1, 1, bias=bias)]
            if non_linearity:
                layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            return nn.Sequential(*layers)
        
        self.layer = block(filters, bias, non_linearity)

    def forward(self, x):
        return self.layer(x)
    
    
class CompressBlock(nn.Module):
    """
    Compress the features to vectors
    """
    
    def __init__(self, filters, step=7, latent_size=512, bias=False, non_linearity=False):
        super(CompressBlock, self).__init__()
        
        self.compress = nn.Sequential(nn.Conv2d(filters, step+1, 3, 1, 1, bias=bias))
        
        self.linear = nn.Linear(16, latent_size)

    def forward(self, x): # Nx64x4x4
        returns = []
        x = self.compress(x) # Nx8x4x4
        returns.append(x[:, 0])
        x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]) # Nx8x16
        x = self.linear(x) # Nx8x512       
        for i in range(1, x.shape[1]):
            returns.append(x[:, i]) # each Nx512
        return returns


class DenseResidualBlock(nn.Module):
    """
    Based on: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class RRDBNet(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16):
        super(RRDBNet, self).__init__()

        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        feats = self.conv2(x)
        return feats

class Encoder(nn.Module):
    def __init__(self, img_channels=3, base_filters=64, latent_features=128,  num_res_blocks=16):
        super(Encoder, self).__init__()
            
        self.rrdbnet = RRDBNet(img_channels, base_filters, num_res_blocks) # 64 channels
        self.down1 = DownSampleBlock(base_filters)
        self.down2 = DownSampleBlock(base_filters)
        self.down3 = DownSampleBlock(base_filters)
        self.compress = CompressBlock(base_filters, latent_features)
            
    def forward(self, img):
        f0 = self.rrdbnet(img) # f0: 64x64x64
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = self.down3(f2) # f3: 64x4x4
        c = self.compress(f3)
        
        return [f3, f2, f1, f0], c

class Decoder(nn.Module):
    def __init__(self, img_channels=3, base_filters=64, latent_features=128):
        super(Decoder, self).__init__()
        
        self.fusion1 = nn.Sequential(nn.Conv2d(288, base_filters, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True))         
        self.fusion2 = nn.Sequential(nn.Conv2d(144, base_filters, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        self.up1 = UpSampleBlock(base_filters, base_filters) # in 64x32x32 out 64x64x64
        self.up2 = UpSampleBlock(base_filters*2, base_filters) # in 128x64x64 out 64x128x128
        self.up3 = UpSampleBlock(base_filters*2, base_filters) # in 128x128x128 out 64x256x256
        self.out = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_filters, 3, 3, 1, 1),
        )        # in 64x256x256 out 3x256x256
        
    def forward(self, codes, infos): # om 32x32
        
        fuse1 = self.fusion1(torch.cat((codes[0], infos[1]), 1))
        fuse2 = self.fusion2(torch.cat((codes[1], infos[2]), 1))
        
        out1 = self.up1(infos[0]) # out 64x64x64
        out2 = self.up2(torch.cat((out1, fuse1), 1))
        out3 = self.up3(torch.cat((out2, fuse2), 1))
        out = self.out(torch.cat((out3, codes[2]), 1))
        
        return out