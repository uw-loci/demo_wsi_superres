import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(torch.nn.Module):   
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, norm=None):
        super(ConvBlock, self).__init__()
        self.is_norm = norm   
        self.conv0 = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias)
        if norm == 'instance':
            self.norm0 = nn.InstanceNorm2d(output_size)
            self.norm1 = nn.InstanceNorm2d(output_size)
        elif norm == 'batch':
            self.norm0 = nn.BatchNorm2d(output_size)
            self.norm1 = nn.BatchNorm2d(output_size)
        self.act = nn.PReLU()
        self.conv1 = nn.Conv2d(output_size, output_size, 1, 1, bias=False)
        
    def forward(self, x):      
        x = self.conv0(x)
        if self.is_norm is not None:
            x = self.norm0(x)
            x = self.act(x)
            x = self.conv1(x)
            x = self.norm1(x)
        else:
            x = self.act(x)
            x = self.conv1(x)              
        return x
    
    
class SkipBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, stride=2):
        super(SkipBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, 1, stride, bias=False)    
    def forward(self, x):  
        x = self.conv(x)  
        return x
    
    
class DownBlock(nn.Module):
    def __init__(self, in_size, out_size, norm=None):
        super(DownBlock, self).__init__()
        self.conv0 = ConvBlock(in_size, out_size, 4, 2, 1, True, norm)
        self.skip = SkipBlock(in_size, out_size, stride=2)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.conv0(x) + self.skip(x)
        x = self.act(x)
        return x
    
    
class UpBlock(nn.Module):
    def __init__(self, in_size, out_size, norm=None):
        super(UpBlock, self).__init__()
        self.conv = ConvBlock(in_size, out_size, 3, 1, 1, True, norm)
        self.skip = SkipBlock(in_size, out_size, stride=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv(x) + self.skip(x)
        return x
    
class PixBlock(nn.Module):
    def __init__(self, in_size, out_size=3, scale=2, norm=None):
        super(PixBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size * (2**scale), 1, 1)
        self.up = nn.PixelShuffle(scale)
    def forward(self, x):
        x = self.conv1(x)
        x = self.up(x)
        return x
    
    
class CompressionBlock(nn.Module):
    def __init__(self, in_size=16, code_size=512):
        super(CompressionBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, in_size, 3, 1, 1)  
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.act = nn.PReLU()
        self.linear = nn.Linear(in_size, code_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x) # N x C x 1 x 1
        x = self.act(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.linear(x)
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, img_channel=3, base_channel=8, norm=None, code_size=512):
        super(Encoder, self).__init__()
        self.down0 = DownBlock(img_channel, base_channel*2, norm=norm)
        self.down1 = DownBlock(base_channel*2, base_channel*4, norm=norm)
        self.down2 = DownBlock(base_channel*4, base_channel*8, norm=norm)
        self.down3 = DownBlock(base_channel*8, base_channel*8, norm=norm) 
        self.down4 = DownBlock(base_channel*8, base_channel*8, norm=norm)
        self.down5 = DownBlock(base_channel*8, base_channel*8, norm=norm)
        
        self.comp = CompressionBlock(img_channel, code_size=code_size)
        self.comp0 = CompressionBlock(base_channel*2, code_size=code_size)
        self.comp1 = CompressionBlock(base_channel*4, code_size=code_size)
        self.comp2 = CompressionBlock(base_channel*8, code_size=code_size)
        self.comp3 = CompressionBlock(base_channel*8, code_size=code_size)
        self.comp4 = CompressionBlock(base_channel*8, code_size=code_size)
        self.comp5 = CompressionBlock(base_channel*8, code_size=code_size)
        
     
        self.skip0 = ConvBlock(base_channel*2, base_channel*2)
        self.skip1 = ConvBlock(base_channel*4, base_channel*4)
        self.skip2 = ConvBlock(base_channel*8, base_channel*8)
    
    def forward(self, x): # Nx3x256x256
        
        d0 = self.down0(x)  # Nx16x128x128 
        d1 = self.down1(d0) # Nx32x64x64
        d2 = self.down2(d1) # Nx64x32x32
        d3 = self.down3(d2) # Nx64x16x16
        d4 = self.down4(d3) # Nx64x8x8
        d5 = self.down5(d4) # Nx64x4x4
        
        s0 = self.skip0(d0) # Nx16x128x128
        s1 = self.skip1(d1) # Nx32x64x64
        s2 = self.skip2(d2) # Nx64x32x32
        
        c0 = self.comp0(d0)
        c1 = self.comp1(d1)
        c2 = self.comp2(d2)
        c3 = self.comp3(d3)
        c4 = self.comp4(d4)
        c5 = self.comp5(d5)
        
        feats = [d5, d4, d3, d2, d1, d0]
        codes = [torch.mean(d5, axis=1), c5, c4, c3, c2, c1, c0, self.comp(x)]
        infos = [s2, s1, s0]
        
        return feats, codes, infos
    

    
class Decoder(nn.Module):
    def __init__(self, img_channel=3, base_channel=64, norm=None):
        super(Decoder, self).__init__()
        
        self.fusion1 = nn.Sequential(nn.Conv2d(288, base_channel, 3, 1, 1), nn.PReLU())         
        self.fusion2 = nn.Sequential(nn.Conv2d(144, base_channel, 3, 1, 1), nn.PReLU())
    
        self.up0 = UpBlock(base_channel, base_channel, norm=norm) # 64x32x32 -> 64x64s64
        self.up1 = UpBlock(base_channel*2, base_channel, norm=norm) # 128x64x64- > 64x128x128
        self.up2 = UpBlock(base_channel*2, base_channel, norm=norm) # 128x128x128 -> 64x256x256
        self.out = nn.Sequential(
            nn.Conv2d(base_channel*2, base_channel, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(base_channel, img_channel, 3, 1, 1),
        )          

    def forward(self, codes, infos): # om 32x32
        
        fuse1 = self.fusion1(torch.cat((codes[0], infos[1]), 1))
        fuse2 = self.fusion2(torch.cat((codes[1], infos[2]), 1))
        
        out1 = self.up0(infos[0]) # out 64x64x64
        out2 = self.up1(torch.cat((out1, fuse1), 1))
        out3 = self.up2(torch.cat((out2, fuse2), 1))
        out = self.out(torch.cat((out3, codes[2]), 1))
        
        return out