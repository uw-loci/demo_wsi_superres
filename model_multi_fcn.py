import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
# ResU
    
#####################################
#   Convolutional Skip Connection
#####################################

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
class SkipBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(SkipBlock, self).__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1
            ),
            nn.InstanceNorm2d(out_features, affine=True),
            nn.Conv2d(
                in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1, bias=False
            ),
            nn.InstanceNorm2d(out_features, affine=True),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        return self.skip(x)

##############################
#          RES U-NET
##############################

# Downsampling layers with residual connection
# 2 layers
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 4, 2, 1),
            nn.InstanceNorm2d(out_size, affine=True),
            nn.Conv2d(out_size, out_size, 3, 1, 1),
            nn.InstanceNorm2d(out_size, affine=True),
            nn.Conv2d(out_size, out_size, 1, 1, bias=False),
            nn.InstanceNorm2d(out_size, affine=True),
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.1, True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        side = [
            nn.Conv2d(in_size, out_size, 2, 2, bias=False),
        ]
        self.model = nn.Sequential(*layers)
        self.side = nn.Sequential(*side)

    def forward(self, x):
        x = self.model(x) + self.side(x)
        return x

# Upsampling layers with residual connection
# 2 layers
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 3, 1, 1),
            nn.InstanceNorm2d(out_size, affine=True),
            nn.Conv2d(out_size, out_size, 1, 1, bias=False),
            nn.InstanceNorm2d(out_size, affine=True),
            nn.LeakyReLU(0.1, True),
        ]
        side = [
            nn.Conv2d(in_size, out_size, 1, 1, bias=False),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)
        self.side = nn.Sequential(*side)

    def forward(self, x, skip_input):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.model(x) + self.side(x)
        x = torch.cat((x, skip_input), 1)

        return x

# 6 downsampling + 5 skip connection + pixelshuffle
class ResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.skip1 = SkipBlock(64, 64)
        self.down2 = UNetDown(64, 128)
        self.skip2 = SkipBlock(128, 128)
        self.down3 = UNetDown(128, 256)
        self.skip3 = SkipBlock(256, 256)
        self.down4 = UNetDown(256, 256)
        self.skip4 = SkipBlock(256, 256)
        self.down5 = UNetDown(256, 256)
        self.skip5 = SkipBlock(256, 256)
        self.down6 = UNetDown(256, 256)

        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(512, 256)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.final3c = nn.Sequential(
            nn.Conv2d(128, 12, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Sigmoid()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, self.skip5(d5))
        u2 = self.up2(u1, self.skip4(d4))
        u3 = self.up3(u2, self.skip3(d3))
        u4 = self.up4(u3, self.skip2(d2))
        u5 = self.up5(u4, self.skip1(d1))


        return self.final3c(u5)
    
class OneNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(OneNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.skip1 = SkipBlock(64, 64)
        self.skip2 = SkipBlock(64, 64)
        self.skip3 = SkipBlock(64, 64)
        self.skip4 = SkipBlock(64, 64)
        self.skip5 = SkipBlock(64, 64)
        self.final = nn.Sequential(
            nn.Conv2d(128, 12, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Sigmoid()
        )
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.skip1(d1)
        d3 = self.skip2(d2+d1)
        d4 = self.skip3(d3+d2)
        d5 = self.skip4(d4+d3)
        d6 = self.skip4(d5+d4)
        
        return self.final(torch.cat((d6, d1), 1))
        
        
##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 128, normalization=False),
            *discriminator_block(128, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)