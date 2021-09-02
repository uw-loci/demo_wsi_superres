import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class EDLatentBank(nn.Module):
    def __init__(self, encoder, decoder, latent_bank):
        super(EDLatentBank, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.latent_bank = latent_bank
        
    def forward(self, img, step=6, bank=True, code_dim=512):
        device = next(self.encoder.parameters()).device
        feats, codes, infos = self.encoder(img)
        gen = torch.randn(img.shape[0], code_dim, device=device)
        bank_codes = self.latent_bank(gen, step=step, bank=bank, feats=feats, codes=codes)
        out = self.decoder(codes=bank_codes, infos=infos)
        
        return out
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, norm=None):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, norm=None):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if norm == 'instance':
                layers.append(nn.InstanceNorm2d(out_filters))
            if norm == 'batch':
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, norm),
            *discriminator_block(64, 128, norm),
            *discriminator_block(128, 256, norm),
            *discriminator_block(256, 512, norm),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        """Concatenate image and condition image by channels to produce input"""
        img_B = F.interpolate(img_B, size=(img_A.shape[2], img_A.shape[3]))
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
    
        
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)