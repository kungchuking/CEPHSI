# -- Adapted from https://medium.com/@abbesnessim/mobilenetv2-autoencoder-an-efficient-approach-for-feature-extraction-and-image-reconstruction-9c70ba58947a

import torch
import torch.nn as nn
from torchsummary import summary

from thop import profile
from calflops import calculate_flops

DEBUG = False

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super(InvertedResidualBlock, self).__init__()

        self.stride = stride
        self.expansion_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.ReLU6(inplace=True)
        )

        self.depthwise_layer = nn.Sequential(
            nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=3, stride=stride, padding=1, groups=in_channels * expansion, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.ReLU6(inplace=True)
        )

        self.pointwise_layer = nn.Sequential(
            nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expansion_layer(x)
        out = self.depthwise_layer(out)
        out = self.pointwise_layer(out)

        # if self.stride == 1:
        #     out = out + x
        # -- TODO: Work in Progress
        if DEBUG:
            print ("[ENC-INFO] out.shape: ", out.shape)

        return out

class InvertedResidualBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, stride=1):
        super(InvertedResidualBlockTranspose, self).__init__()

        self.stride = stride

        self.expansion_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True)
        )

        self.depthwise_layer_transpose = nn.Sequential(
            nn.ConvTranspose2d(in_channels * t, in_channels * t, kernel_size=3, stride=stride, padding=1,
                               output_padding=(0 if stride == 1 else 1), groups=in_channels * t, bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True)
        )

        self.pointwise_layer_transpose = nn.Sequential(
            nn.ConvTranspose2d(in_channels * t, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expansion_layer(x)
        out = self.depthwise_layer_transpose(out)
        out = self.pointwise_layer_transpose(out)
        # -- TODO: Work in Progress 
        if DEBUG:
            print ("[DEC-INFO] out.shape: ", out.shape)

        return out


class MobileNetV2CAE(nn.Module):
    def __init__(self,
            in_channels=3,
            out_channels=3,
            num_classes=10,
            frame_n=8,
            n_feats=2):
        super(MobileNetV2CAE, self).__init__()

        self.frame_n = frame_n
        self.encoder = nn.Sequential(
            # -- nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            # -- Modified by Chu King on Oct 27, 2024
            # -- Preserve the original size of the image for the insertion of temporal embeddings later.
            nn.Conv2d(in_channels, 8*n_feats, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8*n_feats),
            nn.ReLU6(inplace=True),
            InvertedResidualBlock(8*n_feats, 4*n_feats, 1, 1),
            InvertedResidualBlock(4*n_feats, 6*n_feats, 6, 2),
            InvertedResidualBlock(6*n_feats, 6*n_feats, 6, 1),
            InvertedResidualBlock(6*n_feats, 8*n_feats, 6, 2),
            InvertedResidualBlock(8*n_feats, 8*n_feats, 6, 1),
            InvertedResidualBlock(8*n_feats, 8*n_feats, 6, 1),
            InvertedResidualBlock(8*n_feats, 16*n_feats, 6, 2),
            InvertedResidualBlock(16*n_feats, 16*n_feats, 6, 1),
            InvertedResidualBlock(16*n_feats, 16*n_feats, 6, 1),
            InvertedResidualBlock(16*n_feats, 16*n_feats, 6, 1),
            InvertedResidualBlock(16*n_feats, 24*n_feats, 6, 1),
            InvertedResidualBlock(24*n_feats, 24*n_feats, 6, 1),
            InvertedResidualBlock(24*n_feats, 24*n_feats, 6, 1),
            InvertedResidualBlock(24*n_feats, 40*n_feats, 6, 2),
            InvertedResidualBlock(40*n_feats, 40*n_feats, 6, 1),
            # InvertedResidualBlock(40*n_feats, 40*n_feats, 6, 1),
            # InvertedResidualBlock(40*n_feats, 80*n_feats, 6, 1)
        )

        self.decoder = nn.Sequential(
            # InvertedResidualBlockTranspose(80*n_feats, 40*n_feats, 6, 1),
            # InvertedResidualBlockTranspose(40*n_feats, 40*n_feats, 6, 1),
            InvertedResidualBlockTranspose(40*n_feats, 40*n_feats, 6, 2),
            InvertedResidualBlockTranspose(40*n_feats, 24*n_feats, 6, 1),
            InvertedResidualBlockTranspose(24*n_feats, 24*n_feats, 6, 1),
            InvertedResidualBlockTranspose(24*n_feats, 24*n_feats, 6, 1),
            InvertedResidualBlockTranspose(24*n_feats, 16*n_feats, 6, 1),
            InvertedResidualBlockTranspose(16*n_feats, 16*n_feats, 6, 1),
            InvertedResidualBlockTranspose(16*n_feats, 16*n_feats, 6, 1),
            InvertedResidualBlockTranspose(16*n_feats, 16*n_feats, 6, 1),
            InvertedResidualBlockTranspose(16*n_feats, 8*n_feats, 6, 2),
            InvertedResidualBlockTranspose(8*n_feats, 8*n_feats, 6, 1),
            InvertedResidualBlockTranspose(8*n_feats, 8*n_feats, 6, 1),
            InvertedResidualBlockTranspose(8*n_feats, 6*n_feats, 6, 2),
            InvertedResidualBlockTranspose(6*n_feats, 6*n_feats, 6, 1),
            InvertedResidualBlockTranspose(6*n_feats, 4*n_feats, 6, 2), # Change stride from 1 to 2
            # -- Modified by Chu King on Oct 27, 2024
            # -- Changed the values of stride and output padding, as the parameters of the first layer of the encoder has been changed.
            # -- Changed the number of output channels from in_channels to frame_n * in_channels
            # -- nn.ConvTranspose2d(4*n_feats, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.ConvTranspose2d(4*n_feats, out_channels*frame_n, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, ce_blur, time_idx=None, ce_code=None):
        # -- TODO: Work in Progress 
        if DEBUG:
            print ("[INPUT-INFO] ce_blur.shape: ", ce_blur.shape)

        x = self.encoder(ce_blur)

        # -- TODO: Work in Progress 
        if DEBUG:
            print ("[LATENT-INFO] x.shape: ", x.shape)

        x = self.decoder(x)

        return torch.reshape(x, (-1, self.frame_n, *ce_blur.shape[-3:]))

"""
if DEBUG:
    model = MobileNetV2CAE(in_channels=1, n_feats=1)
    
    # Instantiate the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # -- print(summary(model, input_size=(3, 32, 32), device=device.type))
    
    # -- z = model(torch.randn(1, 3, 480, 640))
    
    flops, macs, params = calculate_flops(model=model, input_shape=(1, 1, 720, 1280))
    flops, params = profile(model, (torch.randn(1, 1, 720, 1280),))
    print (macs)
    print(f"FLOPs: {flops*1e-9} GFLOPs")
"""
