import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()


        # Encoder
        self.enc_conv0 = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 16)
        )
        self.pool0 = nn.MaxPool3d(kernel_size=2, stride=2)  

        self.enc_conv1 = nn.Sequential(
            ConvBlock(16, 64),
            ConvBlock(64, 64)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2) 

        self.enc_conv2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2) 

        # Bottleneck
        self.bottleneck_conv = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 128)
        )

        # Decoder 
        self.upsample0 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv0 = nn.Sequential(
            ConvBlock(256, 128),
            ConvBlock(128, 64)
        )

        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv1 = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 16)
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv2 = nn.Sequential(
            ConvBlock(32, 16),
            ConvBlock(16, 16),
            nn.Conv3d(16, 1, kernel_size=1)  # Final 1x1 conv to adjust the channel dimensions to desired output
        )


    def forward(self, x):
        # Encoder
        e0 = self.enc_conv0(x)
        e0_pool = self.pool0(e0)

        e1 = self.enc_conv1(e0_pool)
        e1_pool = self.pool1(e1)

        e2 = self.enc_conv2(e1_pool)
        e2_pool = self.pool2(e2)


        # Bottleneck
        b = self.bottleneck_conv(e2_pool)

        # Decoder
        d0 = self.dec_conv0(torch.cat((self.upsample0(b), e2), dim=1))
        d1 = self.dec_conv1(torch.cat((self.upsample1(d0), e1), dim=1))
        d2 = self.dec_conv2(torch.cat((self.upsample2(d1), e0), dim=1))


        return  torch.sigmoid(d2)