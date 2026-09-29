import torch
import torch.nn as nn   
import torch.nn.functional as F
from torchvision.transforms import v2

# convolutional block class definition, contains a convolutional layer, leaky ReLU activation, batch normalization, and optional max pooling
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, maxpool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool2d(2) if maxpool else None

    # computes a forward pass for the convolutional block, applying convolution, leaky ReLU activation, batch normalization, and optional max pooling
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leaky_relu(x)        
        if self.pool is not None:
            x = self.pool(x)
        return x

class YOLOV1(nn.Module):
    def __init__(self, num_classes, initial_kernel_size):
        super().__init__()
        self.num_classes = num_classes
        self.initial_kernel_size = initial_kernel_size
        self.stage1 = ConvBlock(3, 64, kernel_size=initial_kernel_size, stride=2, padding=3, maxpool=True)
        self.stage2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1, maxpool=True)
        self.stage3 = nn.Sequential(
            ConvBlock(192, 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1, maxpool=True))
        stage4 = []
        for _ in range(3):
            stage4.extend([
                ConvBlock(512, 256, kernel_size=1),
                ConvBlock(256, 512, kernel_size=3, padding=1),
            ])
        stage4.extend([
            ConvBlock(512, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1, maxpool=True),
        ])
        self.stage4 = nn.Sequential(*stage4)
        self.stage5 = nn.Sequential(
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1)
            )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, num_classes) 
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        # self.batchnorm = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        return self