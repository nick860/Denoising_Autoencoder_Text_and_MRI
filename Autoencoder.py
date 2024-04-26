import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding='same'), # 300x300x3 -> 300x300x64
            nn.ReLU(), # Apply ReLU activation function
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding='same'), # 300x300x64 -> 300x300x128
            nn.ReLU(),
            nn.BatchNorm2d(128),  
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding='same'), # 300x300x128 -> 300x300x64
            nn.BatchNorm2d(64), # Apply Batch Normalization
            nn.MaxPool2d(2)  # Downsample the image to 150x150         
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),  # Upsample the image to 300x300
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 3, 3, padding='same'),
        )       
    def forward(self, x):
        """
        param x: Input image
        return: Output image
        """
        x = self.encoder(x)
        x = self.decoder(x)

        return x
