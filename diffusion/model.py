import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.network(x)

class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.enc_blocks = nn.ModuleList([ConvBlock(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return x, features

class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up_blocks = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i], kernel_size=2, stride=2)
            for i in range(len(channels) - 1)
        ])
        self.dec_blocks = nn.ModuleList([
            ConvBlock(channels[i] * 2, channels[i+1]) for i in range(len(channels) - 1)
        ])

    def forward(self, x, enc_features):
        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x)
            x = torch.cat([x, enc_features[-(i+1)]], dim=1)
            x = self.dec_blocks[i](x)
        return x

class UNet(nn.Module):
    def __init__(self, img_channels=3, num_classes=None):
        super().__init__()
        self.encoder = Encoder([img_channels, 64, 128, 256, 512])
        self.bottleneck = ConvBlock(512, 512)
        self.decoder = Decoder([512, 256, 128, 64])
        self.num_classes = num_classes
        if num_classes:
            self.label_emb = nn.Embedding(num_classes, 512)

        self.final_conv = nn.Conv2d(64, img_channels, kernel_size=3, padding=1)

    def forward(self, x, t, y=None):
        if self.num_classes and y is not None:
            y_emb = self.label_emb(y).unsqueeze(-1).unsqueeze(-1)
            x = x + y_emb
        
        x, enc_features = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, enc_features)
        return self.final_conv(x)


# import torch
# import torch.nn as nn

# class DiffusionUNet(nn.Module):
#     def __init__(self):
#         pass

#     def forward(self, x):
#         pass


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.network(x)


# class Encoder(nn.Module):
#     def __init__(self, channels=[3, 64, 128, 256, 512]):
#         super().__init__()
#         self.enc_blocks = nn.ModuleList([
#             ConvBlock(channels[i], channels[i+1]) for i in range(len(channels) - 1)
#         ])
#         self.pool = nn.MaxPool2d(2)
    
#     def forward(self, x):
#         features = []  # Store feature maps for skip connections
#         for block in self.enc_blocks:
#             x = block(x)
#             features.append(x)
#             x = self.pool(x)  # Reduce spatial size
#         return x, features  # Return bottleneck output and features


# class Decoder(nn.Module):
#     def __init__(self, channels=[512, 256, 128, 64]):
#         super().__init__()
        
#         # ✅ Update upsampling layers
#         self.up_blocks = nn.ModuleList([
#             nn.ConvTranspose2d(channels[i], channels[i], kernel_size=2, stride=2) 
#             for i in range(len(channels) - 1)
#         ])

#         # ✅ Since we concatenate, input to ConvBlock should be doubled
#         self.dec_blocks = nn.ModuleList([
#             ConvBlock(channels[i] * 2, channels[i+1])  # Fix input channels
#             for i in range(len(channels) - 1)
#         ])

#     def forward(self, x, encoder_features):
#         for i in range(len(self.up_blocks)):
#             x = self.up_blocks[i](x)  # Upsample
#             x = torch.cat([x, encoder_features[-(i+1)]], dim=1)  # ✅ Fix: Concatenate along channels
#             x = self.dec_blocks[i](x)  # Pass through ConvBlock with correct input channels
#         return x



# class TimeEmbedding(nn.Module):
#     def __init__(self, emb_dim):
#         super().__init__()
#         self.linear = nn.Linear(1, emb_dim)
    
#     def forward(self, t):
#         t = t.view(-1, 1).float()  # Ensure correct shape
#         return self.linear(t).unsqueeze(-1).unsqueeze(-1)  # Reshape for broadcasting

# class UNet(nn.Module):
#     def __init__(self, img_channels=3, time_emb_dim=256):
#         super().__init__()
#         self.encoder = Encoder([img_channels, 64, 128, 256, 512])
#         self.bottleneck = ConvBlock(512, 512)
#         self.decoder = Decoder([512, 256, 128, 64])
#         self.time_embed = TimeEmbedding(512)
#         self.final_conv = nn.Conv2d(64, img_channels, kernel_size=1)  # Output same shape as input

#     def forward(self, x, t):
#         t_emb = self.time_embed(t)  # Get time embedding
        
#         x, enc_features = self.encoder(x)  # Encode
#         x = self.bottleneck(x) + t_emb  # Inject time info at bottleneck
#         x = self.decoder(x, enc_features)  # Decode
#         x = self.final_conv(x)  # Predict noise
#         return x
