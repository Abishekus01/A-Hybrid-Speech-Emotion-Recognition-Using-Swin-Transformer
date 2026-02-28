import torch
import torch.nn as nn

class SwinSERModel(nn.Module):
    def __init__(self, num_classes=7, handcrafted_dim=33):
        super(SwinSERModel, self).__init__()

        # CNN branch for log-mel spectrogram
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4))
        )
        self.cnn_fc = nn.Linear(128*4*4, 256)

        # Handcrafted branch
        self.handcrafted_fc = nn.Sequential(
            nn.Linear(handcrafted_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(256+128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_spec, x_hand):
        # Ensure x_spec shape: (B,1,H,W)
        if x_spec.dim()==5:
            x_spec = x_spec.squeeze(1)
        x = self.cnn(x_spec)
        x = x.view(x.size(0), -1)
        x = self.cnn_fc(x)
        h = self.handcrafted_fc(x_hand)
        fused = torch.cat((x,h), dim=1)
        out = self.fusion(fused)
        return out