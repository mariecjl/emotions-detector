import torch
import torch.nn as nn

class EmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        #mlp design
        self.net = nn.Sequential(
            #linear, fully connected layer
            nn.Linear(1434, 256),
            nn.ReLU(),
            #dropouts
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    #forward prop
    def forward(self, x):
        return self.net(x)
