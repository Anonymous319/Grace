import sys
import torch
import torch.nn as nn

class Generator_Conv2(nn.Module):
    def __init__(self, inputsize, class_num, latent_dim=100):
        super(Generator_Conv2, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = inputsize[1]
        self.class_num = class_num
        self.label_emb = nn.Embedding(self.class_num, self.class_num)
        self.output_channel = inputsize[0]
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim + self.class_num, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.LeakyReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, self.output_channel, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, self.label_emb(labels)], 1)
        x = self.fc(x)
        x = x.view(-1, 256,4,4)
        x = self.deconv(x)
        return x
