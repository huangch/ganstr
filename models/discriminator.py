# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, gene_num, subtype_num):
        super(Discriminator, self).__init__()

        self.fcnLayer = nn.Sequential(
            nn.BatchNorm1d(gene_num),
            nn.Linear(gene_num, 1024),
            nn.LeakyReLU(0.1, inplace=True), # nn.ReLU(inplace=True), # nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25), 
            # nn.BatchNorm1d(1024),
            # nn.Linear(1024, 1024),
            # nn.LeakyReLU(0.1, inplace=True), # nn.ReLU(inplace=True), # nn.LeakyReLU(0.1, inplace=True),
            # nn.Dropout(0.25),
            )

        # Output layers
        self.advLayer = nn.Sequential(nn.BatchNorm1d(1024), nn.Linear(1024, 1), nn.Sigmoid())
        self.auxLayer = nn.Sequential(nn.BatchNorm1d(1024), nn.Linear(1024, subtype_num+1), nn.Softmax(dim=1))
        self.clsLayer = nn.Sequential(nn.BatchNorm1d(1024), nn.Linear(1024, subtype_num), nn.Softmax(dim=1))
        
    def forward(self, trns):
        output = trns
        output = self.fcnLayer(output)
        validity = self.advLayer(output)
        label = self.auxLayer(output)
        cls = self.clsLayer(output)

        return validity, label,cls
