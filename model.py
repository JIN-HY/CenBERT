import torch
import torch.nn as nn

from config import *


class GlobalCentFormer(nn.Module):

    def __init__(
        self,
        d_model=D_MODEL,
        nhead=8,
        num_layers=4
    ):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Sequential(

            nn.Linear(d_model, 256),

            nn.GELU(),

            nn.Dropout(0.1),

            nn.Linear(256, 1)
        )

    def forward(self, x):

        x = self.encoder(x)

        out = self.head(x).squeeze(-1)

        return out