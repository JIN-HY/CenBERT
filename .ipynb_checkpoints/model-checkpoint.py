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


import torch.nn.functional as F

class TransformerEncoderLayerWithAttention(nn.TransformerEncoderLayer):
    """
    Custom Transformer Encoder Layer that returns attention weights.
    Compatible with PyTorch 1.13
    """
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: [batch, seq_len, d_model]
            src_mask: Optional mask for self-attention
            src_key_padding_mask: Optional padding mask
            
        Returns:
            x: [batch, seq_len, d_model] - encoded output
            attn_weights: [batch, seq_len, seq_len] - averaged attention weights
        """
        x = src
        
        if self.norm_first:
            # First norm, then self-attention
            x, attn_weights = self._sa_block_with_attn(
                self.norm1(x), src_mask, src_key_padding_mask
            )
            # Add residual connection
            x = src + x
            
            # FFN with residual
            x = x + self._ff_block(self.norm2(x))
        else:
            # Original ordering: self-attention first
            x, attn_weights = self._sa_block_with_attn(
                x, src_mask, src_key_padding_mask
            )
            x = self.norm1(x)
            
            # FFN
            x = self.norm2(x + self._ff_block(x))
        
        return x, attn_weights
    
    def _sa_block_with_attn(self, x, attn_mask, key_padding_mask):
        """
        Self-attention block that returns attention weights.
        """
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True  # Returns [batch, seq_len, seq_len]
        )
        return self.dropout1(attn_output), attn_weights

class GlobalCentFormerInterpret(nn.Module):
    
    def __init__(self,
                 d_model=768,
                 nhead=8,
                 num_layers=4):
        
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttention(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=2048,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: [batch, seq_len, d_model]
            return_attention: If True, return attention weights for all layers
        
        Returns:
            out: [batch] - final prediction
            attentions: List of [batch, seq_len, seq_len] - attention weights per layer (optional)
        """
        attentions = []
        
        for layer in self.layers:
            x, attn = layer(x)
            attentions.append(attn)  # Each is [batch, seq_len, seq_len]
        
        out = self.head(x).squeeze(-1)  # [batch]
        
        if return_attention:
            return out, attentions
        
        return out