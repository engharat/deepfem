from torch.utils.data import random_split
from torch import nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.models import  ResNet18_Weights
import copy
from transformers import ConvNextConfig, ConvNextModel
import math 
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from utils import *
import torch
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]  # Add positional encoding
        return x

class Transformer2DPointsModel(nn.Module):
    def __init__(self, input_dim,input_seq_len, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout=0.1):
        super(Transformer2DPointsModel, self).__init__()
        self.d_model = d_model
        self.linear_in = nn.Linear(input_dim, d_model)  # Project input to d_model
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_encoder_layers
        )
        self.attention = AttentionLayer(d_model)  # Add attention layer
        #self.linear_out = nn.Linear(d_model, output_dim)  # Map to the desired output size
        self.linear_out = nn.Linear(d_model*input_seq_len, output_dim)  # Map to the desired output size

    def forward(self, src, src_mask=None):
        
        # src shape: (batch_size, seq_len, input_dim)
        src = self.linear_in(src)  # Project input to d_model
        src = self.positional_encoding(src)  # Add positional encoding
        #memory = self.transformer_encoder(src, src_key_padding_mask=src_mask)  # Pass through transformer encoder
                # memory shape: (batch_size, seq_len, d_model)
        # Zero out padded tokens using src_mask
        if src_mask is not None:
            memory = memory * src_mask.unsqueeze(-1)
            src_key_padding_mask = (src_mask == 0) # Convert 0/1 mask to boolean
            memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        else:
            memory = self.transformer_encoder(src)
            
        # Apply attention mechanism
        #aggregated = self.attention(memory, src_mask)  # Shape: (batch_size, d_model)

        # Map to the desired output size
        #output = self.linear_out(aggregated)  # Shape: (batch_size, output_dim)

        memory = memory.view(memory.shape[0], -1)  # Shape: (batch_size, d_model, seq_len)
        # Map to the desired output size
        output = self.linear_out(memory)  # Shape: (batch_size, output_dim)
        return output

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=288, out_features=2493, bias=True)
        self.fc2 = nn.Linear(in_features=2493, out_features=1386, bias=True)
        self.fc3 = nn.Linear(in_features=1386, out_features=280, bias=True)
        self.relu = nn.ReLU()
    def forward(self, src, src_mask=None):
        src = src.view(src.shape[0],-1)
        x = self.fc1(self.relu(src))
        x = self.fc2(self.relu(x))
        x = self.fc3(self.relu(x))
        return x

    

class CircularConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride)

    def forward(self, x):
        pad = self.kernel // 2
        x = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)
        return self.conv(x)


class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv1 = CircularConv1d(channels, channels, kernel_size)
        self.conv2 = CircularConv1d(channels, channels, kernel_size)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = self.act(self.norm(self.conv1(x)))
        x = self.norm(self.conv2(x))
        return self.act(x + residual)

class CircularCNN2DPointsModel(nn.Module):
    def __init__(
        self,
        input_dim=6,
        hidden_dim=256,     # ↓ reduce width
        num_blocks=4,
        output_points=3026
    ):
        super().__init__()

        self.embed = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        self.blocks = nn.Sequential(*[
            ResBlock1D(hidden_dim, kernel_size=3)
            for _ in range(num_blocks)
        ])

        # Point-wise decoder (much cheaper!)
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 3, kernel_size=1)
        )

        self.output_points = output_points

    def forward(self, x,src_mask=None):
        """
        x: (B, 380, 6)
        """
        x = x.transpose(1, 2)          # (B, 6, 380)
        x = self.embed(x)              # (B, H, 380)
        x = self.blocks(x)             # (B, H, 380)
        x = self.decoder(x)            # (B, 3, 380)

        # Resample or upsample if needed
        x = nn.functional.interpolate(
            x,
            size=self.output_points,
            mode='linear',
            align_corners=False
        )

        return x.flatten(1)            # (B, 9078)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv = CircularConv1d(in_ch, out_ch, kernel=3, stride=stride)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x, target_len):
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return self.act(self.norm(self.conv(x)))



class DeformationCNN1D(nn.Module):
    def __init__(self, input_dim=6, base_ch=64, output_points=3026):
        super().__init__()
        self.output_points = output_points

        # -------- Encoder --------
        self.embed = nn.Conv1d(input_dim, base_ch, kernel_size=1)
        self.enc1 = EncoderBlock(base_ch, 128, stride=2)  # 380 → 190
        self.enc2 = EncoderBlock(128, 256, stride=2)      # 190 → 95
        self.enc3 = EncoderBlock(256, 256, stride=1)      # refinement, 95 → 95

        # -------- Bottleneck --------
        self.fc = nn.Sequential(
            nn.Linear(256 * 95, output_points*3),
            nn.ReLU(),
            nn.Linear(output_points*3, 256 * 95),
            nn.ReLU()
        )

        # -------- Decoder --------
        self.dec1 = DecoderBlock(256, 256)  # 95 → 190
        self.dec2 = DecoderBlock(256, 128)  # 190 → 380
        self.dec3 = DecoderBlock(128, 64)   # 380 → 3026

        self.out = nn.Conv1d(64, 3, kernel_size=1)

    def forward(self, x,src_mask=None):
        """
        x: (B, 380, 6)
        returns: (B, 9078)
        """
        x = x.transpose(1, 2)  # (B, 6, 380)
        x = self.embed(x)

        # Encoder
        x = self.enc1(x)       # 380 → 190
        x = self.enc2(x)       # 190 → 95
        x = self.enc3(x)       # 95 → 95

        # Bottleneck
        b, c, l = x.shape
        x = self.fc(x.view(b, -1)).view(b, c, l)

        # Decoder
        x = self.dec1(x, 190)      # 95 → 190
        x = self.dec2(x, 380)      # 190 → 380
        x = self.dec3(x, self.output_points)  # 380 → 3026

        # Final output
        x = self.out(x)            # (B, 3, 3026)
        return x.flatten(1)        # (B, 9078)



import torch
import torch.nn as nn

class VanillaBiLSTM1D(nn.Module):
    def __init__(
        self,
        input_dim=6,
        hidden_dim=512,
        num_layers=4,
        output_points=3026,
        dropout=0.0
    ):
        super().__init__()

        self.seq_len = 380
        self.output_dim = output_points * 3

        # Input projection (same role as Transformer linear_in)
        self.linear_in = nn.Linear(input_dim, hidden_dim)

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Merge forward/backward directions: 1024 → 512
        self.merge_directions = nn.Linear(
            2 * hidden_dim,
            hidden_dim
        )

        # -------- Giant fully connected layer --------
        # (B, 380 * 512) → (B, 9078)
        self.fc = nn.Linear(
            hidden_dim * self.seq_len,
            self.output_dim
        )

    def forward(self, x, src_mask=None):
        """
        x: (B, 380, 6)
        returns: (B, 9078)
        """
        # Project input
        x = self.linear_in(x)      # (B, 380, 512)

        # LSTM encoding
        x, _ = self.lstm(x)        # (B, 380, 1024)

        # Merge bidirectional features
        x = self.merge_directions(x)  # (B, 380, 512)

        # Flatten all points
        x = x.flatten(1)           # (B, 380*512)

        # Final projection
        x = self.fc(x)             # (B, 9078)
        return x



class VanillaCNN1D(nn.Module):
    def __init__(self, input_dim=6, output_points=3026, base_ch=64):
        super().__init__()
        self.output_points = output_points

        # -------- Encoder --------
        self.embed = nn.Conv1d(input_dim, base_ch, kernel_size=1)
        self.enc1 = EncoderBlock(base_ch, 128, stride=1)   # 380 → 380
        self.enc2 = EncoderBlock(128, 256, stride=1)       # 380 → 380
        self.enc3 = EncoderBlock(256, 512, stride=1)       # 380 → 380
        self.enc4 = EncoderBlock(512, 512, stride=1)       # 380 → 380

        # -------- Giant fully-connected layer --------
        # Flatten entire feature map: (B, 512*380)
        self.fc = nn.Linear(512 * 380, output_points * 3)  # 512*380 → 3026*3 = 9078

    def forward(self, x, src_mask=None):
        """
        x: (B, 380, 6)
        returns: (B, 9078)
        """
        x = x.transpose(1, 2)  # (B, 6, 380)
        x = self.embed(x)

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        # Flatten everything
        x = x.flatten(1)        # (B, 512*380)
        x = self.fc(x)          # (B, 9078)
        return x
