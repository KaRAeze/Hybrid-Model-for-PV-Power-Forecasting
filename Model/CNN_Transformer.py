import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time 
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime, time
import torch.nn.functional as F

class Conv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Conv1dLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)  

    def forward(self, x):
        # x shape: (batch_size, seq_len, in_channels)
        x = x.transpose(1, 2)  # (batch_size, in_channels, seq_len)
        x = self.conv(x)
        x = F.relu(x)  
        x = self.bn(x)  
        x = x.transpose(1, 2)  # (batch_size, seq_len, out_channels)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, residual):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(residual + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x
    
class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)  
        self.final_linear = nn.Linear(d_model, 1)  # output dimension : 1

    def forward(self, query, key, value, residual):
        attn_output = self.cross_attn(query, key, value)
        attn_output = self.dropout(attn_output)  #  Dropout 
        x = self.norm1(residual + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        final_output = self.final_linear(x)
        return final_output

class CustomTransformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, seq_len, n_features=5): # "n_features" represents the number of features at station and needs to be adjusted manually
        super(CustomTransformer, self).__init__()
        self.conv_s = Conv1dLayer(n_features, d_model)  
        self.conv_y = Conv1dLayer(1, d_model) 
        self.encoder_s = EncoderLayer(d_model, num_heads, d_ff)
        self.encoder_y = EncoderLayer(d_model, num_heads, d_ff)
        self.cross_attn = CrossAttentionLayer(d_model, num_heads, d_ff)
        self.final_output_linear = nn.Linear(seq_len, 1)  

    def forward(self, x):
        s = x[:, :, :-1]  # (batch_size, seq_len, n_features)
        y = x[:, :, -1].unsqueeze(-1)  # (batch_size, seq_len, 1)

        # Process s
        s_conv = self.conv_s(s)
        S = self.encoder_s(s_conv, s_conv)

        # Process y
        y_conv = self.conv_y(y)
        Y = self.encoder_y(y_conv, y_conv)

        # Cross attention
        output = self.cross_attn(S, S, Y, Y)
        output = output.squeeze(-1)  # (batch_size, seq_len)
        return output