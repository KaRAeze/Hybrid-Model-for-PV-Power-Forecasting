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

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dims, lstm_units):
        super(CNNLSTMModel, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=input_dims, out_channels=64, kernel_size=1)
        self.bn1d_1 = nn.BatchNorm1d(64)  
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2)  
        self.conv1d_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)  
        self.bn1d_2 = nn.BatchNorm1d(128)  
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2) 
        self.dropout = nn.Dropout(0.2)
        self.bilstm = nn.LSTM(input_size=128, hidden_size=lstm_units, bidirectional=True, batch_first=True)
        self.dense = nn.Linear(2 * lstm_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = inputs.permute(0, 2, 1)  # convert to (batch_size, input_dims, time_steps)
        x = self.conv1d_1(x)
        x = F.relu(x)
        x = self.bn1d_1(x)  
        x = self.pool1d_1(x)  
        x = self.conv1d_2(x)  
        x = F.relu(x)
        x = self.bn1d_2(x) 
        x = self.pool1d_2(x)  
        x = x.permute(0, 2, 1)  # convert to (batch_size, time_steps, input_dims)
        x = self.dropout(x)
        lstm_out, _ = self.bilstm(x)
        lstm_out = lstm_out[:, -1, :]  # take the output of the last time step
        output = self.dense(lstm_out)
        output = self.sigmoid(output)
        return output
    
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
