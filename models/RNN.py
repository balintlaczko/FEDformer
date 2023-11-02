import lightning.pytorch as pl
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
import gin.torch

@gin.configurable
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # self.activation = nn.Tanh()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hn = self.gru(x)
        # out = self.activation(out[:,  -1, :])
        # out = self.fc(out)
        out = self.fc(out[:, -1, :])
        return out, hn

@gin.configurable
class RNNModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, learning_rate, lr_gamma):
        super().__init__()
        self.save_hyperparameters()
        self.model = RNN(input_size, hidden_size, output_size, num_layers, dropout)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.lr_gamma = lr_gamma

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # h0 = torch.zeros(self.hparams.num_layers, x.size(0), self.hparams.hidden_size)
        y_hat, hn = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # h0 = torch.zeros(self.hparams.num_layers, x.size(0), self.hparams.hidden_size)
        y_hat, hn = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        # h0 = torch.zeros(self.hparams.num_layers, x.size(0), self.hparams.hidden_size)
        y_hat, hn = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.lr_gamma)
        return [optimizer], [scheduler]
