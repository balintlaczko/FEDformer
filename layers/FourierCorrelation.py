# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com

import numpy as np
import torch
import torch.nn as nn


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


# ########## fourier layer #############
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        # get modes on frequency domain
        self.index_ = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print('modes={}, index={}'.format(modes, self.index_))
        self.register_buffer("index", torch.tensor(self.index_, dtype=torch.long))

        self.scale_ = (1 / (in_channels * out_channels))
        self.register_buffer("scale", torch.tensor(self.scale_, dtype=torch.float32))

        # self.weights1 = nn.Parameter(
        #     self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))
        # redo using simple floats, because NCCL Backend does not support ComplexFloat data type
        # https://github.com/pytorch/pytorch/issues/71613
        self.weights1_real = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.float32))
        self.weights1_imag = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.float32))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        weights1_complex = torch.complex(self.weights1_real, self.weights1_imag)
        for wi, i in enumerate(self.index):
            # out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], weights1_complex[:, :, :, wi])
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return (x, None)


# ########## Fourier Cross Former ####################
class FourierCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        # get modes for queries and keys (& values) on frequency domain
        self.index_q_ = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.register_buffer("index_q", torch.tensor(self.index_q_, dtype=torch.long))
        self.index_kv_ = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)
        self.register_buffer("index_kv", torch.tensor(self.index_kv_, dtype=torch.long))

        print('modes_q={}, index_q={}'.format(len(self.index_q_), self.index_q_))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv_), self.index_kv_))

        self.scale = (1 / (in_channels * out_channels))
        # self.weights1 = nn.Parameter(
        #     self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))
        # redo using simple floats, because NCCL Backend does not support ComplexFloat data type
        # https://github.com/pytorch/pytorch/issues/71613
        self.weights1_real = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.float32))
        self.weights1_imag = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.float32))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # perform attention mechanism on frequency domain
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        weights1_complex = torch.complex(self.weights1_real, self.weights1_imag)
        # xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, weights1_complex)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return (out, None)
    



