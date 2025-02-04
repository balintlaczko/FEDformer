import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    """Classic sine/cosine positional embedding"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """A 1D convolutional layer with kernel size 3"""

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class StepOffset(nn.Module):
    """Offset values by the time step * step_offset"""
    
    def __init__(self, step_offset=1):
        super(StepOffset, self).__init__()
        self.step_offset = step_offset

    def forward(self, x):
        T = x.size(1)
        offset = torch.arange(0, T, device=x.device).float() * self.step_offset / (T - 1) + 1
        offset = offset.view(1, -1, 1)
        return x * offset


class SteppedTokenEmbedding(nn.Module):

    def __init__(self, c_in, d_model, step_offset=1):
        super(SteppedTokenEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.step_offset = StepOffset(step_offset=step_offset)

    def forward(self, x):
        return self.step_offset(self.token_embedding(x))


class FixedEmbedding(nn.Module):
    """Seems to be the same sine/cosine positional embedding as in PositionalEmbedding but this one wraps it into a nn.Embedding with fixed weights"""

    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """A learnable embedding for each time feature (month, day, weekday, hour, minute) that sum up to the final embedding"""

    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """A Linear layer that transforms a frequency map to the final embedding"""

    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        # taken from timefeatures.py
        """
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
        """
        # make an order from year/month at 1 to second at 6
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    """Using all the embeddings above to create the final embedding"""

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(
            x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_onlypos(nn.Module):
    """Using only positional embeddings"""

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_onlypos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """Using temporal embedding and the conv1d positional embedding but not the sine/cosine positional embedding"""

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        # a Conv1d layer with kernel size 3
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # a sine/cosine positional embedding (unused?!)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # a learnable embedding for each time feature (month, day, weekday, hour, minute) that sum up to the final embedding (TemporalEmbedding)
        # or a Linear layer that transforms a frequency map to the final embedding (TimeFeatureEmbedding)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # try:
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        # except:
        #     a = 1
        return self.dropout(x)
    
class DataEmbedding_wo_pos_2(nn.Module):
    """Using temporal embedding and the conv1d positional embedding but not the sine/cosine positional embedding"""

    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_wo_pos_2, self).__init__()

        # a Conv1d layer with kernel size 3
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # a Linear layer that transforms a frequency map to the final embedding (TimeFeatureEmbedding)
        self.time_feature_embedding = TimeFeatureEmbedding(d_model=d_model, freq='a') # freq='a' means yearly, so 1D
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.time_feature_embedding(x_mark)
        return self.dropout(x)
