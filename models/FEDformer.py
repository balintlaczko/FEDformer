import lightning.pytorch as pl
import numpy as np
import math
import sys
sys.path.append("..")
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from layers.SelfAttention_Family import FullAttention, ProbAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, TokenEmbedding, DataEmbedding_onlypos
import torch
import torch.nn as nn
import torch.nn.functional as F


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LitFEDformer(pl.LightningModule):
    def __init__(self, args):
        super(LitFEDformer, self).__init__()
        self.args = args
        self.model = Model_noif(self.args)
        self.loss = nn.MSELoss()
        # self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        # decoder input
        # create a tensor taking the labels and extending it with zeros for the predictions
        # containter for the last pred_len time steps
        dec_inp = torch.zeros_like(
            batch_y[:, -self.args.pred_len:, :]).float()
        # concatenate the first label_len time steps with the container
        dec_inp = torch.cat(
            [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

        outputs = self.model(batch_x, None, dec_inp, None)

        batch_y = batch_y[:, -self.args.pred_len:, :]

        loss = self.loss(outputs, batch_y)

        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        # decoder input
        # create a tensor taking the labels and extending it with zeros for the predictions
        # containter for the last pred_len time steps
        dec_inp = torch.zeros_like(
            batch_y[:, -self.args.pred_len:, :]).float()
        # concatenate the first label_len time steps with the container
        dec_inp = torch.cat(
            [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

        outputs = self.model(batch_x, None, dec_inp, None)

        batch_y = batch_y[:, -self.args.pred_len:, :]

        loss = self.loss(outputs, batch_y)

        self.log('val_loss', loss, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.999)
        return [optimizer], [scheduler]


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.embed = configs.embed

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        # d_model = 512
        # we use "token_only" for RAVE embeddings datasets
        if self.embed.lower() == 'token_only':
            self.enc_embedding = TokenEmbedding(
                configs.enc_in, configs.d_model)
            self.dec_embedding = TokenEmbedding(
                configs.dec_in, configs.d_model)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        if configs.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=configs.modes,
                                                  ich=configs.d_model,
                                                  base=configs.base,
                                                  activation=configs.cross_activation)
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len//2+self.pred_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                      out_channels=configs.d_model,
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=configs.modes,
                                                      mode_select_method=configs.mode_select)
        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        dec_modes = int(
            min(configs.modes, (configs.seq_len//2+configs.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        The main forward call of FEDformer

        Args:
            x_enc (torch.Tensor): The input sequence for the encoder. Shape: [B, T, C] where T = seq_len, C is the number of features
            x_mark_enc (torch.Tensor, optional): The encoded time features for the encoder. Shape: [B, T, C] where T = seq_len, C is the number of encoded time features (seconds, minutes, hours, etc.)
            x_dec (torch.Tensor, optional): The label sequence (extended with zeros for predictions) for the decoder. Shape: [B, T, C], where T = label_len + pred_len
            x_mark_dec (torch.Tensor, optional): The encoded time features for the labels and the predictions for the decoder. Shape: [B, T, C], where T = label_len + pred_len

        Returns:
            torch.Tensor: The predictions for the decoder. Shape: [B, pred_len, C] (I hope)
        """
        # decomp init
        # take the means per feature, and repeat it for the prediction length
        # print(f'x_enc.shape: {x_enc.shape}')
        # print(f'self.pred_len: {self.pred_len}')
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)  # [B, T, C], t = pred_len
        # this seems unused
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len,
        #                     x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(
            x_enc)  # [B, T, C], [B, T, C], t = seq_len
        # decoder input
        # concatenate the last label_len values of the trend_init with the mean that is pred_len long
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)  # [B, T, C], t = label_len + pred_len
        # add zeros like [B, T, C], t = pred_len to the end of the seasonal_init
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))  # [B, T, C], t = label_len + pred_len
        # enc
        if self.embed.lower() == 'token_only':
            enc_out = self.enc_embedding(x_enc)
        else:
            enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, 512]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        if self.embed.lower() == 'token_only':
            dec_out = self.dec_embedding(x_dec)
        else:
            dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            # [B, L, D] (B, pred_len, 512)? or (B, pred_len, C)?
            return dec_out[:, -self.pred_len:, :]


class Model_noif(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, configs):
        super(Model_noif, self).__init__()
        # self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        # self.output_attention = configs.output_attention
        self.output_attention = False
        self.embed = configs.embed

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        # d_model = 512
        # we use "token_only" for RAVE embeddings datasets
        if self.embed.lower() == 'token_only':
            self.enc_embedding = TokenEmbedding(
                configs.enc_in, configs.d_model)
            self.dec_embedding = TokenEmbedding(
                configs.dec_in, configs.d_model)
        elif self.embed.lower() == 'token_pos':
            self.enc_embedding = DataEmbedding_onlypos(
                configs.enc_in, configs.d_model)
            self.dec_embedding = DataEmbedding_onlypos(
                configs.dec_in, configs.d_model)
        else:
            raise NotImplementedError("Embedding type not implemented")


        encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                        out_channels=configs.d_model,
                                        seq_len=self.seq_len,
                                        modes=configs.modes,
                                        mode_select_method=configs.mode_select)
        decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                        out_channels=configs.d_model,
                                        seq_len=self.seq_len//2+self.pred_len,
                                        modes=configs.modes,
                                        mode_select_method=configs.mode_select)
        decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                    out_channels=configs.d_model,
                                                    seq_len_q=self.seq_len//2+self.pred_len,
                                                    seq_len_kv=self.seq_len,
                                                    modes=configs.modes,
                                                    mode_select_method=configs.mode_select)
        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        dec_modes = int(
            min(configs.modes, (configs.seq_len//2+configs.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        The main forward call of FEDformer.
        The encoder input is the batch_x sequence (with seq_len time steps and 8 features).
        The decoder input is the labels part of the batch_y sequence (label_len long) extended with zeros for the predictions (pred_len long).

        Args:
            x_enc (torch.Tensor): The input sequence for the encoder. Shape: [B, T, C] where T = seq_len, C is the number of features
            x_mark_enc (torch.Tensor, optional): The encoded time features for the encoder. Shape: [B, T, C] where T = seq_len, C is the number of encoded time features (seconds, minutes, hours, etc.)
            x_dec (torch.Tensor, optional): The label sequence (extended with zeros for predictions) for the decoder. Shape: [B, T, C], where T = label_len + pred_len
            x_mark_dec (torch.Tensor, optional): The encoded time features for the labels and the predictions for the decoder. Shape: [B, T, C], where T = label_len + pred_len

        Returns:
            torch.Tensor: The predictions for the decoder. Shape: [B, pred_len, C] (I hope)
        """

        # print(f'Encoder input: x_enc.shape: {x_enc.shape}') # [B, T, C], t = seq_len
        # print(f'Prediction length: self.pred_len: {self.pred_len}') # t = pred_len
        # decomp init
        # take the mean feature vector across all time steps, and repeat it for the prediction length
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)  # [B, T, C], t = pred_len
        # print(f'mean.shape: {mean.shape}')
        seasonal_init, trend_init = self.decomp(x_enc)  # [B, T, C], [B, T, C], t = seq_len
        # print(f'seasonal_init.shape: {seasonal_init.shape}')
        # print(f'trend_init.shape: {trend_init.shape}')
        
        # decoder input
        # concatenate the last label_len values of the trend_init with the mean that is pred_len long
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)  # [B, T, C], t = label_len + pred_len
        # print(f'Labels from trend and the mean for preds: trend_init.shape: {trend_init.shape}')
        # add zeros like [B, T, C], t = pred_len to the labels part (here: 2nd half) of seasonal_init
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))  # [B, T, C], t = label_len + pred_len
        # print(f'Labels from seasonal and zeros for preds: seasonal_init.shape: {seasonal_init.shape}')

        # enc
        enc_out = self.enc_embedding(x_enc) # [B, T, d_model], d_model = 512, t = seq_len
        # print(f'Embedded encoder input: enc_out.shape: {enc_out.shape}')
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) # [B, T, d_model], d_model = 512, t = seq_len AND len(attns) = 2
        # print(f'Encoder output: enc_out.shape: {enc_out.shape}')
        # print(f'len(attns): {len(attns)}') # len(attns) == e_layers == 2
        # print(f'attns: {attns}') # None, because output_attention = False

        # dec
        # print(f'Decoder input (labels from sequence and zeros for pred_len): x_dec.shape: {x_dec.shape}') # [B, T, C], t = label_len + pred_len
        dec_out = self.dec_embedding(x_dec) # [B, T, d_model], d_model = 512, t = label_len + pred_len
        # print(f'Embedded decoder input (labels from sequence and zeros for pred_len): dec_out.shape: {dec_out.shape}')
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # print(f'Decoder out / seasonal: seasonal_part.shape: {seasonal_part.shape}')
        # print(f'Decoder out / trend: trend_part.shape: {trend_part.shape}')

        # final
        dec_out = trend_part + seasonal_part
        # print(f'Seasonal + trend: dec_out.shape: {dec_out.shape}')
        final_out = dec_out[:, -self.pred_len:, :]
        # print(f'Chopping off only predictions: final_out.shape: {final_out.shape}')

        return final_out


if __name__ == '__main__':
    class Configs(object):
        # ab = 0
        # version = 'Fourier'
        mode_select = 'random'
        modes = 64
        L = 3
        base = 'legendre'
        cross_activation = 'tanh'

        seq_len = 32
        label_len = 16
        pred_len = 8

        enc_in = 8
        dec_in = 8
        c_out = 8
        d_model = 512
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 2048
        moving_avg = 6
        dropout = 0.05
        activation = 'gelu'
        output_attention = False

        # embed = 'timeF'
        # embed = 'token_only'
        embed = 'token_pos'
        # freq = 'h'
        # factor = 1
        # wavelet = 0

        batch_size = 512

    configs = Configs()
    model = Model_noif(configs)

    print('parameter number is {}'.format(sum(p.numel()
          for p in model.parameters())))
    print()

    # batch_x is the input sequence: seq_len time steps with 8 features
    batch_x = torch.randn([configs.batch_size, configs.seq_len, 8]) # [B, T, C], t = seq_len
    # batch_y is the label sequence: label_len + pred_len time steps with 8 features
    batch_y = torch.randn([configs.batch_size, configs.label_len+configs.pred_len, 8]) # [B, T, C], t = label_len + pred_len

    # decoder input
    # create a tensor taking the labels and extending it with zeros for the predictions
    # containter for the last pred_len time steps
    dec_inp = torch.zeros_like(batch_y[:, -configs.pred_len:, :]).float() # [B, T, C], t = pred_len
    # concatenate the first label_len time steps with the container (which are zeros for pred_len time steps)
    dec_inp = torch.cat([batch_y[:, :configs.label_len, :], dec_inp], dim=1).float() # [B, T, C], t = label_len + pred_len

    outputs = model(batch_x, None, dec_inp, None) # [B, T, C], t = pred_len

    # slice out the predictions from the end of the labels
    batch_y = batch_y[:, -configs.pred_len:, :] # [B, T, C], t = pred_len

    # compare the predictions with the labels
    loss = torch.nn.MSELoss()(outputs, batch_y)

    print(f"loss (model predictions MSEd with the truth at prediction time steps from training data): {loss}")
