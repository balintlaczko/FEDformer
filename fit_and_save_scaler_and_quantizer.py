# %%
# imports
import os
import torch
import numpy as np
import lightning.pytorch as pl
from data_provider.data_factory import data_provider_ravenc
from models import FEDformer
import soundfile as sf
import argparse
import tqdm

# %%
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='data/RAVE_encoded_datasets', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='vctk_trimmed_rave_encoded_concat_subjects_chunked.h5', help='data file')
parser.add_argument('--csv_path', type=str, default='vctk_trimmed_rave_encoded_concat_subjects_chunked.csv', help='csv file')
parser.add_argument('--scaler_save_path', type=str, default='checkpoints/scaler_vctk.pkl', help='path to where to save the fit scaler')
parser.add_argument('--quantizer_num_clusters', type=int, default=8192, help='number of clusters for quantization')
parser.add_argument('--quantizer_save_path', type=str, default='checkpoints/quantizer_vctk_8192.pt', help='path to where to save the fit k-means quantizer')
cmd_args = parser.parse_args()

# %%
# basic config
class Configs(object):
    root_path = cmd_args.root_path
    data_path = cmd_args.data_path
    csv_path = cmd_args.csv_path

    seq_len = 256
    label_len = 128
    pred_len = 4

    batch_size = 512
    num_workers = 0

    scale = 1
    scaler_type = 'robust'
    quantize = 1
    quantizer_type = 'kmeans'
    quantizer_num_clusters = cmd_args.quantizer_num_clusters

    filter_vctk = 1

args = Configs()

# %%
# create data loaders for train and val
# this will automatically fit a scaler and a quantizer to the data
train_set, train_loader = data_provider_ravenc(args, "train")

# %%
# save fit scaler to file
train_set.save_scaler(cmd_args.scaler_save_path)
# save fit k-means quantizer to file
train_set.save_quantizer(cmd_args.quantizer_save_path)

# %%
# test loading the scaler and the quantizer
# train_set, train_loader = data_provider_ravenc(args, "train", scaler=cmd_args.scaler_save_path)
# print()
# print("Quantizer num clusters:", cmd_args.quantizer_num_clusters)
# train_set.scale = False
# train_set.quantize = False
# x_unscaled, y_unscaled = train_set[0]
# print(x_unscaled.shape, y_unscaled.shape)
# x_unscaled = torch.cat([x_unscaled, y_unscaled[128:, :]], dim=0)
# print("UNSCALED", x_unscaled.shape, x_unscaled.min(), x_unscaled.max())
# train_set.scale = True
# train_set.quantize = True
# x_scaled, y_scaled = train_set[0]
# x_scaled = torch.cat([x_scaled, y_scaled[128:, :]], dim=0)
# print("SCALED", x_scaled.shape, x_scaled.min(), x_scaled.max())
# print()
# x_inv = train_set.inverse_transform(x_scaled)
# print()
# print("INVERSE", x_inv.shape, x_inv.min(), x_inv.max())
# print("LOSS", torch.nn.functional.mse_loss(x_unscaled, x_inv))
# diff = x_unscaled - x_inv
# print("DIFF", diff.min(), diff.max())
# print()
