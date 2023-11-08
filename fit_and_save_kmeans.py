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
parser.add_argument('--data_path', type=str, default='vctk_trimmed_rave_encoded.h5', help='data file')
parser.add_argument('--csv_path', type=str, default='vctk_trimmed_rave_encoded.csv', help='csv file')
parser.add_argument('--quantizer_num_clusters', type=int, default=512, help='number of clusters for quantization')
parser.add_argument('--quantizer_save_path', type=str, help='path to where to save the fit k-means quantizer')
cmd_args = parser.parse_args()

# %%
# basic config
class Configs(object):
    root_path = cmd_args.root_path
    data_path = cmd_args.data_path
    csv_path = cmd_args.csv_path

    seq_len = 32
    label_len = 16
    pred_len = 8

    batch_size = 512
    num_workers = 0

    scale = 1
    quantize = 1
    quantizer_num_clusters = cmd_args.quantizer_num_clusters

args = Configs()

# %%
# create data loaders for train and val
train_set, train_loader = data_provider_ravenc(args, "train")

# %%
# save fit k-means quantizer to file
train_set.save_quantizer(cmd_args.quantizer_save_path)

# %%
# test loading the quantizer
train_set, train_loader = data_provider_ravenc(args, "train", quantizer=cmd_args.quantizer_save_path)
print(train_set.quantizer.predict(torch.rand(8, 5)))
