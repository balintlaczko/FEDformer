# %%
# imports
import os
import torch
import numpy as np
import lightning.pytorch as pl
from data_provider.data_factory import data_provider_ravenc
import soundfile as sf
import argparse
import tqdm
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from models import FEDformer

# %%
# basic config
class Configs(object):
    # ab = 0
    # version = 'Fourier'
    mode_select = 'random'
    modes = 64
    L = 3
    base = 'legendre'
    cross_activation = 'tanh'

    root_path = "data/RAVE_encoded_datasets"
    data_path = "vctk_trimmed_rave_encoded_concat_subjects.h5"
    csv_path = "vctk_trimmed_rave_encoded_concat_subjects.csv"

    seq_len = 256
    label_len = 128
    pred_len = 256

    enc_in = 8
    dec_in = 8
    c_out = 8
    d_model = 256
    n_heads = 8
    e_layers = 2
    d_layers = 1
    d_ff = 2048
    moving_avg = 6
    dropout = 0.2
    activation = 'gelu'
    output_attention = False

    # embed = 'timeF'
    embed = 'step_feature'
    # freq = 'h'
    # factor = 1
    # wavelet = 0

    batch_size = 1
    num_workers = 0

    scale = 1
    scaler_load_path = "checkpoints/scaler_concat.pkl"
    quantize = 1
    quantizer_type = "msprior"
    quantizer_num_clusters = 64
    quantizer_load_path = None

    learning_rate = 0.0001

args = Configs()

# %%
# create data loaders for test set
test_dataset, test_loader = data_provider_ravenc(args, "test", scaler=args.scaler_load_path, quantizer=args.quantizer_load_path)

# %%
# torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# load model
checkpoint_path = "checkpoints/model_hpc_long-v2.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=device)
fedformer = FEDformer.LitFEDformer(args).to(device)
fedformer.load_state_dict(checkpoint['state_dict'])
fedformer.eval()
print()
print(fedformer.model.encoder.attn_layers[0].attention.inner_correlation.index)
print()

# %%
# test save and load with torch.save and torch.load
checkpoint_path = "checkpoints/mode_init_test.pt"
fedformer = FEDformer.LitFEDformer(args).to(device)
fedformer.eval()
print()
print(fedformer.model.encoder.attn_layers[0].attention.inner_correlation.index)
print()
torch.save(fedformer, checkpoint_path)
# fedformer = FEDformer.LitFEDformer(args).to(device)
fedformer = torch.load(checkpoint_path, map_location=device)
# fedformer.load_state_dict(checkpoint['state_dict'])
fedformer.eval()
print()
print(fedformer.model.encoder.attn_layers[0].attention.inner_correlation.index)
print()

# %%
# simulate creating a checkpoint during training

# create a pytorch lightning model
fedformer = FEDformer.LitFEDformer(args)

# create a pytorch lightning trainer
model_ckpt = ModelCheckpoint(
    monitor='train_loss',
    dirpath='checkpoints',
    filename='mode_init_test',
    save_top_k=1,
    mode='min',
)

trainer = pl.Trainer(callbacks=[model_ckpt], limit_train_batches=4, limit_val_batches=1, max_epochs=4)

# train
trainer.fit(model=fedformer, train_dataloaders=test_loader, val_dataloaders=test_loader)

# %%
# recall checkpoint, check modes
checkpoint_path = "checkpoints/mode_init_test.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=device)
fedformer = FEDformer.LitFEDformer(args).to(device)
fedformer.load_state_dict(checkpoint['state_dict'])
fedformer.eval()
print()
print(fedformer.model.encoder.attn_layers[0].attention.inner_correlation.index)
print()
# %%
