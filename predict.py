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
parser.add_argument('--num_files', type=int, default=1, help='number of files to generate')
parser.add_argument('--num_steps', type=int, default=100, help='number of embedding steps to generate per file')
parser.add_argument('--output_folder', type=str, default='generated_audio', help='output folder to save generated audio')
parser.add_argument('--device', type=str, default='cuda', help='device to use for inference (cpu or cuda). Default: cuda')
cmd_args = parser.parse_args()

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

    root_path = 'data/RAVE_encoded_datasets'
    data_path = 'vctk_trimmed_rave_encoded.h5'
    csv_path = 'vctk_trimmed_rave_encoded.csv'

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
    embed = 'token_only'
    # embed = 'token_pos'
    # freq = 'h'
    # factor = 1
    # wavelet = 0

    batch_size = 512
    num_workers = 8

args = Configs()

# %%
# create data loaders for train and val
train_set, train_loader = data_provider_ravenc(args, "train")
val_dataset, val_loader = data_provider_ravenc(args, "val", scaler=train_set.scaler, train_set=train_set)
test_dataset, test_loader = data_provider_ravenc(args, "test", scaler=train_set.scaler, train_set=train_set)

# %%
# torch device
device = torch.device(cmd_args.device)

# %%
# load model
checkpoint_path = "checkpoints/model_hpc_silence_trimmed_high_prec.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=device)
fedformer = FEDformer.LitFEDformer(args).to(device)
fedformer.load_state_dict(checkpoint['state_dict'])
# disable randomness, dropout, etc...
fedformer.eval()

# %%
# load a pretrained RAVE model via torch.script
models_folder = "rave_pretrained_models"
chosen_model = "VCTK"
rave_model = torch.jit.load(os.path.join(models_folder, chosen_model + ".ts"), map_location=device)
rave_model.eval()


# %%
# autoregressive setup
output_folder = cmd_args.output_folder
os.makedirs(output_folder, exist_ok=True)

progress_bar = tqdm.tqdm(range(cmd_args.num_files))
progress_bar.set_description(f"Generating {cmd_args.num_files} files")

for generation_id in progress_bar:
    # choose random id
    id = np.random.randint(len(test_dataset))
    dataset_id, _ = test_dataset.chunk_dataset[id]
    generated_length = cmd_args.num_steps
    num_generations = generated_length // args.pred_len
    # get the batch
    x, y = test_dataset[id]
    x = x.unsqueeze(0).to(device) # (1, seq_len, 8)
    y = y.unsqueeze(0).to(device) # (1, label_len + pred_len, 8)

    zeros4preds = torch.zeros_like(y[:, -args.pred_len:, :]).float().to(device) # (1, pred_len, 8)
    y_labels = y[:, :args.label_len, :].to(device) # (1, label_len, 8)

    generated = []
    with torch.no_grad():
        for i in range(num_generations):
            # create a new decoder input
            dec_inp = torch.cat([y_labels, zeros4preds], dim=1).float() # (1, label_len + pred_len, 8)
            # get model predictions
            model_preds = fedformer.model(x, None, dec_inp, None) # (1, pred_len, 8)
            # append the predictions to the generated list
            generated.append(model_preds)
            # print(model_preds.shape)
            # shift x by pred_len and append the predictions
            x = torch.cat([x[:, args.pred_len:, :], model_preds], dim=1) # (1, seq_len, 8)
            y_labels = x[:, -args.label_len:, :] # (1, label_len, 8)

    # concatenate all the generated predictions
    generated = torch.cat(generated, dim=1) # (1, generated_length, 8)

    # %%
    # use the train set scaler to inverse transform the generated predictions
    generated = generated.squeeze(0).cpu().numpy()
    generated = train_set.scaler.inverse_transform(generated)
    generated = torch.from_numpy(generated).unsqueeze(0).to(device)

    # %%
    # decode the generated predictions
    with torch.no_grad():
        # reshape generated to (1, 8, generated_length) and decode
        decoded = rave_model.decode(generated.transpose(1, 2))

    # %%
    # save decoded to wav file
    # reshape to channels-last, squeeze batch dimension
    buffer = decoded.transpose(1, 2).squeeze(0).cpu().numpy()
    output_file_name = f"{chosen_model}_FEDformer_generated_{generation_id}.wav"
    # VCTK is RAVE V1 - Default, that uses 44100 Hz sampling rate
    sf.write(os.path.join(output_folder, output_file_name), buffer, 44100)

# %%
