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
parser.add_argument('--scale', type=int, default=1, help='scale the data')
parser.add_argument('--scaler_load_path', type=str, default='checkpoints/scaler.pt', help='path to where to load the fit scaler from')
parser.add_argument('--quantize', type=int, default=1, help='quantize the data')
parser.add_argument('--quantizer_type', type=str, default='kmeans', help='type of quantizer to use (kmeans or msprior)')
parser.add_argument('--quantizer_num_clusters', type=int, default=4096, help='number of clusters for quantization')
parser.add_argument('--quantizer_load_path', type=str, default='checkpoints/quantizer.pt', help='path to where to load the fit k-means quantizer from')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--embed', type=str, default='token_only', help='embedding type (token_only, step_offset, or step_feature)')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/model_hpc_only_quantize-v13.ckpt', help='path to model checkpoint')
parser.add_argument('--rave_model_path', type=str, default='rave_pretrained_models/VCTK.ts', help='path to RAVE model')
parser.add_argument('--rave_model_sr', type=int, default=44100, help='sampling rate of the RAVE model')
parser.add_argument('--num_files', type=int, default=1, help='number of files to generate')
parser.add_argument('--num_steps', type=int, default=256, help='number of embedding steps to generate per file')
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

    root_path = cmd_args.root_path
    data_path = cmd_args.data_path
    csv_path = cmd_args.csv_path

    seq_len = 256
    label_len = 128
    pred_len = 256

    enc_in = 8
    dec_in = 8
    c_out = 8
    d_model = cmd_args.d_model
    n_heads = 8
    e_layers = 2
    d_layers = 1
    d_ff = 2048
    moving_avg = 48
    dropout = 0.2
    activation = 'gelu'
    output_attention = False

    # embed = 'timeF'
    # embed = 'token_only'
    embed = cmd_args.embed
    # freq = 'h'
    # factor = 1
    # wavelet = 0

    batch_size = 512
    num_workers = 8

    scale = cmd_args.scale
    scaler_load_path = cmd_args.scaler_load_path
    quantize = cmd_args.quantize
    quantizer_type = cmd_args.quantizer_type
    quantizer_num_clusters = cmd_args.quantizer_num_clusters
    quantizer_load_path = cmd_args.quantizer_load_path

args = Configs()

# %%
# create data loaders for test set
test_dataset, test_loader = data_provider_ravenc(args, "test", scaler=args.scaler_load_path, quantizer=args.quantizer_load_path)

# %%
# torch device
device = torch.device(cmd_args.device)

# %%
# load model
checkpoint_path = cmd_args.checkpoint_path
checkpoint = torch.load(checkpoint_path, map_location=device)
fedformer = FEDformer.LitFEDformer(args).to(device)
fedformer.load_state_dict(checkpoint['state_dict'])
# disable randomness, dropout, etc...
fedformer.eval()
print()
print(fedformer.model.encoder.attn_layers[0].attention.inner_correlation.index)
print()

# %%
# load a pretrained RAVE model via torch.script
chosen_model = os.path.basename(cmd_args.rave_model_path).split(".")[0]
rave_model = torch.jit.load(cmd_args.rave_model_path, map_location=device)
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
    generated = test_dataset.scaler.inverse_transform(generated)
    generated = torch.from_numpy(generated).unsqueeze(0).to(device)

    # %%
    # decode the generated predictions
    # TODO: reset the RAVE model state between generations
    # the ugly way
    rave_model = torch.jit.load(cmd_args.rave_model_path, map_location=device)
    rave_model.eval()
    with torch.no_grad():
        # reshape generated to (1, 8, generated_length) and decode
        decoded = rave_model.decode(generated.transpose(1, 2))
        # sanity check with inputs
        # x = train_set.scaler.inverse_transform(x.squeeze(0).cpu().numpy())
        # x = torch.from_numpy(x).unsqueeze(0).to(device)
        # decoded = rave_model.decode(x.transpose(1, 2))

    # %%
    # save decoded to wav file
    # reshape to channels-last, squeeze batch dimension
    buffer = decoded.transpose(1, 2).squeeze(0).cpu().numpy()
    output_file_name = f"{chosen_model}_FEDformer_generated_{generation_id}.wav"
    sf.write(os.path.join(output_folder, output_file_name), buffer, cmd_args.rave_model_sr)

# %%
