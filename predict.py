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
parser.add_argument('--version', type=str, default='Fourier', help='version of the model (Fourier or Wavelets)')
parser.add_argument('--modes', type=int, default=64, help='number of modes to use for the Fourier layer')
parser.add_argument('--pred_len', type=int, default=256, help='number of steps to predict')
parser.add_argument('--scale', type=int, default=1, help='scale the data')
parser.add_argument('--scaler_type', type=str, default='minmax', help='type of scaler to use (minmax or global)')
parser.add_argument('--scaler_load_path', type=str, default='checkpoints/scaler.pt', help='path to where to load the fit scaler from')
parser.add_argument('--quantize', type=int, default=1, help='quantize the data')
parser.add_argument('--quantizer_type', type=str, default='msprior', help='type of quantizer to use (kmeans or msprior)')
parser.add_argument('--quantizer_num_clusters', type=int, default=64, help='number of clusters for quantization')
parser.add_argument('--quantizer_load_path', type=str, default='checkpoints/quantizer.pt', help='path to where to load the fit k-means quantizer from')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--embed', type=str, default='step_feature', help='embedding type (token_only, step_offset, or step_feature)')
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
    version = cmd_args.version
    mode_select = 'random'
    modes = cmd_args.modes
    L = 3
    base = 'legendre'
    cross_activation = 'tanh'

    root_path = cmd_args.root_path
    data_path = cmd_args.data_path
    csv_path = cmd_args.csv_path

    seq_len = 256
    label_len = 128
    pred_len = cmd_args.pred_len

    enc_in = 8
    dec_in = 8
    c_out = 8
    d_model = cmd_args.d_model
    n_heads = 8
    e_layers = 2
    d_layers = 1
    d_ff = 2048
    moving_avg = 2
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
    scaler_type = cmd_args.scaler_type
    scaler_load_path = cmd_args.scaler_load_path
    quantize = cmd_args.quantize
    quantizer_type = cmd_args.quantizer_type
    quantizer_num_clusters = cmd_args.quantizer_num_clusters
    quantizer_load_path = cmd_args.quantizer_load_path

    filter_vctk = True

args = Configs()

# %%
# basic config
class Configs_debug(object):
    # ab = 0
    version = "Fourier"
    mode_select = 'random'
    modes = 64
    L = 3
    base = 'legendre'
    cross_activation = 'tanh'

    root_path = "data/RAVE_encoded_datasets"
    data_path = "vctk_trimmed_rave_encoded_concat_subjects_chunked.h5"
    csv_path = "vctk_trimmed_rave_encoded_concat_subjects_chunked.csv"

    seq_len = 256
    label_len = 128
    pred_len = 4

    enc_in = 8
    dec_in = 8
    c_out = 8
    d_model = 512
    n_heads = 8
    e_layers = 2
    d_layers = 1
    d_ff = 2048
    moving_avg = 2
    dropout = 0.2
    activation = 'gelu'
    output_attention = False

    # embed = 'timeF'
    # embed = 'token_only'
    embed = "token_pos"
    # freq = 'h'
    # factor = 1
    # wavelet = 0

    batch_size = 512
    num_workers = 8

    scale = 1
    # scaler_type = "global"
    scaler_type = "minmax"
    scaler_load_path = None
    quantize = 0
    quantizer_type = "msprior"
    quantizer_num_clusters = 64
    quantizer_load_path = ""

# args = Configs_debug()

# %%
# create data loaders for test set
# train_dataset, train_loader = data_provider_ravenc(args, "train")
test_dataset, test_loader = data_provider_ravenc(args, "test", scaler=args.scaler_load_path,)
# %%
# in "global" scaling mode we need the global min and max from the train set
if args.scaler_type == 'global':
    train_dataset, train_loader = data_provider_ravenc(args, "train", scaler=args.scaler_load_path)
    test_dataset.global_min = train_dataset.global_min
    test_dataset.global_max = train_dataset.global_max
    print(f"Global min: {test_dataset.global_min}")
    print(f"Global max: {test_dataset.global_max}")


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
# print(fedformer.model.encoder.attn_layers[0].attention.inner_correlation.index)
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

ids = np.arange(len(test_dataset))
np.random.shuffle(ids)
ids = ids[:cmd_args.num_files]

for generation_id in progress_bar:
    # choose random id
    # id = np.random.randint(len(test_dataset))
    id = ids[generation_id]
    dataset_id, _ = test_dataset.chunk_dataset[id]
    generated_length = cmd_args.num_steps
    num_generations = generated_length // args.pred_len
    # get the batch
    x_original, y_original = test_dataset[id]
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
    if args.scale == 1:
        generated = test_dataset.inverse_transform(generated.cpu())
        generated = generated.to(device)

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
        # x = test_dataset.inverse_transform(x_original.cpu())
        # x = x.to(device)
        # decoded = rave_model.decode(x.transpose(1, 2))

    # %%
    # save decoded to wav file
    # reshape to channels-last, squeeze batch dimension
    buffer = decoded.transpose(1, 2).squeeze(0).cpu().numpy()
    output_file_name = f"{chosen_model}_FEDformer_generated_{generation_id}.wav"
    sf.write(os.path.join(output_folder, output_file_name), buffer, cmd_args.rave_model_sr)

# %%
