# %%
# imports
import os
import torch
import numpy as np
import lightning.pytorch as pl
from data_provider.data_factory import data_provider_ravenc
from models import FEDformer
import soundfile as sf

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
    data_path = 'vctk_rave_encoded.h5'
    csv_path = 'vctk_rave_encoded.csv'

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
# convert configs to dict
def props(cls):   
  return [i for i in cls.__dict__.keys() if i[:1] != '_']

properties = props(Configs)
print(properties)

dict_args = {}
for prop in properties:
    dict_args[prop] = getattr(args, prop)
print(dict_args)

# %%
# load checkpoint
checkpoint_path = "checkpoints/model_hpc-v1.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
print(checkpoint.keys())

# %%
# load model
checkpoint_path = "checkpoints/model_hpc-v1.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
fedformer = FEDformer.LitFEDformer(args).cuda()
fedformer.load_state_dict(checkpoint['state_dict'])
# disable randomness, dropout, etc...
fedformer.eval()

# %%
# get a random batch from the test set
# choose random id
id = np.random.randint(len(test_dataset))

# get the batch
x, y = test_dataset[id]
x = x.unsqueeze(0)
y = y.unsqueeze(0)

print(id)
print(x.shape)
print(y.shape)

# %%
# predict with the model
dec_inp = torch.zeros_like(y[:, -args.pred_len:, :]).float()
dec_inp = torch.cat([y[:, :args.label_len, :], dec_inp], dim=1).float()
print(x.shape)
print(dec_inp.shape)
with torch.no_grad():
    y_hat = fedformer.model(x, None, dec_inp, None)
y_labels_for_preds = y[:, -args.pred_len:, :]
loss = torch.nn.functional.mse_loss(y_hat, y_labels_for_preds)
print(y_hat.shape)
print(loss)

# %%
# load a pretrained RAVE model via torch.script
models_folder = "rave_pretrained_models"
# model_names = ["percussion", "darbouka_onnx",
#                "nasa", "VCTK", "vintage", "wheel"]
chosen_model = "VCTK"
# models = [torch.jit.load(os.path.join(models_folder, name + ".ts"))
#           for name in model_names]
rave_model = torch.jit.load(os.path.join(models_folder, chosen_model + ".ts"), map_location=torch.device('cuda'))
# eval mode
# for model in models:
#     model.eval()
rave_model.eval()


# %%
# autoregressive setup
# choose random id
id = np.random.randint(len(test_dataset))
print(f'random seed id: {id}')
dataset_id, _ = test_dataset.chunk_dataset[id]
print(f'dataset id: {dataset_id}')
generated_length = 200
num_generations = generated_length // args.pred_len
# get the batch
x, y = test_dataset[id]
x = x.unsqueeze(0).cuda() # (1, seq_len, 8)
y = y.unsqueeze(0).cuda() # (1, label_len + pred_len, 8)

zeros4preds = torch.zeros_like(y[:, -args.pred_len:, :]).float().cuda() # (1, pred_len, 8)
y_labels = y[:, :args.label_len, :].cuda() # (1, label_len, 8)

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
print(generated.shape)

# %%
# use the train set scaler to inverse transform the generated predictions
generated = generated.squeeze(0).cpu().numpy()
generated = train_set.scaler.inverse_transform(generated)
generated = torch.from_numpy(generated).unsqueeze(0).cuda()


# %%
# decode the generated predictions
with torch.no_grad():
    # reshape generated to (1, 8, generated_length) and decode
    decoded = rave_model.decode(generated.transpose(1, 2))
print(decoded.shape)

# %%
# save decoded to wav file
# reshape to channels-last, squeeze batch dimension
buffer = decoded.transpose(1, 2).squeeze(0).cpu().numpy()
print(buffer.shape)
destination_folder = "generated_audio"
os.makedirs(destination_folder, exist_ok=True)
output_file_name = f"{chosen_model}_FEDformer_generated_.wav"
# VCTK is RAVE V1 - Default, that uses 44100 Hz sampling rate
sf.write(os.path.join(destination_folder, "generated.wav"), buffer, 44100)
# %%
