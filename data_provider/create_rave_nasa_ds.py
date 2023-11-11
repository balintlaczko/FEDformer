# %%
# imports
import h5py
import soundfile as sf
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import tqdm
from sklearn.model_selection import train_test_split
import torchaudio

# %%
# read the apollo audio file
apollo_wav_path = "../data/Apollo11_44100_mono_norm.wav"
buf, sr = torchaudio.load(apollo_wav_path, normalize=True)

# %%
# load the nasa rave model
models_folder = "../rave_pretrained_models"
model_names = ["percussion", "darbouka_onnx",
               "nasa", "VCTK", "vintage", "wheel"]
nasa_rave = torch.jit.load(os.path.join(models_folder, "nasa.ts"))
nasa_rave.eval()

# %%
# Define the path to the h5 database for the whole apollo audio as one entry
db_path = "../data/RAVE_encoded_datasets/nasa_rave_encoded.h5"

# %%
# write all encoded tensors to the h5 file
with h5py.File(db_path, 'w') as f:
    # encode
    with torch.no_grad():
        z = nasa_rave.encode(buf.unsqueeze(0))
        print(z.shape)
    # save to h5 file
    f.create_dataset(str(0), data=z.numpy())

# %%
# load the encoded tensor from the h5 file
with h5py.File(db_path, 'r') as f:
    z = f["0"][()]
z = torch.from_numpy(z)

# %%
# take the z and divide it into 125 equal chunks
z_chunks = torch.chunk(z, 1250, dim=-1)
print(len(z_chunks))

# %%
# Define the path to the h5 database for the apollo audio split into 125 entries
db_path_chunked = "../data/RAVE_encoded_datasets/nasa_rave_encoded_split_2.h5"

# %%
# write all chunks to the h5 file
with h5py.File(db_path_chunked, 'w') as f:
    for i, z in enumerate(z_chunks):
        # save to h5 file
        f.create_dataset(str(i), data=z.numpy())

# %%
# create a pandas dataframe with the file names and the h5 dataset indices,
# also performing a train/val/test split
df = pd.DataFrame(list(range(len(z_chunks))), columns=["dataset_index"])
# df["dataset_index"] = df.index
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
# merge the 3 datasets into one
df_train["dataset"] = "train"
df_val["dataset"] = "val"
df_test["dataset"] = "test"
df = pd.concat([df_train, df_val, df_test], axis=0)
df = df.reset_index(drop=True)
# save the dataframe to a csv file
path_to_csv = "../data/RAVE_encoded_datasets/nasa_rave_encoded_split_2.csv"
df.to_csv(path_to_csv, index=False)

# %%
