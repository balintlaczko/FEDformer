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

# %%
# gather all flac files in vctk folder

vctk_root_folder = "/Volumes/T7RITMO/VCTK/DS_10283_3443/VCTK-Corpus-0.92/wav48_silence_trimmed"

# crawl through the folder and find all the flac files
flac_files = []
for root, dirs, files in os.walk(vctk_root_folder):
    for file in files:
        if file.endswith(".flac") and not file.startswith("."):
            flac_files.append(os.path.join(root, file))

print(f"Found {len(flac_files)} flac files")


# %%
# load the vctk rave model
models_folder = "/Users/balintl/Documents/Max 8/Library/ts_models/"
model_names = ["percussion", "darbouka_onnx",
               "nasa", "VCTK", "vintage", "wheel"]
vctk_rave = torch.jit.load(os.path.join(models_folder, "VCTK.ts"))

# %%
# Define the path to the h5 database
db_path = "/Volumes/T7RITMO/RAVE_encoded_datasets/vctk_rave_encoded.h5"


# %%
# write all encoded tensors to the h5 file
with h5py.File(db_path, 'w') as f:
    for i, file in enumerate(tqdm.tqdm(flac_files)):
        # load the file
        buf, sr = sf.read(file)
        # convert to tensor
        tensor = torch.from_numpy(buf).float().unsqueeze(0).unsqueeze(0)
        # encode
        with torch.no_grad():
            z = vctk_rave.encode(tensor)
        # save to h5 file
        # looks a bit clunky to create a "dataset" to every single z,
        # but this way you don't have to zero-pad everything to the
        # longest embedding sequence length
        f.create_dataset(str(i), data=z.numpy())


# %%
# create a pandas dataframe with the file names and the h5 dataset indices,
# also performing a train/val/test split
df = pd.DataFrame(flac_files, columns=["file_path"])
df["dataset_index"] = df.index
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
# merge the 3 datasets into one
df_train["dataset"] = "train"
df_val["dataset"] = "val"
df_test["dataset"] = "test"
df = pd.concat([df_train, df_val, df_test], axis=0)
df = df.reset_index(drop=True)
# save the dataframe to a csv file
path_to_csv = "/Volumes/T7RITMO/RAVE_encoded_datasets/vctk_rave_encoded.csv"
df.to_csv(path_to_csv, index=False)

# %%
