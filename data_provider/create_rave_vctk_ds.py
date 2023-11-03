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
# gather all flac files in vctk folder

vctk_root_folder = "/Volumes/T7RITMO/VCTK/DS_10283_3443/VCTK-Corpus-0.92/wav48_silence_trimmed"

# crawl through the folder and find all the flac files
flac_files = []
for root, dirs, files in os.walk(vctk_root_folder):
    for file in files:
        if file.endswith("mic1.flac") and not file.startswith("."):
            flac_files.append(os.path.join(root, file))

print(f"Found {len(flac_files)} flac files")


# %%
# load the vctk rave model
models_folder = "/Users/balintl/Documents/Max 8/Library/ts_models/"
model_names = ["percussion", "darbouka_onnx",
               "nasa", "VCTK", "vintage", "wheel"]
vctk_rave = torch.jit.load(os.path.join(models_folder, "VCTK.ts"))
vctk_rave.eval()

# %%
# Define the path to the h5 database
db_path = "/Volumes/T7RITMO/RAVE_encoded_datasets/vctk_trimmed_rave_encoded.h5"

# %%
# write all encoded tensors to the h5 file
with h5py.File(db_path, 'w') as f:
    for i, file in enumerate(tqdm.tqdm(flac_files)):
        # load the file
        buf, sr = torchaudio.load(file, normalize=True)
        # normalize to 0.9
        buf = buf / torch.max(torch.abs(buf))
        buf *= 0.9
        # create transforms
        # used to trim the start of the file
        vad_transform_front = torchaudio.transforms.Vad(
            sample_rate=sr,
            allowed_gap=0.5,
            trigger_level=7,
            trigger_time=0.6,
            search_time=0.5,
            boot_time=0.2,
            hp_filter_freq=85,
        )
        # used to trim the end of the file
        vad_transform_back = torchaudio.transforms.Vad(
            sample_rate=sr,
            allowed_gap=0.3,
            trigger_level=8,
            trigger_time=0.1,
            search_time=0.3,
            boot_time=0.2,
            hp_filter_freq=85,
        )
        # add 25 ms fade in and fade out
        fade = torchaudio.transforms.Fade(
            fade_in_len=int(sr/1000*25),
            fade_out_len=int(sr/1000*25),
            fade_shape="half_sine",
        )
        # front trim
        buf = vad_transform_front(buf)
        # reverse
        buf = torch.flip(buf, [1])
        # back trim
        buf = vad_transform_back(buf)
        # reverse again
        buf = torch.flip(buf, [1])
        # fade in and out
        buf = fade(buf)
        # print(buf.shape)
        # encode
        with torch.no_grad():
            z = vctk_rave.encode(buf.unsqueeze(0))
            # print(z.shape)
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
path_to_csv = "/Volumes/T7RITMO/RAVE_encoded_datasets/vctk_trimmed_rave_encoded.csv"
df.to_csv(path_to_csv, index=False)

# %%
