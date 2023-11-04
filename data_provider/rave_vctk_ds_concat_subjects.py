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
# load the dataset
db_path = "../data/RAVE_encoded_datasets/vctk_trimmed_rave_encoded.h5"
csv_path = "../data/RAVE_encoded_datasets/vctk_trimmed_rave_encoded.csv"

df = pd.read_csv(csv_path)

whole_file_embeddings = []
with h5py.File(db_path, 'r') as f:
    # get progress bar form indices
    pbar = tqdm.tqdm(f.keys())
    pbar.set_description("loading all embeddings in memory")
    for i in pbar:
        whole_file_embeddings.append(
            torch.from_numpy(f[str(int(i))][()]))
        
df.head()

# %%
# add new columns for subject_id and sentence_id
df["subject_id"] = df["file_path"].apply(lambda x: os.path.basename(x).split("_")[0])
df["sentence_id"] = df["file_path"].apply(lambda x: os.path.basename(x).split("_")[1])
df.head()

# %%
# for each subject, filter df, sort by sentence_id and concatenate the embeddings
subjects_embeddings = []
subjects_ids = []
for subject_id in tqdm.tqdm(df["subject_id"].unique()):
    subject_df = df[df["subject_id"] == subject_id]
    subject_df = subject_df.sort_values(by="sentence_id")
    subject_embeddings = []
    for sentence_id in subject_df["sentence_id"].unique():
        sentence_df = subject_df[subject_df["sentence_id"] == sentence_id]
        dataset_index = sentence_df["dataset_index"].values[0]
        sentence_embedding = whole_file_embeddings[dataset_index]
        subject_embeddings.append(sentence_embedding)
    subject_embeddings = torch.cat(subject_embeddings, dim=-1)
    subjects_embeddings.append(subject_embeddings)
    subjects_ids.append(subject_id)

# %%
for id, subject_id in enumerate(subjects_ids):
    print(subject_id)
    print(subjects_embeddings[id].shape)
# %%
len(subjects_embeddings)

# %%
# create a new h5 dataset with the concatenated embeddings
new_db_path = "../data/RAVE_encoded_datasets/vctk_trimmed_rave_encoded_concat_subjects.h5"
with h5py.File(new_db_path, 'w') as f:
    for id, subject_id in enumerate(subjects_ids):
        f.create_dataset(str(id), data=subjects_embeddings[id].numpy())

# %%
# create another dataframe with the subject ids and the dataset indices
subjects_concat_df = pd.DataFrame(subjects_ids, columns=["subject_id"])
subjects_concat_df["dataset_index"] = subjects_concat_df.index
subjects_concat_df_train, subjects_concat_df_test = train_test_split(subjects_concat_df, test_size=0.2, random_state=42)
subjects_concat_df_train, subjects_concat_df_val = train_test_split(subjects_concat_df_train, test_size=0.2, random_state=42)
# merge the 3 datasets into one
subjects_concat_df_train["dataset"] = "train"
subjects_concat_df_val["dataset"] = "val"
subjects_concat_df_test["dataset"] = "test"
subjects_concat_df = pd.concat([subjects_concat_df_train, subjects_concat_df_val, subjects_concat_df_test], axis=0)
subjects_concat_df = subjects_concat_df.reset_index(drop=True)
# save the dataframe to a csv file
path_to_csv = "../data/RAVE_encoded_datasets/vctk_trimmed_rave_encoded_concat_subjects.csv"
subjects_concat_df.to_csv(path_to_csv, index=False)
# %%
