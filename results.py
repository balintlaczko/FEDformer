# %%
# imports
import os
import torch
import numpy as np
from data_provider.data_factory import data_provider_ravenc
from models import FEDformer
import soundfile as sf
import tqdm
from frechet_audio_distance import FrechetAudioDistance

# %%
# set up configs for model A and model B

# config for model A


class ConfigA(object):
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
    filter_vctk = True

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

    embed = "token_pos"

    batch_size = 1
    num_workers = 0

    scale = 1
    scaler_type = "global"
    # scaler_load_path = "checkpoints/mm_scaler_vctk.pkl"
    scaler_load_path = None
    quantize = 0
    quantizer_type = "msprior"
    quantizer_num_clusters = 64
    quantizer_load_path = ""


class ConfigB(ConfigA):
    # pred_len = 4
    scale = 0
    # scaler_type = "global"
    # filter_vctk = True
    modes = 128
    d_model = 1024


args_a = ConfigA()
args_b = ConfigB()

# %%
# torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# load model A
model_a_ckpt_path = "checkpoints/model_hpc_vctk_p239/model_hpc_vctk_p239_last_epoch=9999.ckpt"
checkpoint_a = torch.load(model_a_ckpt_path, map_location=device)
model_a = FEDformer.LitFEDformer(args_a).to(device)
model_a.load_state_dict(checkpoint_a['state_dict'])
model_a.eval()

# %%
# load model B
model_b_ckpt_path = "checkpoints/model_hpc_vctk_p239_beefy/model_hpc_vctk_p239_beefy_last_epoch=452.ckpt"
checkpoint_b = torch.load(model_b_ckpt_path, map_location=device)
model_b = FEDformer.LitFEDformer(args_b).to(device)
model_b.load_state_dict(checkpoint_b['state_dict'])
model_b.eval()

# %%
# load a pretrained RAVE model via torch.script
rave_model_path = "rave_pretrained_models/VCTK.ts"
chosen_model = os.path.basename(rave_model_path).split(".")[0]
rave_model = torch.jit.load(rave_model_path, map_location=device)
rave_model.eval()

# %%
# create data loaders for test set
train_dataset_a, train_loader_a = data_provider_ravenc(args_a, "train")
test_dataset_a, test_loader_a = data_provider_ravenc(args_a, "test")
test_dataset_a.global_min = train_dataset_a.global_min
test_dataset_a.global_max = train_dataset_a.global_max

# train_dataset_b, train_loader_b = data_provider_ravenc(args_b, "train")
test_dataset_b, test_loader_b = data_provider_ravenc(args_b, "test")

# %%
# decode test set chunks for comparison

test_ds_decoded_dir = "data/RAVE_decoded_datasets/test_ds_decoded"
os.makedirs(test_ds_decoded_dir, exist_ok=True)

files_to_generate = 500
seq_len = 64
# [8, 1] (same for both models)
test_set_chunk_ids = test_dataset_a.df.dataset_index.values

for file_id in tqdm.tqdm(range(files_to_generate)):
    # alternate between using the two test chunks (8 or 1)
    if file_id % 2 == 0:
        ind = test_set_chunk_ids[0]
    else:
        ind = test_set_chunk_ids[1]

    # get data
    x = test_dataset_a.whole_file_embeddings[ind]  # (1, 8, 1905)

    # get a random sequence of length seq_len
    random_start = np.random.randint(0, x.shape[-1] - seq_len)
    x = x[..., random_start:random_start + seq_len]  # (1, seq_len, 8)

    # reload rave model
    rave_model = torch.jit.load(rave_model_path, map_location=device)
    rave_model.eval()

    # decode
    with torch.no_grad():
        decoded = rave_model.decode(x)

    # save
    buffer = decoded.transpose(1, 2).squeeze(0).cpu().numpy()
    output_file_name = f"test_ds_decoded_{file_id}.wav"
    sf.write(os.path.join(test_ds_decoded_dir,
             output_file_name), buffer, 44100)


# %%
# load and decode evo initials

evo_initials_decoded_dir = "data/RAVE_decoded_datasets/evo_initials_decoded"
os.makedirs(evo_initials_decoded_dir, exist_ok=True)

evo_initials_encoded = np.load("genes/genes.npy")  # (500, 8, 64)

for ind in tqdm.tqdm(range(evo_initials_encoded.shape[0])):
    # get data
    x = torch.from_numpy(evo_initials_encoded[ind]).unsqueeze(
        0).float().to(device)

    # reload rave model
    rave_model = torch.jit.load(rave_model_path, map_location=device)
    rave_model.eval()

    # decode
    with torch.no_grad():
        decoded = rave_model.decode(x.transpose(0, 1))

    # save
    buffer = decoded.transpose(1, 2).squeeze(0).cpu().numpy()
    output_file_name = f"evo_decoded_{ind}.wav"
    sf.write(os.path.join(evo_initials_decoded_dir,
             output_file_name), buffer, 44100)

# %%
# generate predictions; model A; condition: TEST

model_a_pred_cond_test_dir = "data/RAVE_decoded_datasets/model_a_pred_cond_test"
os.makedirs(model_a_pred_cond_test_dir, exist_ok=True)

seq_len = args_a.seq_len  # 256
label_len = args_a.label_len  # 128
pred_len = args_a.pred_len  # 4
files_to_generate = 500
generated_length = 64
num_generations = generated_length // pred_len  # 1

# [8, 1] (same for both models)
test_set_chunk_ids = test_dataset_a.df.dataset_index.values

for file_id in tqdm.tqdm(range(files_to_generate)):
    # alternate between using the two test chunks (8 or 1)
    if file_id % 2 == 0:
        ind = test_set_chunk_ids[0]
    else:
        ind = test_set_chunk_ids[1]

    # get data
    x = test_dataset_a.whole_file_embeddings[ind].transpose(
        1, 2)  # (1, 1905, 8)

    # apply global scaling
    x = test_dataset_a.scale_to_global_minmax(x)  # (1, 1905, 8)

    # get a random sequence of length seq_len
    random_start = np.random.randint(0, x.shape[1] - seq_len)
    x = x[:, random_start:random_start + seq_len, :]  # (1, seq_len, 8)

    # create buffers for model forward pass
    zeros4preds = torch.zeros_like(
        x[:, -pred_len:, :]).float().to(device)  # (1, pred_len, 8)
    y_labels = x[:, -label_len:, :].to(device)  # (1, label_len, 8)

    # autoregressive loop
    generated = []
    with torch.no_grad():
        for i in range(num_generations):
            # create a new decoder input
            # (1, label_len + pred_len, 8)
            dec_inp = torch.cat([y_labels, zeros4preds], dim=1).float()
            # get model predictions
            model_preds = model_a.model(
                x, None, dec_inp, None)  # (1, pred_len, 8)
            # append the predictions to the generated list
            generated.append(model_preds)
            # shift x by pred_len and append the predictions
            x = torch.cat([x[:, pred_len:, :], model_preds],
                          dim=1)  # (1, seq_len, 8)
            y_labels = x[:, -label_len:, :]  # (1, label_len, 8)

    # concatenate all the generated predictions
    generated = torch.cat(generated, dim=1)  # (1, generated_length, 8)

    # use the global min/max to inverse transform the generated predictions
    generated = test_dataset_a.unscale_from_global_minmax(generated)

    # reload rave model
    rave_model = torch.jit.load(rave_model_path, map_location=device)
    rave_model.eval()

    # decode
    with torch.no_grad():
        decoded = rave_model.decode(generated.transpose(1, 2))

    # save
    buffer = decoded.transpose(1, 2).squeeze(0).cpu().numpy()
    output_file_name = f"model_a_pred_cond_test_{file_id}.wav"
    sf.write(os.path.join(model_a_pred_cond_test_dir,
             output_file_name), buffer, 44100)


# %%
# concatenate test set, measure mean and std vectors
# [8, 1] (same for both models)
test_set_chunk_ids = test_dataset_a.df.dataset_index.values
test_embeddings = [test_dataset_a.whole_file_embeddings[ind]
                   for ind in test_set_chunk_ids]
test_embeddings = torch.cat(test_embeddings, dim=-1)  # (1, 8, 4133)
# channels last
test_embeddings = test_embeddings.transpose(1, 2)  # BTC: (1, 4133, 8)
test_dataset_a_mean = test_embeddings.mean(dim=1).squeeze(0)  # (8,)
test_dataset_a_std = test_embeddings.std(dim=1).squeeze(0)  # (8,)
print("test_dataset_a_mean:", test_dataset_a_mean)
print("test_dataset_a_std:", test_dataset_a_std)
# create a distribution from the mean and std vectors
test_dataset_a_dist = torch.distributions.normal.Normal(
    test_dataset_a_mean, test_dataset_a_std)
# sample from the distribution
sample = test_dataset_a_dist.sample((1, 256))
print("sample shape: ", sample.shape)

# %%
# generate predictions; model A; condition: NORMAL


model_a_pred_cond_norm_dir = "data/RAVE_decoded_datasets/model_a_pred_cond_norm"
os.makedirs(model_a_pred_cond_norm_dir, exist_ok=True)

seq_len = args_a.seq_len  # 256
label_len = args_a.label_len  # 128
pred_len = args_a.pred_len  # 4
files_to_generate = 500
generated_length = 64
num_generations = generated_length // pred_len  # 1

# [8, 1] (same for both models)
test_set_chunk_ids = test_dataset_a.df.dataset_index.values


for file_id in tqdm.tqdm(range(files_to_generate)):
    # sample from the normal distribution
    x = test_dataset_a_dist.sample((1, seq_len)).to(device)

    # apply global scaling
    x = test_dataset_a.scale_to_global_minmax(x)  # (1, seq_len, 8)

    # create buffers for model forward pass
    zeros4preds = torch.zeros_like(
        x[:, -pred_len:, :]).float().to(device)  # (1, pred_len, 8)
    y_labels = x[:, -label_len:, :].to(device)  # (1, label_len, 8)

    # autoregressive loop
    generated = []
    with torch.no_grad():
        for i in range(num_generations):
            # create a new decoder input
            # (1, label_len + pred_len, 8)
            dec_inp = torch.cat([y_labels, zeros4preds], dim=1).float()
            # get model predictions
            model_preds = model_a.model(
                x, None, dec_inp, None)  # (1, pred_len, 8)
            # append the predictions to the generated list
            generated.append(model_preds)
            # shift x by pred_len and append the predictions
            x = torch.cat([x[:, pred_len:, :], model_preds],
                          dim=1)  # (1, seq_len, 8)
            y_labels = x[:, -label_len:, :]  # (1, label_len, 8)

    # concatenate all the generated predictions
    generated = torch.cat(generated, dim=1)  # (1, generated_length, 8)

    # use the global min/max to inverse transform the generated predictions
    generated = test_dataset_a.unscale_from_global_minmax(generated)

    # reload rave model
    rave_model = torch.jit.load(rave_model_path, map_location=device)
    rave_model.eval()

    # decode
    with torch.no_grad():
        decoded = rave_model.decode(generated.transpose(1, 2))

    # save
    buffer = decoded.transpose(1, 2).squeeze(0).cpu().numpy()
    output_file_name = f"model_a_pred_cond_norm_{file_id}.wav"
    sf.write(os.path.join(model_a_pred_cond_norm_dir,
             output_file_name), buffer, 44100)


# %%
# generate predictions; model A; condition: EVO


model_a_pred_cond_evo_dir = "data/RAVE_decoded_datasets/model_a_pred_cond_evo"
os.makedirs(model_a_pred_cond_evo_dir, exist_ok=True)

evo_initials_encoded = np.load("genes/genes.npy")  # (500, 8, 64)

seq_len = args_a.seq_len  # 256
label_len = args_a.label_len  # 128
pred_len = args_a.pred_len  # 4
generated_length = 64
num_generations = generated_length // pred_len  # 1


for ind in tqdm.tqdm(range(evo_initials_encoded.shape[0])):
    # get data
    x = torch.from_numpy(evo_initials_encoded[ind]).unsqueeze(
        0).transpose(1, 2).float().to(device)  # (1, 64, 8)
    # repeat 4 times to get a sequence of length 256
    x = x.repeat(1, 4, 1)  # (1, 256, 8)

    # apply global scaling
    x = test_dataset_a.scale_to_global_minmax(x)  # (1, seq_len, 8)

    # create buffers for model forward pass
    zeros4preds = torch.zeros_like(
        x[:, -pred_len:, :]).float().to(device)  # (1, pred_len, 8)
    y_labels = x[:, -label_len:, :].to(device)  # (1, label_len, 8)

    # autoregressive loop
    generated = []
    with torch.no_grad():
        for i in range(num_generations):
            # create a new decoder input
            # (1, label_len + pred_len, 8)
            dec_inp = torch.cat([y_labels, zeros4preds], dim=1).float()
            # get model predictions
            model_preds = model_a.model(
                x, None, dec_inp, None)  # (1, pred_len, 8)
            # append the predictions to the generated list
            generated.append(model_preds)
            # shift x by pred_len and append the predictions
            x = torch.cat([x[:, pred_len:, :], model_preds],
                          dim=1)  # (1, seq_len, 8)
            y_labels = x[:, -label_len:, :]  # (1, label_len, 8)

    # concatenate all the generated predictions
    generated = torch.cat(generated, dim=1)  # (1, generated_length, 8)

    # use the global min/max to inverse transform the generated predictions
    generated = test_dataset_a.unscale_from_global_minmax(generated)

    # reload rave model
    rave_model = torch.jit.load(rave_model_path, map_location=device)
    rave_model.eval()

    # decode
    with torch.no_grad():
        decoded = rave_model.decode(generated.transpose(1, 2))

    # save
    buffer = decoded.transpose(1, 2).squeeze(0).cpu().numpy()
    output_file_name = f"model_a_pred_cond_evo_{ind}.wav"
    sf.write(os.path.join(model_a_pred_cond_evo_dir,
             output_file_name), buffer, 44100)


# %%
# generate FAD scores for model A predictions

# to use `vggish`
frechet = FrechetAudioDistance(
    ckpt_dir="./checkpoints/vggish",
    model_name="vggish",
    sample_rate=16000,
    use_pca=False,
    use_activation=False,
    verbose=False
)

# Specify the paths to your saved embeddings
background_embds_path = "/data/FAD/test_bg_embeddings.npy"
model_a_test_embds_path = "/data/FAD/model_a_test_embeddings.npy"
model_a_norm_embds_path = "/data/FAD/model_a_norm_embeddings.npy"
model_a_evo_embds_path = "/data/FAD/model_a_evo_embeddings.npy"

background_test_set_path = "/data/RAVE_decoded_datasets/test_ds_decoded"
model_a_test_set_path = "/data/RAVE_decoded_datasets/model_a_pred_cond_test"
model_a_norm_set_path = "/data/RAVE_decoded_datasets/model_a_pred_cond_norm"
model_a_evo_set_path = "/data/RAVE_decoded_datasets/model_a_pred_cond_evo"

# %%
# Compute FAD score while reusing the saved embeddings (or saving new ones if paths are provided and embeddings don't exist yet)

model_a_test_fad_score = frechet.score(
    background_test_set_path,
    model_a_test_set_path,
    background_embds_path=background_embds_path,
    eval_embds_path=model_a_test_embds_path,
    dtype="float32"
)

model_a_norm_fad_score = frechet.score(
    background_test_set_path,
    model_a_norm_set_path,
    background_embds_path=background_embds_path,
    eval_embds_path=model_a_norm_embds_path,
    dtype="float32"
)

model_a_evo_fad_score = frechet.score(
    background_test_set_path,
    model_a_evo_set_path,
    background_embds_path=background_embds_path,
    eval_embds_path=model_a_evo_embds_path,
    dtype="float32"
)

# %%
