# %%
# imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# %%
# parse arguments
parser = argparse.ArgumentParser(description='Plot the logs')
parser.add_argument('--log_files', nargs='+', default=[], help='log files')
args = parser.parse_args()

# %%
# read all csv files into a list of dataframes
dfs = []
for log_file in args.log_files:
    dfs.append(pd.read_csv(log_file))
# combine the dataframes
df = pd.concat(dfs)


# %%
# keep only columns epoch, train_loss_epoch, val_loss_epoch
df = df[['epoch', 'train_loss_epoch', 'val_loss_epoch']]
df.head()
# %%
# fill empty values in epoch column with the last value
df['epoch'] = df['epoch'].fillna(method='ffill')
df.head()
# %%
# keep only rows where there is either a train_loss_epoch or a val_loss_epoch value
df = df[df['train_loss_epoch'].notna() | df['val_loss_epoch'].notna()]
df.head()
# %%
# fill empty values in train_loss_epoch column with the last value
df['train_loss_epoch'] = df['train_loss_epoch'].fillna(method='ffill')
# fill empty values in val_loss_epoch column with the last value
df['val_loss_epoch'] = df['val_loss_epoch'].fillna(method='ffill')
df.head()
# %%
# drop odd rows
df = df[df.index % 2 == 1]
df.head()
# %%
# plot the train and validation loss, use log scale for y axis
plt.figure(figsize=(15, 10))
plt.plot(df['epoch'], df['train_loss_epoch'], label='train_loss_epoch')
plt.plot(df['epoch'], df['val_loss_epoch'], label='val_loss_epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss')
plt.locator_params(axis='both', nbins=20) 
plt.grid(True, which='both')
plt.legend()
plt.savefig(os.path.join(os.path.dirname(args.log_files[-1]), 'loss.png'))
plt.show()


# %%
