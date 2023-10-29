# %%
# imports
import os
import pandas as pd
import matplotlib.pyplot as plt

# %%
# load the log file
log_file = os.path.join(os.getcwd(), 'lightning_logs', 'version_15', 'metrics.csv')
df = pd.read_csv(log_file)

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
plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig(os.path.join(os.path.dirname(log_file), 'loss.png'))
plt.show()


# %%
