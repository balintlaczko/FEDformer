import torch
import random
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from data_provider.data_factory import data_provider_ravenc
from models.RNN import RNNModel
import gin
import gin.torch

@gin.configurable
def train(
        batch_size,
        root_path,
        data_path,
        csv_path,
        seq_len,
        label_len,
        pred_len,
        num_workers,
        num_devices,
        train_epochs,
        train_steps_limit,
        val_steps_limit,
        checkpoints_dir,
        checkpoint_name,
        resume_ckpt_path,
):

    # create args object for data provider
    class Args():
        pass

    args = Args()
    args.batch_size = batch_size
    args.root_path = root_path
    args.data_path = data_path
    args.csv_path = csv_path
    args.seq_len = seq_len
    args.label_len = label_len
    args.pred_len = pred_len
    args.num_workers = num_workers

    # create data loaders for train and val
    train_set, train_loader = data_provider_ravenc(args, flag="train")
    _, val_loader = data_provider_ravenc(args, flag="val", scaler=train_set.scaler, train_set=train_set)

    # create model
    model = RNNModel()

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoints_dir,
        filename=checkpoint_name,
        save_top_k=1,
        mode="min",
    )

    # loggers
    csv_logger = CSVLogger(save_dir="./logs/", name=checkpoint_name)
    tensorboard_logger = TensorBoardLogger(save_dir="./logs/", name=checkpoint_name)

    # trainer settings
    trainer_strategy = "ddp_find_unused_parameters_true" if num_devices != 1 else "auto"
    train_steps_limit = train_steps_limit if train_steps_limit > 0 else None
    val_steps_limit = val_steps_limit if val_steps_limit > 0 else None

    # create trainer
    trainer = Trainer(
        strategy=trainer_strategy,
        devices=num_devices, 
        accelerator="gpu",
        max_epochs=train_epochs, 
        enable_checkpointing=True, 
        limit_train_batches=train_steps_limit, 
        limit_val_batches=val_steps_limit, 
        callbacks=[checkpoint_callback],
        logger=[csv_logger, tensorboard_logger],
    )

    # train
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_ckpt_path)


def main():
    # fix_seed = 2021
    # random.seed(fix_seed)
    # torch.manual_seed(fix_seed)
    # np.random.seed(fix_seed)

    # read gin config
    # gin.external_configurable(ModelCheckpoint)
    # gin.external_configurable(CSVLogger)
    # gin.external_configurable(TensorBoardLogger)
    # gin.external_configurable(Trainer)
    gin.parse_config_file("configs/rnn.gin")

    # train
    train()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("highest")
    main()