import argparse
import os
import sys
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from data_provider.data_factory import data_provider_ravenc
from models import FEDformer


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(
        description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    # parser.add_argument('--is_training', type=int, default=1, help='status')
    # parser.add_argument('--task_id', type=str, default='test', help='task id')
    # parser.add_argument('--model', type=str, default='FEDformer',
    #                     help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')

    # supplementary config for FEDformer model
    # parser.add_argument('--version', type=str, default='Fourier',
    #                     help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64,
                        help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str,
                        default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')

    # data loader
    # parser.add_argument('--data', type=str,
    #                     default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str,
                        default='./dataset/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str,
                        default='ETTh1.csv', help='data file')
    parser.add_argument('--csv_path', type=str,
                        default='ETTh1.csv', help='csv file')
    # parser.add_argument('--features', type=str, default='M',
    #                     help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
    #                          'S:univariate predict univariate, MS:multivariate predict univariate')
    # parser.add_argument('--target', type=str, default='OT',
    #                     help='target feature in S or MS task')
    # parser.add_argument('--freq', type=str, default='h',
    #                     help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
    #                          'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    # parser.add_argument('--detail_freq', type=str, default='h',
    #                     help='like freq, but use in predict')
    # parser.add_argument('--checkpoints', type=str,
    #                     default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=32,
                        help='input sequence length')
    parser.add_argument('--label_len', type=int,
                        default=16, help='start token length')
    parser.add_argument('--pred_len', type=int, default=8,
                        help='prediction sequence length')
    # parser.add_argument('--cross_activation', type=str, default='tanh'

    # model define
    parser.add_argument('--enc_in', type=int, default=8,
                        help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=8,
                        help='decoder input size')
    parser.add_argument('--c_out', type=int, default=8, help='output size')
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='dimension of fcn')
    parser.add_argument(
        '--moving_avg', type=int, default=6, help='window size of moving average')
    
    # data scaling and quantization
    parser.add_argument('--scale', type=int, default=1, help='whether to scale data')
    parser.add_argument('--scaler_load_path', type=str, default='checkpoints/scaler.pkl', help='path to where to load the fit scaler from')
    parser.add_argument('--quantize', type=int, default=1, help='whether to quantize data')
    parser.add_argument('--quantizer_type', type=str, default='kmeans', help='type of quantizer to use')
    parser.add_argument('--quantizer_num_clusters', type=int, default=64, help='number of clusters for quantizer')
    parser.add_argument('--quantizer_load_path', type=str, default='checkpoints/quantizer.pt', help='path to where to load the fit k-means quantizer from')
    # parser.add_argument('--factor', type=int, default=1, help='attn factor')
    # parser.add_argument('--distil', action='store_false',
    #                     help='whether to use distilling in encoder, using this argument means not using distilling',
    #                     default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='token_only',
                        help='input embedding, options:[token_only, token_pos]')
    parser.add_argument('--activation', type=str,
                        default='gelu', help='activation')

    # optimization
    parser.add_argument('--num_workers', type=int,
                        default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int,
                        default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--train_steps_limit', type=int, default=-1, help='train steps limit. -1 means no limit')
    parser.add_argument('--val_steps_limit', type=int, default=-1, help='validation steps limit. -1 means no limit')
    # parser.add_argument('--patience', type=int, default=3,
    #                     help='early stopping patience')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='optimizer learning rate')
    # parser.add_argument('--des', type=str, default='test',
    #                     help='exp description')
    # parser.add_argument('--loss', type=str, default='mse',
    #                     help='loss function')
    # parser.add_argument('--lradj', type=str, default='type1',
    #                     help='adjust learning rate')
    # parser.add_argument('--use_amp', action='store_true',
    #                     help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--num_devices', type=int, nargs='*', default=[-1],
                        help='number of gpus to use. if list it will be the device ids to use')
    
    # checkpoint & logging
    parser.add_argument('--ckpt_name', type=str, default='model_hpc', help='checkpoint name')
    parser.add_argument('--resume_ckpt_path', type=str, default=None,)

    # quick comment
    parser.add_argument('--comment', type=str, default='', help='add a comment if needed')

    args = parser.parse_args()
    args.quantize = True

    print('Args in experiment:')
    print(args)

    # create data loaders for train and val
    train_set, train_loader = data_provider_ravenc(args, "train", scaler=args.scaler_load_path, quantizer=args.quantizer_load_path)
    _, val_loader = data_provider_ravenc(args, "val", scaler=train_set.scaler, quantizer=train_set.quantizer, train_set=train_set)

    fedformer = FEDformer.LitFEDformer(args)

    # checkpoint callback
    checkpoint_path = "./checkpoints/" + args.ckpt_name
    val_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_val_{epoch:02d}-{val_loss:.4f}",
        save_top_k=10,
        mode="min",
    )
    last_checkpoint_callback = ModelCheckpoint(
        monitor="epoch",
        dirpath=checkpoint_path,
        filename=args.ckpt_name + "_last_{epoch:02d}",
        save_top_k=1,
        mode="max",
    )

    # csv logger callback
    csv_logger = CSVLogger(save_dir="./logs/", name=args.ckpt_name)
    tensorboard_logger = TensorBoardLogger(save_dir="./logs/", name=args.ckpt_name)

    train_steps_limit = args.train_steps_limit if args.train_steps_limit > 0 else None
    val_steps_limit = args.val_steps_limit if args.val_steps_limit > 0 else None

    trainer_strategy = "ddp_find_unused_parameters_true" if len(args.num_devices) != 1 else "auto"
    # trainer_strategy = "auto"
    trainer = pl.Trainer(
        strategy=trainer_strategy,
        devices=args.num_devices, 
        accelerator="gpu",
        max_epochs=args.train_epochs, 
        enable_checkpointing=True, 
        limit_train_batches=train_steps_limit, 
        limit_val_batches=val_steps_limit, 
        callbacks=[val_checkpoint_callback, last_checkpoint_callback],
        logger=[csv_logger, tensorboard_logger],
    )
    
    trainer.fit(model=fedformer, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume_ckpt_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
