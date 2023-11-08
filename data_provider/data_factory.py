from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_RAVEnc
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'RAVEnc': Dataset_RAVEnc,
}

def data_provider_ravenc(args, flag, scaler=None, quantizer=None, train_set=None):
    shuffle_flag = flag == 'train'
    drop_last = flag == 'train'
    scale_flag = args.scale > 0
    quantize_flag = args.quantize > 0
    # if flag == 'test':
    #     quantize_flag = False
    # keep batch size for pred to 1
    # but for train, val and test use the batch size from args
    batch_size = 1 if flag == 'pred' else args.batch_size
    data_set = Dataset_RAVEnc(
        root_path=args.root_path,
        data_path=args.data_path,
        csv_path=args.csv_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        scale=scale_flag,
        quantize=quantize_flag,
        num_clusters=args.quantizer_num_clusters,
        all_in_memory=True,
        scaler=scaler,
        quantizer=quantizer,
        train_set=train_set,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


def data_provider(args, flag, scaler=None):
    Data = data_dict[args.data]

    # for our RAVE encoded datasets
    if args.data.lower() == 'ravenc':
        shuffle_flag = flag == 'train'
        drop_last = flag == 'train'
        # keep batch size for pred to 1
        # but for train, val and test use the batch size from args
        batch_size = 1 if flag == 'pred' else args.batch_size
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            csv_path=args.csv_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            scale=True,
            all_in_memory=True,
            scaler=scaler,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

    # for everything else (their original datasets)
    else:
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
