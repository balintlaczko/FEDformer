import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import h5py
import tqdm
from torchpq.clustering import KMeans
from pickle import dump, load
import math

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # 12 * 30 * 24 is presumably a year
        # border1s = [0, a year - seq_len, a year + 4 months - seq_len]
        border1s = [0, 12 * 30 * 24 - self.seq_len,
                    12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        # border2s = [a year, a year + 4 months, a year + 8 months]
        border2s = [12 * 30 * 24, 12 * 30 * 24 +
                    4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        # for "train": border1 = 0, border2 = a year
        # for "val": border1 = a year - seq_len, border2 = a year + 4 months
        # for "test": border1 = a year + 4 months - seq_len, border2 = a year + 8 months
        # this will set the borders according to the flag
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            # the values after the first column
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            # the scaler is always fitted on the training data
            train_data = df_data[border1s[0]:border2s[0]]  # 0 to a year length
            self.scaler.fit(train_data.values)
            # ...and then applied to the whole data
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # take the date column within the sequence borders (for train: 0 -> a year)
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # if no time encoding is used, then the date values are filled with ones(?)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            # dropping year?
            data_stamp = df_stamp.drop(['date'], 1).values
        # if we use time encoding, then scale all time scales between -0.5 and 0.5
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(
                df_stamp['date'].values), freq=self.freq)
            # data_stamp.shape: (features, dates)
            # transpose to (dates, features)
            data_stamp = data_stamp.transpose(1, 0)

        # data_x and data_y hold the same data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # data_stamp is also between border1 and border2
        # and it holds the date and time
        # where each scope (year, month, week, day, hour, etc) is encoded between -0.5 and 0.5
        # and is returned together as a multivariate series
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # s_begin and s_end are the start and end index of the input sequence
        s_begin = index
        s_end = s_begin + self.seq_len  # 24 * 4 * 4
        # r_begin and r_end are the start and end index of the output sequence
        # it starts at the last label_len steps of the input sequence
        # and ends at the input sequence + pred_len steps position
        r_begin = s_end - self.label_len  # 24 * 4
        r_end = r_begin + self.label_len + self.pred_len  # 24 * 4 + 24 * 4

        # data_x and data_y hold the same data
        # seq_x is the input sequence (begin --> begin + seq_len)
        seq_x = self.data_x[s_begin:s_end]
        # seq_x.shape: (seq_len, features)
        # seq_y is the output sequence (begin + seq_len - label_len --> begin + seq_len + pred_len)
        seq_y = self.data_y[r_begin:r_end]
        # seq_y.shape: (label_len + pred_len, features)
        # x_mark and y_mark are the encoded date and time of the input and output sequences
        seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_x_mark.shape: (seq_len, time features)
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # seq_y_mark.shape: (label_len + pred_len, time features)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 *
                    30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 *
                    30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(
                df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len,
                    len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(
                df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(
            tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(
                df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_RAVEnc(Dataset):
    def __init__(
        self,
        root_path="/Volumes/T7RITMO/RAVE_encoded_datasets",
        data_path="vctk_rave_encoded.h5",
        csv_path="vctk_rave_encoded.csv",
        flag='train',
        size=None,  # [seq_len, label_len, pred_len]
        scale=True,
        scaler=None,
        quantize=False,
        num_clusters=64,
        quantizer=None,
        quantizer_type="kmeans",
        all_in_memory=True,
        train_set=None,
    ) -> None:
        super().__init__()

        # parse size, if provided [seq_len, label_len, pred_len]
        if size == None:
            # TODO: what should the default values be for us?
            self.seq_len = 32
            self.label_len = 16
            self.pred_len = 8
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # parse flag
        assert flag in ['train', 'test', 'val']
        self.flag = flag

        # parse paths to data
        self.root_path = root_path
        self.data_path = data_path
        self.csv_path = csv_path

        # parse options
        self.scale = scale
        self.all_in_memory = all_in_memory

        # parse scaler
        self.scaler_is_fit = False
        # if the argument is a string, load the pickled scaler from file
        if type(scaler) == str:
            self.scaler = load(open(scaler, 'rb'))
        else:
            self.scaler = scaler
        if self.scaler != None:
            self.scaler_is_fit = True

        # qunatizer
        self.quantize = quantize
        self.quantizer = quantizer
        self.num_clusters = num_clusters
        self.quantizer_type = quantizer_type
        self.quantizer_is_fit = False
        if self.quantize:
            if self.quantizer_type == "kmeans":
                if self.quantizer == None:
                    self.quantizer = KMeans(n_clusters=self.num_clusters, distance="euclidean")
                elif type(quantizer) == str:
                    self.quantizer = KMeans(n_clusters=self.num_clusters, distance="euclidean")
                    self.quantizer.load_state_dict(torch.load(quantizer))
                    self.quantizer_is_fit = True
            elif self.quantizer_type == "msprior":
                self.quantizer = self.msprior_quantizer
            else:
                raise NotImplementedError("quantizer type not implemented")

        # optionally get whole file embeddings from train set
        self.whole_file_embeddings = None
        if train_set != None:
            self.whole_file_embeddings = train_set.whole_file_embeddings

        # read data and generate chunked dataset
        self.__read_data__()

    def __read_data__(self) -> None:
        """
        Read the dataset, filter for the set we want (train/val/test),
        then read all dataset elements in the chosen set and chunk them 
        into seq_len + pred_len chunks, and return a dataset of samples 
        where: ds[i] = (embedding_id, start_frame)
        """

        # read the csv file
        self.df = pd.read_csv(os.path.join(self.root_path, self.csv_path))
        # filter for the set we want (train/val/test)
        self.df = self.df[self.df.dataset == self.flag]

        # load all embeddings in memory if needed
        # and generate the chunk dataset
        if self.all_in_memory:
            self.load_all_in_memory()
            self.generate_chunk_dataset_from_memory()
        else:
            self.generate_chunk_dataset()

        # fit scaler if needed
        if self.scale or self.quantize:
            if self.flag == 'train':
                self.fit_scaler()

    def __getitem__(self, index):
        """
        Get a sample from the dataset, where a sample is a tuple of:
        (seq_x, seq_y). seq_x is the input sequence (start_frame --> start_frame + seq_len),
        seq_y is the output sequence (start_frame + seq_len - label_len --> start_frame + seq_len + pred_len).

        Args:
            index: The index in the chunked dataset. We use it to look up the embedding id and start frame.

        Returns:
            tuple(torch.Tensor): The input and output sequences.
        """
        dataset_index, start_frame = self.chunk_dataset[index]
        if not self.all_in_memory:
            # fetch the embedding chunk from the h5 file
            with h5py.File(os.path.join(self.root_path, self.data_path), 'r') as h5_file:
                chunk = self.get_chunk(h5_file, dataset_index, start_frame)
        else:
            chunk = self.get_chunk_from_memory(dataset_index, start_frame)
        # reshape from BCT to BTC
        chunk = chunk.transpose(1, 2)
        # scale the chunk if needed
        if self.scale:
            chunk = self.scaler.transform(chunk.squeeze(0).numpy())
            chunk = torch.from_numpy(chunk).unsqueeze(0)
        # quantize the chunk if needed
        if self.quantize:
            if self.quantizer_type == "kmeans":
                quantizer_labels = self.quantizer.predict(chunk.transpose(1, 2).squeeze(0))
                chunk = self.quantizer.centroids.transpose(0, 1)[quantizer_labels].unsqueeze(0)
            elif self.quantizer_type == "msprior":
                chunk = self.quantizer(chunk, resolution=self.num_clusters)
        # extract the input and output sequences
        seq_x, seq_y = self.get_x_y(chunk) # now returns BTC

        return seq_x.squeeze(0), seq_y.squeeze(0)  # as TC

    def __len__(self):
        # return the number of chunks
        return self.chunk_dataset.shape[0]
    
    def save_scaler(self, destination_path: str):
        """
        Save the (fit) standard scaler to a file.
        """
        dump(self.scaler, open(destination_path, 'wb'))
    
    def save_quantizer(self, destination_path: str):
        """
        Save the (fit) K-means quantizer to a file.
        """
        torch.save(self.quantizer.state_dict(), destination_path)

    def msprior_quantizer(self, x, resolution=64):
        # this part is from https://github.com/caillonantoine/msprior/blob/1e7b1e27b366e051bbc244613f07d1c62b7dc3a1/msprior_scripts/preprocess.py#L108
        x = x / 2
        x = .5 * (1 + torch.erf(x / math.sqrt(2)))
        x = torch.floor(x * (resolution - 1))
        x = x / (resolution - 1) - 0.5 # but this is me scaling it into [-0.5, 0.5]
        return x

    def fit_scaler(self) -> None:
        """
        Fit a standard scaler to the dataset.
        """
        # guard clause to avoid fitting the scaler twice
        if self.scaler_is_fit:
            print("scaler is already fitted")
            if self.quantizer_type == "kmeans":
                self.quantizer = self.quantizer.cpu()
            return
        self.scaler = StandardScaler()
        # get progress bar from chunks dataset
        pbar = tqdm.tqdm(self.whole_file_embeddings)
        pbar.set_description("fitting standard scaler")
        all_embeddings = torch.cat(self.whole_file_embeddings, dim=-1) # the embeddings are BCT
        print("all embeddings shape: ", all_embeddings.shape)
        # channels last
        all_embeddings = all_embeddings.transpose(1, 2) # BTC
        print("all embeddings shape: ", all_embeddings.shape)

        # fit the scaler
        if self.scale:
            print("fitting scaler...")
            all_embeddings = self.scaler.fit_transform(all_embeddings.squeeze(0).numpy())
            print("scaler fitted")
            all_embeddings = torch.from_numpy(all_embeddings)

        # fit quantizer
        if self.quantize:
            if not self.quantizer_is_fit:
                print("fitting quantizer...")
                cluster_ids_x = self.quantizer.fit(all_embeddings.cuda().transpose(0, 1).contiguous())
                print("quantizer fitted")
            else:
                print("quantizer is already fitted")
            self.quantizer = self.quantizer.cpu()


    def inverse_transform(self, data):
        # inverse transform the data with a scaler fit to the chunks ds
        if self.scaler == None:
            self.fit_scaler()
        # scale without the batch dimension
        it = self.scaler.inverse_transform(data.squeeze(0).numpy())
        # return with the batch dimension
        return torch.from_numpy(it).unsqueeze(0)

    def load_all_in_memory(self) -> None:
        """
        Load all embeddings in memory.
        """
        # guard clause to avoid loading the embeddings twice
        if self.whole_file_embeddings != None:
            return
        self.whole_file_embeddings = []
        with h5py.File(os.path.join(self.root_path, self.data_path), 'r') as f:
            # get progress bar form indices
            pbar = tqdm.tqdm(f.keys())
            pbar.set_description("loading all embeddings in memory")
            for i in pbar:
                self.whole_file_embeddings.append(
                    torch.from_numpy(f[str(int(i))][()]))

    def generate_chunk_dataset(self) -> None:
        """
        Generate a list of tuples (embedding_id, start_frame)
        to get a dataset of chunks of length seq_len + pred_len.
        For efficiency, we just save the index of the file embedding
        from the h5 file, and the start frame of the chunk.
        We can use that to extract the input and output sequences.
        """
        self.chunk_dataset = []
        # open the h5 file
        with h5py.File(os.path.join(self.root_path, self.data_path), 'r') as h5_file:
            # for each dataset in the csv file
            for i, row in self.df.iterrows():
                # get the dataset index
                dataset_index = row["dataset_index"].values[0]
                # retrieve the whole embedding as a tensor
                embedding = h5_file[str(int(dataset_index))][()]
                embedding = torch.from_numpy(embedding)
                # generate the start frames for the chunks
                chunk_indices = self.chunk_indices(embedding.shape[-1])
                # append each chunk index with the dataset index to the chunk dataset
                for chunk_index in chunk_indices:
                    self.chunk_dataset.append(
                        list((dataset_index, chunk_index)))
        self.chunk_dataset = np.array(self.chunk_dataset)

    def generate_chunk_dataset_from_memory(self) -> None:
        """
        Generate a list of tuples (embedding_id, start_frame)
        to get a dataset of chunks of length seq_len + pred_len.
        For efficiency, we just save the index of the file embedding
        from the h5 file, and the start frame of the chunk.
        We can use that to extract the input and output sequences.
        """
        self.chunk_dataset = []
        # for each dataset in the csv file
        for i, row in self.df.iterrows():
            # get the dataset index
            # dataset_index = row["dataset_index"].values[0]
            dataset_index = row["dataset_index"]
            # retrieve the whole embedding as a tensor
            embedding = self.whole_file_embeddings[dataset_index]
            # generate the start frames for the chunks
            chunk_indices = self.chunk_indices(embedding.shape[-1])
            # append each chunk index with the dataset index to the chunk dataset
            for chunk_index in chunk_indices:
                self.chunk_dataset.append(
                    list((dataset_index, chunk_index)))
        self.chunk_dataset = np.array(self.chunk_dataset)
        # print("chunk dataset shape in memory: ", self.chunk_dataset.shape)

    def chunk_indices(self, tensor_length: int):
        """
        Given the length of a tensor, return a list of inidices as
        start frames for chunks of length seq_len + pred_len

        Args:
            tensor_length (int): The length of the tensor (sequence) to chunk into smaller sequences of length seq_len + pred_len.

        Returns:
            list: A list of indices as start frames for chunks of length seq_len + pred_len.
        """
        # get the number of chunks
        num_chunks = tensor_length - (self.seq_len + self.pred_len)
        # return the range of start indices
        return list(range(num_chunks))

    def get_chunk(self, h5_file: h5py.File, dataset_index: int, start_frame: int) -> torch.Tensor:
        """
        Given the h5 file, the dataset index, and the start frame,
        return the chunk of length seq_len + pred_len.

        Args:
            h5_file (h5py.File): The opened h5 file.
            dataset_index (int): The index of the tensor in the h5 file. The value under the column "dataset_index" in the corresponding csv file.
            start_frame (int): The starting frame of the chunk within the entire sequence (that represents the whole audio file).

        Returns:
            torch.Tensor: The chunk of length seq_len + pred_len.
        """
        # read the full tensor from the h5 file
        tensor = torch.from_numpy(h5_file[str(int(dataset_index))][()])
        # extract the chunk
        return tensor[..., start_frame:start_frame+self.seq_len+self.pred_len]

    def get_chunk_from_memory(self, dataset_index: int, start_frame: int) -> torch.Tensor:
        """
        Given the dataset index, and the start frame,
        return the chunk of length seq_len + pred_len.

        Args:
            dataset_index (int): The index of the tensor in the h5 file. The value under the column "dataset_index" in the corresponding csv file.
            start_frame (int): The starting frame of the chunk within the entire sequence (that represents the whole audio file).

        Returns:
            torch.Tensor: The chunk of length seq_len + pred_len.
        """
        # read the full tensor of the embedded audio file
        tensor = self.whole_file_embeddings[dataset_index]
        # extract the chunk
        return tensor[..., start_frame:start_frame+self.seq_len+self.pred_len]

    def get_x_y(self, chunk: torch.Tensor):
        """
        Given a chunk of length seq_len + pred_len, return the input and output sequences.

        Args:
            chunk (torch.Tensor): The chunk of length seq_len + pred_len.

        Returns:
            tuple: The input and output sequences.
        """
        # chunk as BCT:
        # seq_x = chunk[..., :self.seq_len]
        # seq_y = chunk[..., self.seq_len -
        #               self.label_len:self.seq_len+self.pred_len]

        # chunk as BTC:
        seq_x = chunk[:, :self.seq_len, :]
        seq_y = chunk[:, self.seq_len -
                      self.label_len:self.seq_len+self.pred_len, :]
        # print("seq_x shape: ", seq_x.shape)
        # print("seq_y shape: ", seq_y.shape)
        return seq_x, seq_y
