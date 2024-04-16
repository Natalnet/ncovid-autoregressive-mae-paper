import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader


class DataManagement():
    def __init__(self,
                 data,
                 features,
                 targets_features,
                 inseqlen=7,
                 outseqlen=7,
                 normalize_data=True
        ):
        super().__init__()

        self.data = data
        self.features = features
        self.targets_features = targets_features
        self.inseqlen = inseqlen
        self.outseqlen = outseqlen
        if normalize_data:
            self.data_normalization = 'MinMax'
            self.data_normalize()

    def data_normalize(self):
        if self.data_normalization == 'MinMax':
            self.scaler = MinMaxScaler()
            self.data[self.data.columns] = self.scaler.fit_transform(self.data[self.data.columns])

        self.data = self.data[~self.data.index.duplicated(keep='first')]

    def data_tensor_generate(self, input_len, output_len):
        if input_len:
            self.inseqlen = input_len
        if output_len:
            self.outseqlen = output_len

        window_dataset = self.data.iloc[:-self.outseqlen].rolling(self.inseqlen, min_periods=1, win_type=None, center=False)

        inputs, targets = [], []
        for window in window_dataset:
            if len(window) == self.inseqlen:
                inpt = window[self.features]
                inputs.append(inpt.T.values)
                last_inpt_day = pd.Timestamp(inpt.index[-1])
                next_day_after_last = last_inpt_day + pd.DateOffset(1)
                start_pred_range = next_day_after_last
                data_forward_range = pd.date_range(start=start_pred_range, periods=self.outseqlen)
                trg = self.data[self.targets_features].loc[data_forward_range]
                targets.append(trg.T.values)

        self.inputs_tensor = torch.Tensor(np.array(inputs))
        self.targets_tensor = torch.Tensor(np.array(targets))

    def set_as_train_data(self, x, y):
        self.X_train = x
        self.Y_train = y
    
    def set_as_test_data(self, x, y):
        self.X_test = x
        self.Y_test = y
    
    def train_test_split(self, prct_to_train=0.8):
        self.percent_to_train = prct_to_train
        x_to_train = self.inputs_tensor[:int(len(self.inputs_tensor)*prct_to_train)]
        y_to_train = self.targets_tensor[:int(len(self.targets_tensor)*prct_to_train)]
        self.set_as_train_data(x_to_train, y_to_train)

        x_to_test = self.inputs_tensor[int(len(self.inputs_tensor)*prct_to_train):]
        y_to_test = self.targets_tensor[int(len(self.targets_tensor)*prct_to_train):]
        self.set_as_test_data(x_to_test, y_to_test)

    def dataloader_train_generate(self, batch_size=32):
        self.batch_size = batch_size
        self.data_train_tensor = TensorDataset(self.X_train, self.Y_train)
        self.data_train = DataLoader(self.data_train_tensor, batch_size=batch_size, shuffle=False)
    
    def dataloader_test_generate(self, batch_size=32):
        self.batch_size = batch_size
        self.data_test_tensor = TensorDataset(self.X_test, self.Y_test)
        self.data_test = DataLoader(self.data_test_tensor, batch_size=batch_size, shuffle=False)

    def dataloader_create(self, batch_size=32):
        self.dataloader_train_generate(batch_size)
        self.dataloader_test_generate(batch_size)
    
    def data_to_test_create(self, data_test=None):
        if data_test is not None:
            self.data_to_test = data_test
        else:
            self.data_to_test = self.data

        data_test_vect = []
        for i in range(0, len(self.data_to_test), self.outseqlen):
            window = self.data_to_test[self.features][i: i+self.inseqlen]
            if len(window) == self.inseqlen:
                data_test_vect.append(window.T.values)

        self.data_to_test = torch.Tensor(np.array(data_test_vect))
    
    def data_split_by_feature(self):
        features_splitted = []
        for i in range(self.X_train.shape[1]):
            ft = torch.Tensor([input[i].numpy() for input in self.X_train])
            ft_dataset = TensorDataset(ft, ft)
            ft_dataloader = DataLoader(ft_dataset, batch_size=self.batch_size, shuffle=False)
            features_splitted.append(ft_dataloader)
        
        self.splited_data = features_splitted
    
    def att_data(self, new_features, new_targets_features, data=None):
        if data:
            self.data = data
        self.features = new_features
        self.targets_features = new_targets_features

        self.data_tensor_generate(self.inseqlen, self.outseqlen)
        self.train_test_split(self.percent_to_train)
        self.dataloader_create(self.batch_size)
        self.data_split_by_feature()
