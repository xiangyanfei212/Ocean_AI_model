import os
import glob
import h5py
import numpy as np
import pandas as pd
from icecream import ic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_data_loader(data_path:str, years:list, mode:str, normalize:str, norm_file_path:str,
                    batch_size:int, num_workers:int):
    """
    params:
        data_path: the path of sample files
        years: a list
        mode: 'train' or 'valid' or 'test'
        normalize: 'min_max_value_norm'
        norm_file_path: value for normalize
    """


    dataset = GetDataset(data_path, years, mode, normalize, norm_file_path)


    dataloader = DataLoader(dataset,
                            batch_size = batch_size,
                            shuffle = True if mode == 'train' else False,
                            sampler = None,
                            batch_sampler=None,
                            num_workers = num_workers,
                            drop_last = False,
                            pin_memory = torch.cuda.is_available())

    return dataloader


class GetDataset(Dataset):
    def __init__(self, sample_dir:str, years:list, mode:str, normalize:str, min_max_val_file_path:str):
        self.sample_dir = sample_dir
        self.mode = mode
        self.years = years
        self.n_years = len(years)
        self.normalize = normalize
        self.min_max_val_file_path = min_max_val_file_path

        self._get_files_stats()

        self.features = [None for _ in range(self.n_years * 12)]
        self.labels   = [None for _ in range(self.n_years * 12)]

    def _get_files_stats(self):
        self.files_paths = [os.path.join(self.sample_dir, f"{y}.h5") for y in self.years]
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f['feature'].shape[0]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        print("Number of samples per month: {}".format(self.n_samples_per_year))
        print("Number of samples: {}".format(self.n_samples_total))

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.features[year_idx] = _file['feature']
        self.labels[year_idx]   = _file['label']

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        ic(global_idx, self.n_samples_per_year)
        year_idx = int(global_idx/self.n_samples_per_year) # which year we are on
        local_idx = int(global_idx%self.n_samples_per_year) # which sample in that year we are on - determines indices for centering

        # open file
        if self.features[year_idx] is None:
            self._open_file(year_idx)

        X  = self.features[year_idx][local_idx]
        Y  = self.labels[year_idx][local_idx]

        if self.normalize == 'min_max_norm':
            X, Y = normalize_data_using_min_max_value(self.min_max_val_file_path, X, Y)

        return torch.as_tensor(X), torch.as_tensor(Y)

def normalize_data_using_min_max_value(min_max_val_file_path, feature, label):

    with h5py.File(min_max_val_file_path, 'r') as f:
        feature_max_value = f['feature_max_value']
        feature_min_value = f['feature_min_value']
        label_max_value = f['label_max_value']
        label_min_value = f['label_min_value']

        ic(feature_max_value.shape, feature.shape)

        feature_max_value = np.expand_dims(feature_max_value, axis=(1,2))
        feature_max_value = np.repeat(feature_max_value, feature.shape[1], axis=1)
        feature_max_value = np.repeat(feature_max_value, feature.shape[2], axis=2)

        feature_min_value = np.expand_dims(feature_min_value, axis=(1,2))
        feature_min_value = np.repeat(feature_min_value, feature.shape[1], axis=1)
        feature_min_value = np.repeat(feature_min_value, feature.shape[2], axis=2)

        label_max_value = np.expand_dims(label_max_value, axis=(1,2))
        label_max_value = np.repeat(label_max_value, label.shape[1], axis=1)
        label_max_value = np.repeat(label_max_value, label.shape[2], axis=2)

        label_min_value = np.expand_dims(label_min_value, axis=(1,2))
        label_min_value = np.repeat(label_min_value, label.shape[1], axis=1)
        label_min_value = np.repeat(label_min_value, label.shape[2], axis=2)

        ic(feature.shape, feature_min_value.shape)
        ic(np.max(feature), np.min(feature), np.max(label), np.min(label))

        feature = (feature - feature_min_value) / (feature_max_value - feature_min_value)
        label = (label - label_min_value) / (label_max_value - label_min_value)

    print('After normalize...')
    ic(np.max(feature), np.min(feature), np.max(label), np.min(label))

    return feature, label


if __name__ == '__main__':

    data_path = '/work/home/acrzcyisbk/Ocean_AI_model/sample_01'
    years = [1995, 1996, 1997]
    mode = 'train'
    normalize = 'min_max_norm'
    batch_size = 4
    num_workers = 4
    norm_file_path = os.path.join(data_path, 'min_max_val_1995_1997.h5')

    data_loader = get_data_loader(data_path, years, mode, normalize, norm_file_path,
                                  batch_size, num_workers)
    for trn_i, (X, Y) in enumerate(data_loader):
        print(f'train index: {trn_i}')
        print(f'X: {X.shape}')
        print(f'Y: {Y.shape}')

        # assert torch.max(X) <= 1.1
        # assert torch.min(X) >= -0.1
        # assert torch.max(Y) <= 1.1
        # assert torch.min(Y) >= -0.1
