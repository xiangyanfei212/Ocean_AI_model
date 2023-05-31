# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The code was authored by the following people:
#
# Jaideep Pathak - NVIDIA Corporation
# Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
# Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
# Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory
# Ashesh Chattopadhyay - Rice University
# Morteza Mardani - NVIDIA Corporation
# Thorsten Kurth - NVIDIA Corporation
# David Hall - NVIDIA Corporation
# Zongyi Li - California Institute of Technology, NVIDIA Corporation
# Kamyar Azizzadenesheli - Purdue University
# Pedram Hassanzadeh - Rice University
# Karthik Kashinath - NVIDIA Corporation
# Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation
# 
# Yanfei Xiang

import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py
import math
# import cv2
from utils.img_utils import reshape_fields


def get_data_loader(params, files_pattern, distributed, train):
    dataset = GetDataset(params, files_pattern, train)
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None  # 获取data 的sampler
    # 在多机多卡情况下分布式训练数据的读取也是一个问题，不同的卡读取到的数据应该是不同的。
    # dataparallel的做法是直接将batch切分到不同的卡，这种方法对于多机来说不可取，因为多机之间直接进行数据传输会严重影响效率。
    # 于是有了利用sampler确保dataloader只会load到整个数据集的一个特定子集的做法。
    # DistributedSampler就是做这件事的。它为每一个子进程划分出一部分数据集，以避免不同进程之间数据重复。

    # 构造加载器dataloader
    dataloader = DataLoader(dataset,
                            batch_size=int(params.batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=False,  # (sampler is None),
                            sampler=sampler if train else None,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available())  # pin_memory能加快内存的Tensor转义到GPU的显存的速度

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


class GetDataset(Dataset):
    def __init__(self, params, location, train):
        self.params = params
        self.location = location
        self.train = train
        self.orography = params.orography
        self.normalize = params.normalize
        self.dt = params.dt  # 需要预测的时间节点个数
        self.n_history = params.n_history # 需要考虑的过去数据的时间节点个数
        self.in_channels = np.array(params.in_channels)
        self.out_channels = np.array(params.out_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)

        self._get_files_stats()
        self.add_noise = params.add_noise if train else False


    # 获取文件统计信息
    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)

        with h5py.File(self.files_paths[0], 'r') as _f: 
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            # self.n_samples_per_year = _f['fields'].shape[0] # 每年数据量
            self.n_samples_per_year = 250 
            # !! 这里指定为300的原因是因为每年的样本数量有多有少，选择最少的年份的样本量作为n_samples_per_year，否则后面会报"out of index"

            # original image shape (before padding)
            self.img_shape_x = _f['fields'].shape[2] - 1 # just get rid of one of the pixels
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]

        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location,
                                                                                                       self.n_samples_total,
                                                                                                       self.img_shape_x,
                                                                                                       self.img_shape_y,
                                                                                                       self.n_in_channels))
        logging.info("Delta t: {} days".format(1 * self.dt))
        logging.info("Including {} days of past history in training at a frequency of {} days".format(
            1 * self.dt * self.n_history, 1 * self.dt))

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file['fields'] 

        if self.orography and self.params.normalization == 'zscore': 
            _orog_file = h5py.File(self.params.orography_norm_zscore_path, 'r')
        if self.orography and self.params.normalization == 'maxmin': 
            _orog_file = h5py.File(self.params.orography_norm_maxmin_path, 'r')
        self.orography_field = _orog_file['orog']

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx = int(global_idx / self.n_samples_per_year)  # which year
        local_idx = int(global_idx % self.n_samples_per_year)  # which sample in a year

        if self.files[year_idx] is None:
            self._open_file(year_idx)

        # If there are not enough historical time steps available in the features, shift to future time steps.
        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        # If the sample is the final one for the year, predict the current time step. Otherwise, predict the next time step.
        step = 0 if local_idx >= self.n_samples_per_year - self.dt else self.dt

        if self.orography:
            orog = self.orography_field 
            if np.shape(orog)[0] == 721:
                orog = orog[0:720]
            # logging.info(f'orog: {orog.shape}')
        else:
            orog = None
        


        # logging.info(f'year_idx: {year_idx}, local_idx:{local_idx}, dt:{self.dt}, n_history:{self.n_history}, in_channels:{self.in_channels}')
        # logging.info(self.files[year_idx][(local_idx - self.dt * self.n_history):(local_idx + 1):self.dt, self.in_channels].shape)
        # logging.info(self.files[year_idx][local_idx + step, self.out_channels].shape)

        inp = reshape_fields( self.files[year_idx][(local_idx - self.dt * self.n_history):(local_idx + 1):self.dt, self.in_channels], 'inp', self.params, self.train, self.normalize, orog, self.add_noise )
        tar = reshape_fields( self.files[year_idx][local_idx + step, self.out_channels], 'tar', self.params, self.train, self.normalize, orog )

        return inp, tar 
