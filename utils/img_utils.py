import h5py
import math
import glob
import random
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from types import new_class

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.functional as TF


class PeriodicPad2d(nn.Module):
    """
    pad longitudinal (left-right) circular
    and pad latitude (top-bottom) with zeros
    """

    def __init__(self, pad_width):
        super(PeriodicPad2d, self).__init__()
        self.pad_width = pad_width

    def forward(self, x):
        # pad left and right circular
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular")
        # pad top and bottom zeros
        out = F.pad(
            out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0
        )
        return out


def reshape_fields(
    img, inp_or_tar, params, train, normalize=True, orog=None, add_noise=False
):
    # Takes in np array of size (n_history+1, c, h, w)
    # returns torch tensor of size ((n_channels*(n_history+1), crop_size_x, crop_size_y)

    if len(np.shape(img)) == 3:
        img = np.expand_dims(img, 0)

    if np.shape(img)[2] == 721:
        img = img[:, :, 0:720, :]  # remove last pixel

    n_history = np.shape(img)[0] - 1
    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]
    n_channels = np.shape(img)[1]  # this will either be N_in_channels or N_out_channels
    channels = params.in_channels if inp_or_tar == "inp" else params.out_channels

    if normalize and params.normalization == "minmax":
        maxs = np.load(params.global_maxs_path)[:, channels]
        mins = np.load(params.global_mins_path)[:, channels]
        img = (img - mins) / (maxs - mins)

    if normalize and params.normalization == "zscore":
        means = np.load(params.global_means_path)[:, channels]
        stds = np.load(params.global_stds_path)[:, channels]
        img -= means
        img /= stds

    if normalize and params.normalization == "zscore_lat":
        means = np.load(params.global_lat_means_path)[:, channels, :720]
        stds = np.load(params.global_lat_stds_path)[:, channels, :720]
        img -= means
        img /= stds

    if params.add_grid:
        if inp_or_tar == "inp" and params.gridtype == "linear":
            assert (
                params.N_grid_channels == 2
            ), "N_grid_channels must be set to 2 for gridtype linear"
            x = np.meshgrid(np.linspace(-1, 1, img_shape_x))
            y = np.meshgrid(np.linspace(-1, 1, img_shape_y))
            grid_x, grid_y = np.meshgrid(y, x)
            grid = np.stack((grid_x, grid_y), axis=0)
        if inp_or_tar == "inp" and params.gridtype == "sinusoidal":
            assert (
                params.N_grid_channels == 4
            ), "N_grid_channels must be set to 4 for gridtype sinusoidal"
            x1 = np.meshgrid(np.sin(np.linspace(0, 2 * np.pi, img_shape_x)))
            x2 = np.meshgrid(np.cos(np.linspace(0, 2 * np.pi, img_shape_x)))
            y1 = np.meshgrid(np.sin(np.linspace(0, 2 * np.pi, img_shape_y)))
            y2 = np.meshgrid(np.cos(np.linspace(0, 2 * np.pi, img_shape_y)))
            grid_x1, grid_y1 = np.meshgrid(y1, x1)
            grid_x2, grid_y2 = np.meshgrid(y2, x2)
            grid = np.expand_dims(
                np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis=0), axis=0
            )
        img = np.concatenate((img, grid), axis=1)

    if params.orography and inp_or_tar == "inp":
        # print('img:', img.shape, 'orog:', orog.shape)
        orog = np.expand_dims(orog, axis=(0, 1))
        orog = np.repeat(orog, repeats=img.shape[0], axis=0)
        # print('img:', img.shape, 'orog:', orog.shape)
        img = np.concatenate((img, orog), axis=1)
        n_channels += 1

    img = np.squeeze(img)

    if add_noise:
        img = img + np.random.normal(0, scale=params.noise_std, size=img.shape)

    return torch.as_tensor(img)


def reshape_downstream_fields(img, inp_or_tar, params, normalize=True):

    if len(np.shape(img)) == 3:
        img = np.expand_dims(img, 0)

    if np.shape(img)[2] == 361:
        img = img[:, :, 0:360, :]  # remove last pixel

    n_history = np.shape(img)[0] - 1
    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]

    if inp_or_tar == "inp":
        channels = params.downstream_in_channels
    if inp_or_tar == "force":
        channels = params.downstream_force_channels
    if inp_or_tar == "tar":
        channels = params.downstream_out_channels

    if normalize and params.normalization == "minmax":
        maxs = np.load(params.downstream_global_maxs_path)[:, channels]
        mins = np.load(params.downstream_global_mins_path)[:, channels]
        img = (img - mins) / (maxs - mins)

    if normalize and params.normalization == "zscore":
        means = np.load(params.downstream_global_means_path)[:, channels]
        stds = np.load(params.downstream_global_stds_path)[:, channels]
        img -= means
        img /= stds

    img = np.squeeze(img)

    return torch.as_tensor(img)


def read_max_min_value(min_max_val_file_path):
    with h5py.File(min_max_val_file_path, "r") as f:
        max_values = f["max_values"]
        min_values = f["min_values"]
    return max_values, min_values
