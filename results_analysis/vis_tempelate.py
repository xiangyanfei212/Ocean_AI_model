#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import netCDF4 as nc

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from datetime import datetime
from icecream import ic

total_var_names = {
    "T0":0,  "T50":1,  "T100":2,  "T300":3,  "T500":4,  "T1000":5,
    "S0":6,  "S50":7,  "S100":8,  "S300":9,  "S500":10, "S1000":11,
    "U0":12, "U50":13, "U100":14, "U300":15, "U500":16, "U1000":17,
    "V0":18, "V50":19, "V100":20, "V300":21, "V500":22, "V1000":30,
    "ssh": 24,
}

global_means_path = "/home/bingxing2/home/scx6115/Ocean_AI_model/sample_00/train/global_means.npy"
global_stds_path =  "/home/bingxing2/home/scx6115/Ocean_AI_model/sample_00/train/global_stds.npy"
land_mask_file = "/home/bingxing2/home/scx6115/Ocean_AI_model/sample_00/land_mask.h5"

def read_pre_tru_from_h5(file_path, var_name, time_step):
    var_idx = total_var_names[var_name]

    means = np.load(global_means_path)
    stds = np.load(global_stds_path)
    print('means: ', means.shape, 'stds: ', stds.shape)
    means = means[0, var_idx, 0, 0]
    stds = stds[0, var_idx, 0, 0]

    tfid = nc.Dataset(file_path)
    # print(tfid.variables)
    pre = tfid.variables['predicted'][0, time_step, var_idx, :, :] 
    tru = tfid.variables['ground_truth'][0, time_step, var_idx, :, :]  # truth data

    pre = pre * stds + means
    tru = tru * stds + means

    return pre, tru

def get_land_mask(depth):
    depth_list = [0, 50, 100, 300, 500, 1000]
    depth_idx = depth_list.index(depth)

    dt = nc.Dataset(land_mask_file)
    # print(tfid.variables)
    mask = dt.variables['fields'][0, depth_idx, :, :] 
    return mask

def global_plot(in_dir, model_name, var_name='ssh', units='m', time_step=5):
    init_time = ''

    lon = np.arange(0, 360, 0.25)    # lontitude
    lat = np.arange(90, -90, -0.25)  # latitude
    print('lon:', len(lon))
    print('lat:', len(lat))

    file_path = os.path.join(in_dir, 'autoregressive_predictions.h5')
    pre, tru = read_pre_tru_from_h5(file_path, var_name, time_step)

    # for colorbar
    levels = np.arange(np.min(tru), np.max(tru) * 1.1, (np.max(tru)*1.1-np.min(tru))/100)
    
    # mask land
    land_mask = get_land_mask(0) # 0:land, 1:ocean 
    pre[land_mask==0] = np.nan
    tru[land_mask==0] = np.nan

    # add cycle to avoid the white line
    pre, lons = add_cyclic_point(pre, coord=lon)
    tru, lons = add_cyclic_point(tru, coord=lon)

    # projection
    proj = ccrs.PlateCarree(central_longitude=0)

    # ticker 
    lat_formatter = LatitudeFormatter()
    lon_formatter = LongitudeFormatter()

    ######  vis prediction ######
    fig = plt.figure(figsize=(8, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    cs = ax.contourf(lons, lat, pre, levels=levels, 
                     transform = proj, 
                     cmap='coolwarm', extend='both')
    cbar = plt.colorbar(cs, shrink=0.9, orientation='horizontal', label=var_name)

    ax.coastlines()

    ax.set_xticks(np.arange(-180,181,60), crs=proj)
    ax.xaxis.set_major_formatter(lon_formatter)

    ax.set_yticks(np.arange(-90,91,30), crs=proj)
    ax.yaxis.set_major_formatter(lat_formatter)


    ax.set_title(f'{model_name}, Prediction\n{var_name} ({units}) \nInit Time:{init_time}, forecast steps = {time_step} (day)')
    plt.savefig(os.path.join(in_dir, f'prediction_{model_name}_{var_name}_{init_time}_{time_step}.png'), dpi=300)
    plt.close()
    #####################

    ##### vis truth #####
    fig = plt.figure(figsize=(8, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    cs = ax.contourf(lons, lat, tru, levels=levels,
                     transform = proj, 
                     cmap='coolwarm', extend='both')
    cbar = plt.colorbar(cs, shrink=0.9, orientation='horizontal', label=var_name)

    ax.coastlines()

    ax.set_xticks(np.arange(-180, 181, 60), crs=proj)
    ax.xaxis.set_major_formatter(lon_formatter)

    ax.set_yticks(np.arange(-90,91,30), crs=proj)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_title(f'Truth\n{var_name} ({units}) \nInit Time:{init_time}, forecast steps = {time_step} (day)')
    plt.savefig(os.path.join(in_dir, f'truth_{var_name}_{init_time}_{time_step}.png'), dpi=300)
    plt.close()
    #####################


if __name__ == '__main__':
    in_dir = '/home/bingxing2/home/scx6115/Ocean_AI_model/exp_6_levels/Masked_AE_Ocean/20230610'
    model_name = 'Masked_AE_Ocean'
    global_plot(in_dir, model_name, var_name='ssh', units='m',   time_step=5)
    global_plot(in_dir, model_name, var_name='U0',  units='m/s', time_step=5)
    global_plot(in_dir, model_name, var_name='V0',  units='m/s', time_step=5)
    global_plot(in_dir, model_name, var_name='S0',  units='psu', time_step=5)
    global_plot(in_dir, model_name, var_name='T0',  units='$\circ$C', time_step=5)
