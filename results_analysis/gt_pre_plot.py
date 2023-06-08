#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import matplotlib
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.mpl.ticker as mticker
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt

from datetime import datetime

total_var_names = {
    "T0":0,  "T50":1,  "T100":2,  "T300":3,  "T500":4,  "T1000":5,
    "S0":6,  "S50":7,  "S100":8,  "S300":9,  "S500":10, "S1000":11,
    "U0":12, "U50":13, "U100":14, "U300":15, "U500":16, "U1000":17,
    "V0":18, "V50":19, "V100":20, "V300":21, "V500":22, "V1000":23,
    "ssh": 24,
}

global_means_path = "/work/home/acrzcyisbk/Ocean_AI_model/sample_03/train/global_means.npy"
global_stds_path = "/work/home/acrzcyisbk/Ocean_AI_model/sample_03/train/global_stds.npy"


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


def date2idx(yr='18', mon='01', day='01', hrs='00:00:00'):
    # 输入4个字符串变量，分别为年、月、日、时
    # 输出为输入的时间以六小时为单位在该年中的参数
    yr = str(yr);
    mon = str(mon);
    day = str(day);
    slash = '/';
    space = ' '
    conbine = day + slash + mon + slash + yr + space + hrs
    time_data = conbine
    # time_data = '4/04/18 00:00:00'
    struct = '%d/%m/%y %H:%M:%S'
    date = datetime.strptime(time_data, struct)
    day_of_year = date.timetuple().tm_yday - 1
    hour_of_day = date.timetuple().tm_hour
    hours_since_jan_01_epoch = 24 * day_of_year + hour_of_day
    idx = int(hours_since_jan_01_epoch / 6)
    print(conbine)
    return idx


def get_land_mask(depth):
    depth_list = [0, 50, 100, 300, 500, 1000]
    depth_idx = depth_list.index(depth)

    land_mask_file = '../sample_03/land_mask.h5'
    dt = nc.Dataset(land_mask_file)
    # print(tfid.variables)
    mask = dt.variables['fields'][0, depth_idx, :, :] 
    return mask


def plot_global_pre_gt(in_dir, var_name='ssh', units='m', time_step=5):
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

    # init figure
    fig = plt.figure(figsize=(8, 14), dpi=300)
    proj = ccrs.PlateCarree(central_longitude=0)

    # prediction
    ax1 = fig.add_subplot(2, 1, 1, projection=proj)
    cs1 = ax1.contourf(lons, lat, pre, levels=levels, 
                       transform = proj, 
                       cmap='coolwarm', extend='both')
    ax1.coastlines()
    ax1.set_xticks(np.arange(-180,181,60), crs=proj)
    lon_formatter = mticker.LongitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)

    ax1.set_yticks(np.arange(-90,91,30), crs=proj)
    lat_formatter = mticker.LatitudeFormatter()
    ax1.yaxis.set_major_formatter(lat_formatter)

    # Truth
    ax2 = fig.add_subplot(2, 1, 2, projection=proj)
    cs2 = ax2.contourf(lons, lat, tru, levels=levels,
                       transform = proj, 
                       cmap='coolwarm', extend='both')
    ax2.coastlines()
    ax2.set_xticks(np.arange(-180, 181, 60), crs=proj)
    lon_formatter = mticker.LongitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)

    ax2.set_yticks(np.arange(-90,91,30), crs=proj)
    lat_formatter = mticker.LatitudeFormatter()
    ax2.yaxis.set_major_formatter(lat_formatter)

    cbar = plt.colorbar(cs2, shrink=0.9, orientation='horizontal', label='Sea Surface Height')


    ax1.set_title(f'Prediction\n{var_name} ({units}) \nInit Time:{init_time}, forecast steps = {time_step} (day)')
    ax2.set_title(f'Truth\n{var_name} ({units}) \nInit Time:{init_time}, forecast steps = {time_step} (day)')

    # plt.tight_layout()
    print('Saving... ')
    plt.savefig(os.path.join(in_dir, f'{var_name}_{init_time}_{time_step}.png'), dpi=300)
    plt.show()


if __name__ == '__main__':


    # %% Plotting the global field with the layout of two subplots (prediction and ground-truth)
    # in_dir = '/work/home/acrzcyisbk/Ocean_AI_model/exp/afno_backbone/20230524-163725/'
    in_dir = '/work/home/acrzcyisbk/Ocean_AI_model/exp/Masked_AE_Ocean/20230528-Mask_AE_v2_landmask/'
    plot_global_pre_gt(in_dir=in_dir, var_name='ssh', units='m', time_step=5)
    plot_global_pre_gt(in_dir=in_dir, var_name='ssh', units='m', time_step=15)
    plot_global_pre_gt(in_dir=in_dir, var_name='ssh', units='m', time_step=20)
    plot_global_pre_gt(in_dir=in_dir, var_name='ssh', units='m', time_step=25)
    plot_global_pre_gt(in_dir=in_dir, var_name='ssh', units='m', time_step=23)

    plot_global_pre_gt(in_dir=in_dir, var_name='T0', units='', time_step=5)
    plot_global_pre_gt(in_dir=in_dir, var_name='T0', units='', time_step=15)
    plot_global_pre_gt(in_dir=in_dir, var_name='T0', units='', time_step=20)
    plot_global_pre_gt(in_dir=in_dir, var_name='T0', units='', time_step=25)
    plot_global_pre_gt(in_dir=in_dir, var_name='T0', units='', time_step=23)

    plot_global_pre_gt(in_dir=in_dir, var_name='S0', units='', time_step=5)
    plot_global_pre_gt(in_dir=in_dir, var_name='S0', units='', time_step=15)
    plot_global_pre_gt(in_dir=in_dir, var_name='S0', units='', time_step=20)
    plot_global_pre_gt(in_dir=in_dir, var_name='S0', units='', time_step=25)
    plot_global_pre_gt(in_dir=in_dir, var_name='S0', units='', time_step=23)

    plot_global_pre_gt(in_dir=in_dir, var_name='U0', units='', time_step=5)
    plot_global_pre_gt(in_dir=in_dir, var_name='U0', units='', time_step=15)
    plot_global_pre_gt(in_dir=in_dir, var_name='U0', units='', time_step=20)
    plot_global_pre_gt(in_dir=in_dir, var_name='U0', units='', time_step=25)
    plot_global_pre_gt(in_dir=in_dir, var_name='U0', units='', time_step=23)

    plot_global_pre_gt(in_dir=in_dir, var_name='V0', units='', time_step=5)
    plot_global_pre_gt(in_dir=in_dir, var_name='V0', units='', time_step=15)
    plot_global_pre_gt(in_dir=in_dir, var_name='V0', units='', time_step=20)
    plot_global_pre_gt(in_dir=in_dir, var_name='V0', units='', time_step=25)
    plot_global_pre_gt(in_dir=in_dir, var_name='V0', units='', time_step=23)
