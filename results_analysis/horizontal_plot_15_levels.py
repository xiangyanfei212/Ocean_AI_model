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

from var_channels import VAR_INDEX_15_LEVELS, LEVELS_15_STATS, LEVELS_15 

def read_pre_tru_from_h5(file_path, var_name, time_step):

    var_idx = VAR_INDEX_15_LEVELS[var_name]
    means = np.load(LEVELS_15_STATS['global_means_path'])
    stds = np.load(LEVELS_15_STATS['global_stds_path'])

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
    depth_idx = LEVELS_15.index(depth)

    dt = nc.Dataset(LEVELS_15_STATS['land_mask_file'])
    # print(tfid.variables)
    mask = dt.variables['fields'][0, depth_idx, :, :] 
    return mask


def plot_global_pre_gt(in_dir, model_name, var_name='SSH', depth=0, units='m', time_step=5):
    init_time = ''

    lon = np.arange(0, 360, 0.25)    # lontitude
    lat = np.arange(90, -90, -0.25)  # latitude
    print('lon:', len(lon))
    print('lat:', len(lat))

    file_path = os.path.join(in_dir, 'autoregressive_predictions.h5')
    pre, tru = read_pre_tru_from_h5(file_path, var_name, time_step)

    # for colorbar
    color_levels = np.arange(np.min(tru), np.max(tru) * 1.1, (np.max(tru)*1.1-np.min(tru))/100)
    
    # mask land
    land_mask = get_land_mask(depth) # 0:land, 1:ocean 
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
    cs1 = ax1.contourf(lons, lat, pre, 
                       levels=color_levels, 
                       transform = proj, 
                       cmap='coolwarm', extend='both')
    ax1.coastlines()
    ax1.set_xticks(np.arange(-180,181,60), crs=proj)
    lon_formatter = LongitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)

    ax1.set_yticks(np.arange(-90,91,30), crs=proj)
    lat_formatter = LatitudeFormatter()
    ax1.yaxis.set_major_formatter(lat_formatter)

    # Truth
    ax2 = fig.add_subplot(2, 1, 2, projection=proj)
    cs2 = ax2.contourf(lons, lat, tru, levels=color_levels,
                       transform = proj, 
                       cmap='coolwarm', extend='both')
    ax2.coastlines()
    ax2.set_xticks(np.arange(-180, 181, 60), crs=proj)
    lon_formatter = LongitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)

    ax2.set_yticks(np.arange(-90,91,30), crs=proj)
    lat_formatter = LatitudeFormatter()
    ax2.yaxis.set_major_formatter(lat_formatter)

    cbar = plt.colorbar(cs2, shrink=0.9, orientation='horizontal', label='Sea Surface Height')


    ax1.set_title(f'{model_name}, Prediction\n{var_name} ({units}) \nInit Time:{init_time}, forecast steps = {time_step} (day)')
    ax2.set_title(f'Truth\n{var_name} ({units}) \nInit Time:{init_time}, forecast steps = {time_step} (day)')

    # plt.tight_layout()
    print('Saving... ')
    plt.savefig(os.path.join(in_dir, f'{model_name}_{var_name}_{init_time}_{time_step}.png'), dpi=300)
    plt.show()


if __name__ == '__main__':


    # %% Plotting the global field with the layout of two subplots (prediction and ground-truth)
    # in_dir = '/home/bingxing2/home/scx6115/Ocean_AI_model/exp_15_levels/afno/20230628-131420'
    # model_name = 'afno'

    in_dir = '../exp_15_levels/Masked_AE_Ocean/20230628-131150/' 
    model_name = 'Masked_AE_Ocean'

    in_dir = '../exp_15_levels/Masked_AE_Ocean/20230628-131150/2_steps_finetune'
    model_name = 'Masked_AE_Ocean_20230628-131150_two_steps_finetune'

    plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='SSH', units='m', time_step=5)
    plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='SSH', units='m', time_step=15)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='SSH', units='m', time_step=20)
    plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='SSH', units='m', time_step=25)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='SSH', units='m', time_step=29)

    plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='T0', units='', time_step=5)
    plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='T0', units='', time_step=15)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='T0', units='', time_step=20)
    plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='T0', units='', time_step=25)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='T0', units='', time_step=29)

    plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='S0', units='', time_step=5)
    plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='S0', units='', time_step=15)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='S0', units='', time_step=20)
    plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='S0', units='', time_step=25)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='S0', units='', time_step=29)

    plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='U0', units='', time_step=5)
    plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='U0', units='', time_step=15)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='U0', units='', time_step=20)
    plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='U0', units='', time_step=25)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='U0', units='', time_step=29)

    plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='V0', units='', time_step=5)
    plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='V0', units='', time_step=15)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='V0', units='', time_step=20)
    plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='V0', units='', time_step=25)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, depth=0, var_name='V0', units='', time_step=29)
