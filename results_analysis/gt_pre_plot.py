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


def read_pre_tru_from_h5(levels:int, file_path, var_name, time_step):

    if levels == 6:
        var_idx = VAR_INDEX_6_levels[var_name]
        means = np.load(levels_6_stats['global_means_path'])
        stds = np.load(levels_6_stats['global_stds_path'])
    if levels == 15:
        var_idx = VAR_INDEX_15_levels[var_name]
        means = np.load(levels_15_stats['global_means_path'])
        stds = np.load(levels_15_stats['global_stds_path'])

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

    dt = nc.Dataset(levels_6_stats['land_mask_file'])
    # print(tfid.variables)
    mask = dt.variables['fields'][0, depth_idx, :, :] 
    return mask


def plot_global_pre_gt(levels:int, in_dir, model_name, var_name='ssh', depth:int, units='m', time_step=5):
    init_time = ''

    lon = np.arange(0, 360, 0.25)    # lontitude
    lat = np.arange(90, -90, -0.25)  # latitude
    print('lon:', len(lon))
    print('lat:', len(lat))

    file_path = os.path.join(in_dir, 'autoregressive_predictions.h5')
    pre, tru = read_pre_tru_from_h5(levels, file_path, var_name, time_step)

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


    ax1.set_title(f'{model_name}, Prediction\n{var_name} ({units}) \nInit Time:{init_time}, forecast steps = {time_step} (day)')
    ax2.set_title(f'Truth\n{var_name} ({units}) \nInit Time:{init_time}, forecast steps = {time_step} (day)')

    # plt.tight_layout()
    print('Saving... ')
    plt.savefig(os.path.join(in_dir, f'{model_name}_{var_name}_{init_time}_{time_step}.png'), dpi=300)
    plt.show()

def plot_global_ssh(in_dir:str, 
                    show_pre:bool,
                    model_name:str, 
                    vmin        = 0,
                    vmax        = 10,
                    lon_cff     = 20, 
                    lat_cff     = 20, 
                    xstep        = 5, 
                    ystep       = 3, 
                    extent      = [-180, 180, -90, 90], 
                    level_num   = 3,
                    cmap        = 'Blues',
                    title       = 'SSH (m)',
                    var_name    = 'ssh', 
                    units       = 'm', 
                    time_step   = 5):
    '''
    display the global ssh via 'coutour' and 'contourf' 

    params:
        in_dir: the directory of autoregressive_predictions.h5
        model_name: the name of the model, used in figure's title
        extent: area to be displayed
        var_name: variable to be displayed
        units: the unit of variable, used in figure's title
        time_step: the forecast time step to be displayed, used in figure's title

    '''
    init_time = ''

    lon = np.arange(0, 360, 0.25)    # lontitude
    lat = np.arange(90, -90, -0.25)  # latitude
    print('lon:', len(lon))
    print('lat:', len(lat))

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    file_path = os.path.join(in_dir, 'autoregressive_predictions.h5')
    pre, tru = read_pre_tru_from_h5(file_path, var_name, time_step)
    if show_pre:
        data_2d = pre
    else:   
        data_2d = tru
    ic(np.nanmin(data_2d), np.nanmax(data_2d))

    # mask land
    # land_mask = get_land_mask(0) # 0:land, 1:ocean 
    # pre[land_mask==0] = np.nan
    # tru[land_mask==0] = np.nan

    # add cycle to avoid the white line
    data_2d, lons = add_cyclic_point(data_2d, coord=lon)

    # init figure
    fig = plt.figure(figsize=(20, 12), dpi=300)

    proj = ccrs.PlateCarree(central_longitude=0)
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    ax.set_extent(extent, crs=proj)
    ax.coastlines('50m', linewidth=1)

    ax.set_yticks(np.linspace(extent[2] + lat_cff, extent[3] - lat_cff, ystep), crs=proj)
    ax.set_xticks(np.linspace(extent[0] + lon_cff, extent[1] - lon_cff, xstep), crs=proj)

    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.tick_params(which='major', direction='out', length=8, width=2, pad=0.3, 
                   labelsize=20, bottom=True, left=True, right=False, top=False)

    ax.set_title(title, fontsize=12, loc='center')

    ic(lon.shape, lons.shape, data_2d.shape)
    ax.contour(lons, lat, data_2d, level_num, colors=['black'], transform=proj)

    levels = np.arange(np.min(tru), np.max(tru) * 1.1, (np.max(tru)*1.1-np.min(tru))/100)
    cs = ax.contourf(lons, lat, data_2d, levels=levels, 
                     transform = proj, 
                     cmap='coolwarm', extend='both')

    cbar = plt.colorbar(cs, ax=ax, shrink=0.7, 
                        pad=0.07, aspect=20, location='bottom',
                        cmap=cmap, norm=norm,
                        orientation='horizontal', label='Sea Surface Height')

    cbar.ax.set_xlabel('m/s', rotation=0, labelpad=10, fontsize=30)
    cbar.ax.tick_params(labelsize=25)

    print('Saving... ')
    plt.savefig(os.path.join(in_dir, f'ssh_{model_name}_{str(time_step)}.png'), dpi=300)
    plt.show()

if __name__ == '__main__':


    # %% Plotting the global field with the layout of two subplots (prediction and ground-truth)
    in_dir = '/home/bingxing2/home/scx6115/Ocean_AI_model/exp/afno_backbone/20230609-155507'
    model_name = 'AFNO'

    in_dir = '/home/bingxing2/home/scx6115/Ocean_AI_model/exp_6_levels/Masked_AE_Ocean/20230610'
    model_name = 'Masked_AE_Ocean'
    
    # plot_global_ssh(in_dir      = in_dir,
    #                 show_pre    = True,
    #                 model_name  = model_name, 
    #                 vmin        = -3,
    #                 vmax        = 3,
    #                 lon_cff     = 20, 
    #                 lat_cff     = 20, 
    #                 xstep        = 5, 
    #                 ystep       = 3, 
    #                 extent      = [-180, 180, -90, 90], 
    #                 level_num   = 3,
    #                 cmap        = 'Blues',
    #                 title       = 'SSH (m)',
    #                 var_name    = 'ssh', 
    #                 units       = 'm', 
    #                 time_step   = 5)

    
    # in_dir = '/home/bingxing2/home/scx6115/Ocean_AI_model/exp/Masked_AE_fusion/20230613-203822'
    # model_name = 'Masked_AE_fusion'

    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='ssh', units='m', time_step=5)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='ssh', units='m', time_step=15)
    # # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='ssh', units='m', time_step=20)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='ssh', units='m', time_step=25)
    # # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='ssh', units='m', time_step=29)

    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='T0', units='', time_step=5)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='T0', units='', time_step=15)
    # # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='T0', units='', time_step=20)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='T0', units='', time_step=25)
    # # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='T0', units='', time_step=29)

    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='S0', units='', time_step=5)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='S0', units='', time_step=15)
    # # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='S0', units='', time_step=20)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='S0', units='', time_step=25)
    # # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='S0', units='', time_step=29)

    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='U0', units='', time_step=5)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='U0', units='', time_step=15)
    # # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='U0', units='', time_step=20)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='U0', units='', time_step=25)
    # # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='U0', units='', time_step=29)

    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='V0', units='', time_step=5)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='V0', units='', time_step=15)
    # # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='V0', units='', time_step=20)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='V0', units='', time_step=25)
    # plot_global_pre_gt(in_dir=in_dir, model_name=model_name, var_name='V0', units='', time_step=29)
