import os
import h5py
import numpy as np
import netCDF4 as nc
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from var_channels import VAR_INDEX_15_LEVELS, LEVELS_15_STATS, LEVELS_15 


def plot_global_stream(
        pre_or_truth:str,        # 可视化预测还是真值，可选："pre"、"truth"
        projection:str,          # 投影方式，可选：'PlateCarree', 'robinson'
        file_path:str,           # 模型预报文件路径
        land_mask_file_path:str, # 海陆掩码文件路径
        interval:int,            # 经纬度间隔, 默认为1
        save_path:str,           # 图像保存路径
        init_condition:int,      # 第几次起报
        time_step:int,           # 第几个预报时间步
    ):
    u_var_idx = VAR_INDEX_15_LEVELS['U0']
    v_var_idx = VAR_INDEX_15_LEVELS['V0']

    # read means and stds, for inverse transform data
    means = np.load(LEVELS_15_STATS['global_means_path'])
    stds  = np.load(LEVELS_15_STATS['global_stds_path'])
    means_u = means[0, u_var_idx, 0, 0]
    means_v = means[0, v_var_idx, 0, 0]
    stds_u  = stds[0, u_var_idx, 0, 0]
    stds_v  = stds[0, v_var_idx, 0, 0]

    # read u, v
    tfid = nc.Dataset(file_path)
    if pre_or_truth == 'pre':
        data_u = tfid.variables['predicted'][init_condition, time_step, u_var_idx, :, :] 
        data_v = tfid.variables['predicted'][init_condition, time_step, v_var_idx, :, :] 
    elif pre_or_truth == 'truth':
        data_u = tfid.variables['ground_truth'][init_condition, time_step, u_var_idx, :, :] 
        data_v = tfid.variables['ground_truth'][init_condition, time_step, v_var_idx, :, :] 
    data_u = data_u * stds_u + means_u
    data_v = data_v * stds_v + means_v
    data = np.sqrt(data_u ** 2 + data_v ** 2)
    data = data[::interval,::interval]
    data = np.flipud(data)

    # read land mask
    # dt = nc.Dataset(LEVELS_15_STATS['land_mask_file'])
    # depth_idx = LEVELS_15.index(0) # 0m
    # land_mask = dt.variables['fields'][0, depth_idx, :, :] 
    # land_mask = land_mask[::interval,::interval]

    # set longitude and latitude
    longitude = np.linspace(-180,180,1441)[0:-1]  # 地图投影, 以180°为中心
    longitude = longitude[::interval]

    latitude = np.linspace(-90,90,721)[0:-1]
    latitude = latitude[::interval]
    longitude2d, latitude2d = np.meshgrid(longitude, latitude)

    if projection == 'PlateCarree':

        # 开始画图
        fig = plt.figure(figsize=(12, 10))
        proj = ccrs.PlateCarree(central_longitude = 180)      # central_longitude设置地图中心经度
        ax = fig.add_axes([0, 1, 0.6, 0.3], projection = proj) # 设置图片位置和大小（左下宽高）

        # 以下为经纬度设置
        ax.set_extent([0, 360.000001, -90, 90.000001], crs=ccrs.PlateCarree()) 
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, linestyle=':', color='k', alpha=0.8) # 绘制经纬线
        xticks = np.arange(0, 360.1, 60)
        yticks = np.arange(-90, 90.1, 30)
        ax.set_xticks(xticks, crs=ccrs.PlateCarree()) # 绘制经纬度标记                                               
        ax.set_yticks(yticks, crs=ccrs.PlateCarree()) 
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())

        lon_formatter = LongitudeFormatter(number_format='.0f', degree_symbol=r'$^o$')
        lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol=r'$^o$')
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(labelsize=10)

        # land
        land = cfeature.NaturalEarthFeature('physical', 'land', scale='110m', 
                                            edgecolor='black', facecolor='white', 
                                            lw=0.5, zorder=2) # 覆盖陆地
        ax.add_feature(land)
        cfeature.COLORS['land']

        # color
        mycolor = ['#FFFBCE','#EDDF96','#D9C55D','#B5B049','#8DA229','#5B9214','#3C8221','#15672E','#19522C','#1D3822']
        cmap_color = mpl.colors.LinearSegmentedColormap.from_list('my_list', mycolor)
        cmap = cmap_color

        # plot 
        cf = ax.pcolormesh(longitude2d, latitude2d, data, cmap=cmap, vmin=0, vmax=1.5, transform=ccrs.PlateCarree(central_longitude=180))

        # colorbar
        rect = [0.05, 0.92, 0.5, 0.02]
        cbar_ax = fig.add_axes(rect)    
        c = plt.colorbar(cf, cax=cbar_ax,extend="both",orientation='horizontal', aspect=20, pad=0.1)
        c.set_ticks(np.arange(0, 1.6, 0.1))  # 自定义数字间隔
        c.ax.tick_params(which='major', direction='in', labelsize=10, length=15) # 色卡设置

        #c.set_label('Global Stream',loc="center",fontsize=12,rotation=360)
        fig.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        return

    elif projection == 'robinson':

        # 开始画图
        fig = plt.figure(figsize=(12,10))
        
        # Use Robinson projection
        proj = ccrs.Robinson(central_longitude = 180)
        
        ax = fig.add_axes([0, 1, 0.6, 0.3],projection = proj)#设置图片位置和大小（左下宽高）
        ax.set_global()  # Set the extent to cover the whole globe
        
        land = cfeature.NaturalEarthFeature('physical', 'land', scale='110m', 
                                            edgecolor='black', facecolor='white',
                                            lw=0.5, zorder=2)#覆盖陆地
        ax.add_feature(land)
        
        # color
        mycolor = ['#FFFBCE','#EDDF96','#D9C55D','#B5B049','#8DA229','#5B9214','#3C8221','#15672E','#19522C','#1D3822']
        cmap_color = mpl.colors.LinearSegmentedColormap.from_list('my_list', mycolor)
        cmap = cmap_color
        
        # The range of Colorbar
        level = np.linspace(0,1.50001,16)
        
        cf = ax.contourf(longitude2d, latitude2d, data, cmap=cmap, levels=level, extend='both', 
                         transform=ccrs.PlateCarree(central_longitude = 180))

        # colorbar
        rect = [0.05,0.92,0.5,0.02]
        cbar_ax = fig.add_axes(rect)
        c=plt.colorbar(cf, cax=cbar_ax,extend="both",orientation='horizontal', aspect=20, pad=0.1)
        c.set_ticks(np.arange(0,1.500001,0.1)) #自定义数字间隔
        c.ax.tick_params(which='major',direction='in',labelsize=10,length=15) #色卡设置

        fig.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        return

def plot_global_stream_diff_between_pred_truth(
        projection:str,          # 投影方式，可选：'PlateCarree', 'robinson'
        file_path:str,           # 模型预报文件路径
        interval:int,            # 经纬度间隔, 默认为1
        save_path:str,           # 图像保存路径
        init_condition:int,      # 第几次起报
        time_step:int,           # 第几个预报时间步
    ):
    u_var_idx = VAR_INDEX_15_LEVELS['U0']
    v_var_idx = VAR_INDEX_15_LEVELS['V0']

    # read means and stds, for inverse transform data
    means = np.load(LEVELS_15_STATS['global_means_path'])
    stds  = np.load(LEVELS_15_STATS['global_stds_path'])
    means_u = means[0, u_var_idx, 0, 0]
    means_v = means[0, v_var_idx, 0, 0]
    stds_u  = stds[0, u_var_idx, 0, 0]
    stds_v  = stds[0, v_var_idx, 0, 0]

    # read u, v
    tfid = nc.Dataset(file_path)
    pre_data_u = tfid.variables['predicted'][init_condition, time_step, u_var_idx, :, :] 
    pre_data_v = tfid.variables['predicted'][init_condition, time_step, v_var_idx, :, :] 

    truth_data_u = tfid.variables['ground_truth'][init_condition, time_step, u_var_idx, :, :] 
    truth_data_v = tfid.variables['ground_truth'][init_condition, time_step, v_var_idx, :, :] 

    pre_data_u = pre_data_u * stds_u + means_u
    pre_data_v = pre_data_v * stds_v + means_v
    truth_data_u = truth_data_u * stds_u + means_u
    truth_data_v = truth_data_v * stds_v + means_v

    pre_data   = np.sqrt(pre_data_u ** 2 + pre_data_v ** 2)
    truth_data = np.sqrt(truth_data_u ** 2 + truth_data_v ** 2)

    data = pre_data - truth_data
    data = data[::interval,::interval]
    data = np.flipud(data)

    # set longitude and latitude
    longitude = np.linspace(-180,180,1441)[0:-1]  # 地图投影, 以180°为中心
    longitude = longitude[::interval]

    latitude = np.linspace(-90,90,721)[0:-1]
    latitude = latitude[::interval]
    longitude2d, latitude2d = np.meshgrid(longitude, latitude)

    vmin =  -0.5
    vmax = 0.5
    v_interval = 0.1

    if projection == 'PlateCarree':

        # 开始画图
        fig = plt.figure(figsize=(12, 10))
        proj = ccrs.PlateCarree(central_longitude = 180)      # central_longitude设置地图中心经度
        ax = fig.add_axes([0, 1, 0.6, 0.3], projection = proj) # 设置图片位置和大小（左下宽高）

        # 以下为经纬度设置
        ax.set_extent([0, 360.000001, -90, 90.000001], crs=ccrs.PlateCarree()) 
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, linestyle=':', color='k', alpha=0.8) # 绘制经纬线
        xticks = np.arange(0, 360.1, 60)
        yticks = np.arange(-90, 90.1, 30)
        ax.set_xticks(xticks, crs=ccrs.PlateCarree()) # 绘制经纬度标记                                               
        ax.set_yticks(yticks, crs=ccrs.PlateCarree()) 
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())

        lon_formatter = LongitudeFormatter(number_format='.0f', degree_symbol=r'$^o$')
        lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol=r'$^o$')
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(labelsize=10)

        # land
        land = cfeature.NaturalEarthFeature('physical', 'land', scale='110m', 
                                            edgecolor='black', facecolor='white', 
                                            lw=0.5, zorder=2) # 覆盖陆地
        ax.add_feature(land)
        cfeature.COLORS['land']

        # color
        # mycolor = ['#FFFBCE','#EDDF96','#D9C55D','#B5B049','#8DA229','#5B9214','#3C8221','#15672E','#19522C','#1D3822']
        # cmap_color = mpl.colors.LinearSegmentedColormap.from_list('my_list', mycolor)
        # cmap = cmap_color
        cmap = plt.get_cmap('RdBu')


        # plot 
        cf = ax.pcolormesh(longitude2d, latitude2d, data, cmap=cmap, 
                            vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(central_longitude=180))

        # colorbar
        rect = [0.05, 0.92, 0.5, 0.02]
        cbar_ax = fig.add_axes(rect)    
        c = plt.colorbar(cf, cax=cbar_ax,extend="both",orientation='horizontal', aspect=20, pad=0.1)
        c.set_ticks(np.arange(vmin, vmax+v_interval, v_interval))  # 自定义数字间隔
        c.ax.tick_params(which='major', direction='in', labelsize=10, length=15) # 色卡设置

        #c.set_label('Global Stream',loc="center",fontsize=12,rotation=360)
        fig.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        return

    elif projection == 'robinson':

        # 开始画图
        fig = plt.figure(figsize=(12,10))
        
        # Use Robinson projection
        proj = ccrs.Robinson(central_longitude = 180)
        
        ax = fig.add_axes([0, 1, 0.6, 0.3],projection = proj)#设置图片位置和大小（左下宽高）
        ax.set_global()  # Set the extent to cover the whole globe
        
        land = cfeature.NaturalEarthFeature('physical', 'land', scale='110m', 
                                            edgecolor='black', facecolor='white',
                                            lw=0.5, zorder=2)#覆盖陆地
        ax.add_feature(land)
        
        # color
        # mycolor = ['#FFFBCE','#EDDF96','#D9C55D','#B5B049','#8DA229','#5B9214','#3C8221','#15672E','#19522C','#1D3822']
        # cmap_color = mpl.colors.LinearSegmentedColormap.from_list('my_list', mycolor)
        # cmap = cmap_color
        cmap = plt.get_cmap('RdBu')
        
        # The range of Colorbar
        level = np.linspace(vmin, vmax, int((vmax-vmin) / v_interval))
        
        cf = ax.contourf(longitude2d, latitude2d, data, cmap=cmap, levels=level, extend='both', 
                         transform=ccrs.PlateCarree(central_longitude = 180))

        # colorbar
        rect = [0.05,0.92,0.5,0.02]
        cbar_ax = fig.add_axes(rect)
        c=plt.colorbar(cf, cax=cbar_ax,extend="both",orientation='horizontal', aspect=20, pad=0.1)
        c.set_ticks(np.arange(vmin, vmax+v_interval, v_interval))
        c.ax.tick_params(which='major',direction='in',labelsize=10,length=15) #色卡设置

        fig.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        return

def plot_kuroshio_stream(
        pre_or_truth:str,        # 可视化预测还是真值，可选："pre"、"truth"
        file_path:str,           # 模型预报文件路径
        save_path:str,           # 图像保存路径
        init_condition:int,      # 第几次起报
        time_step:int,           # 第几个预报时间步
):
    # Kuroshio Stream Contourf & Quiver

    u_var_idx = VAR_INDEX_15_LEVELS['U0']
    v_var_idx = VAR_INDEX_15_LEVELS['V0']


    # set longitude and latitude
    longitude = np.linspace(-180,180,1441)[0:-1]  # 地图投影, 以180°为中心
    latitude = np.linspace(-90,90,721)[0:-1]
    box_range = [-60, 10, 0, 60]
    lon_indices = np.where((longitude >= box_range[0]) & (longitude <= box_range[1]))[0]
    lat_indices = np.where((latitude  >= box_range[2]) & (latitude  <= box_range[3]))[0]

    longitude_kuroshio = longitude[lon_indices[0]:lon_indices[-1]]
    latitude_kuroshio  = latitude[lat_indices[0]:lat_indices[-1]]
    longitude2d_kuroshio, latitude2d_kuroshio = np.meshgrid(longitude_kuroshio, latitude_kuroshio)

    # read data
    tfid = nc.Dataset(file_path)
    if pre_or_truth == 'pre':
        data_u = tfid.variables['predicted'][init_condition, time_step, u_var_idx, :, :] 
        data_v = tfid.variables['predicted'][init_condition, time_step, v_var_idx, :, :] 
    elif pre_or_truth == 'truth':
        data_u = tfid.variables['ground_truth'][init_condition, time_step, u_var_idx, :, :] 
        data_v = tfid.variables['ground_truth'][init_condition, time_step, v_var_idx, :, :] 
    data_u = np.flipud(data_u)
    data_v = np.flipud(data_v)

    # read means and stds, inverse transform data
    means = np.load(LEVELS_15_STATS['global_means_path'])
    stds  = np.load(LEVELS_15_STATS['global_stds_path'])
    means_u = means[0, u_var_idx, 0, 0]
    means_v = means[0, v_var_idx, 0, 0]
    stds_u  = stds[0, u_var_idx, 0, 0]
    stds_v  = stds[0, v_var_idx, 0, 0]
    data_u = data_u * stds_u + means_u
    data_v = data_v * stds_v + means_v

    data_u_kuroshio = data_u[lat_indices[0]:lat_indices[-1],lon_indices[0]:lon_indices[-1]]
    data_v_kuroshio = data_v[lat_indices[0]:lat_indices[-1],lon_indices[0]:lon_indices[-1]]
    uv_kuroshio = np.sqrt(data_u_kuroshio**2 + data_v_kuroshio**2)

    # Custom Colormap
    mycolor = ['#FFFBCE','#EDDF96','#D9C55D','#B5B049','#8DA229','#5B9214','#3C8221','#15672E','#19522C','#1D3822']#浅黄-绿
    cmap_color = mpl.colors.LinearSegmentedColormap.from_list('my_list', mycolor)
    cmap = cmap_color
    
    # begin plot 
    fig = plt.figure(figsize=(12,10))
    proj = ccrs.PlateCarree(central_longitude = 180)  # central_longitude设置地图中心经度
    ax = fig.add_axes([0, 1, 0.6, 0.3], projection = proj) # 设置图片位置和大小（左下宽高）

    # 以下为经纬度设置
    ax.set_extent([135, 175.000001, 10, 50.000001], crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, linestyle=':', color='k', alpha=0.8)###绘制经纬线###
    xticks = np.arange(135,175.1, 10)
    yticks = np.arange(10, 50.1, 10)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree()) ###绘制经纬度标记###
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    land = cfeature.NaturalEarthFeature('physical', 'land', scale='110m', 
                                        edgecolor='black', facecolor='white',
                                        lw=0.5,zorder=2) #覆盖陆地
    ax.add_feature(land)
    
    step = 16
    cf = ax.contourf(longitude2d_kuroshio, latitude2d_kuroshio, uv_kuroshio, 
                     levels=np.arange(0.1,1.6,0.1), cmap=cmap, extend='both', zorder=0)

    s = ax.streamplot(longitude_kuroshio[::step], latitude_kuroshio[::step], 
                      data_u_kuroshio[::step, ::step], data_v_kuroshio[::step, ::step],
                      density=0.8, color='midnightblue', linewidth=1, transform=proj)
    
    
    # colorbar
    rect = [0.154,0.92,0.29,0.02]
    cbar_ax = fig.add_axes(rect)
    c=plt.colorbar(cf, cax=cbar_ax,extend="both",orientation='horizontal', aspect=20, pad=0.1)
    c.set_ticks(np.arange(0.0,1.50001,0.2)) #自定义数字间隔
    c.ax.tick_params(which='major', direction='in', labelsize=10, length=15) 

    fig.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close()


def plot_kuroshio_stream_between_pred_truth(
        file_path:str,           # 模型预报文件路径
        save_path:str,           # 图像保存路径
        init_condition:int,      # 第几次起报
        time_step:int,           # 第几个预报时间步
):
    # Kuroshio Stream Contourf & Quiver

    u_var_idx = VAR_INDEX_15_LEVELS['U0']
    v_var_idx = VAR_INDEX_15_LEVELS['V0']


    # set longitude and latitude
    longitude = np.linspace(-180,180,1441)[0:-1]  # 地图投影, 以180°为中心
    latitude = np.linspace(-90,90,721)[0:-1]
    box_range = [-60, 10, 0, 60]
    lon_indices = np.where((longitude >= box_range[0]) & (longitude <= box_range[1]))[0]
    lat_indices = np.where((latitude  >= box_range[2]) & (latitude  <= box_range[3]))[0]

    longitude_kuroshio = longitude[lon_indices[0]:lon_indices[-1]]
    latitude_kuroshio  = latitude[lat_indices[0]:lat_indices[-1]]
    longitude2d_kuroshio, latitude2d_kuroshio = np.meshgrid(longitude_kuroshio, latitude_kuroshio)

    # read data
    tfid = nc.Dataset(file_path)
    pred_data_u  = tfid.variables['predicted'][init_condition, time_step, u_var_idx, :, :] 
    pred_data_v  = tfid.variables['predicted'][init_condition, time_step, v_var_idx, :, :] 
    truth_data_u = tfid.variables['ground_truth'][init_condition, time_step, u_var_idx, :, :] 
    truth_data_v = tfid.variables['ground_truth'][init_condition, time_step, v_var_idx, :, :] 

    # read means and stds, inverse transform data
    means = np.load(LEVELS_15_STATS['global_means_path'])
    stds  = np.load(LEVELS_15_STATS['global_stds_path'])
    means_u = means[0, u_var_idx, 0, 0]
    means_v = means[0, v_var_idx, 0, 0]
    stds_u  = stds[0, u_var_idx, 0, 0]
    stds_v  = stds[0, v_var_idx, 0, 0]

    pred_data_u  = pred_data_u * stds_u + means_u
    pred_data_v  = pred_data_v * stds_v + means_v
    truth_data_u = truth_data_u * stds_u + means_u
    truth_data_v = truth_data_v * stds_v + means_v


    pred_data_u_kuroshio = pred_data_u[lat_indices[0]:lat_indices[-1],lon_indices[0]:lon_indices[-1]]
    pred_data_v_kuroshio = pred_data_v[lat_indices[0]:lat_indices[-1],lon_indices[0]:lon_indices[-1]]
    pred_uv_kuroshio     = np.sqrt(pred_data_u_kuroshio**2 + pred_data_v_kuroshio**2)

    truth_data_u_kuroshio = truth_data_u[lat_indices[0]:lat_indices[-1],lon_indices[0]:lon_indices[-1]]
    truth_data_v_kuroshio = truth_data_v[lat_indices[0]:lat_indices[-1],lon_indices[0]:lon_indices[-1]]
    truth_uv_kuroshio     = np.sqrt(truth_data_u_kuroshio**2 + truth_data_v_kuroshio**2)

    uv_kuroshio = pred_uv_kuroshio - truth_uv_kuroshio
    uv_kuroshio = np.flipud(uv_kuroshio)

    # Custom Colormap
    # mycolor = ['#FFFBCE','#EDDF96','#D9C55D','#B5B049','#8DA229','#5B9214','#3C8221','#15672E','#19522C','#1D3822']#浅黄-绿
    # cmap_color = mpl.colors.LinearSegmentedColormap.from_list('my_list', mycolor)
    # cmap = cmap_color
    cmap = plt.get_cmap('RdBu')
    
    # begin plot 
    fig = plt.figure(figsize=(12,10))
    proj = ccrs.PlateCarree(central_longitude = 180)  # central_longitude设置地图中心经度
    ax = fig.add_axes([0, 1, 0.6, 0.3], projection = proj) # 设置图片位置和大小（左下宽高）

    # 以下为经纬度设置
    ax.set_extent([135, 175.000001, 10, 50.000001], crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, linestyle=':', color='k', alpha=0.8)###绘制经纬线###
    xticks = np.arange(135,175.1, 10)
    yticks = np.arange(10, 50.1, 10)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree()) ###绘制经纬度标记###
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    land = cfeature.NaturalEarthFeature('physical', 'land', scale='110m', 
                                        edgecolor='black', facecolor='white',
                                        lw=0.5, zorder=2) #覆盖陆地
    ax.add_feature(land)
    
    vmin = -0.5
    vmax = 0.5
    v_interval = 0.1
    step = 16
    cf = ax.contourf(longitude2d_kuroshio, latitude2d_kuroshio, uv_kuroshio, 
                     levels=np.arange(vmin, vmax+v_interval, v_interval), cmap=cmap, extend='both', zorder=0)
    
    
    # colorbar
    rect = [0.154,0.92,0.29,0.02]
    cbar_ax = fig.add_axes(rect)
    c=plt.colorbar(cf, cax=cbar_ax,extend="both",orientation='horizontal', aspect=20, pad=0.1)
    c.set_ticks(np.arange(vmin, vmax+v_interval, v_interval)) #自定义数字间隔
    c.ax.tick_params(which='major', direction='in', labelsize=10, length=15) 

    fig.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close()


if __name__ == '__main__':

    file_dir = '/home/bingxing2/home/scx6115/Ocean_AI_model/exp_15_levels/Masked_AE_Ocean/20230628-131150/2_steps_finetune'
    file_path = os.path.join(file_dir, 'autoregressive_predictions.h5') # 模型预报文件路径
    land_mask_file_path = LEVELS_15_STATS['land_mask_file'] # 海陆掩码文件路径
    projection = 'PlateCarree' # 投影方式，可选："robinson"，"PlateCarree"
    pre_or_truth = 'pre'    # 可视化预报还是真值
    init_condition = 1      # 第几次起报
    interval  = 1           # downscale, 经纬度间隔, 默认为1
    time_step = 5           # 第几个预报时间步


    for init_condition in [1, 3, 5, 7, 9]:
        for time_step in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]:
            # plot difference between the truth and prediction
            # save_path = os.path.join(file_dir, f'global_steam_{projection}_init_{init_condition}_timestep_{time_step}_pre_truth_diff.png') 
            # plot_global_stream_diff_between_pred_truth(file_path           = file_path,
            #                                            projection          = projection,
            #                                            init_condition      = init_condition,
            #                                            interval            = interval,
            #                                            time_step           = time_step,
            #                                            save_path           = save_path)

            # save_path = os.path.join(file_dir, f'kuroshio_steam_init_{init_condition}_timestep_{time_step}_pre_truth_diff.png')
            # plot_kuroshio_stream_between_pred_truth(file_path      = file_path,
            #                                         init_condition = init_condition,
            #                                         time_step      = time_step,
            #                                         save_path      = save_path)

            for pre_or_truth in ['pre', 'truth']:

                save_path = os.path.join(file_dir, f'global_steam_{projection}_init_{init_condition}_timestep_{time_step}_{pre_or_truth}.png')  # 图像保存路径
                plot_global_stream(file_path           = file_path,
                                   projection          = projection,
                                   pre_or_truth        = pre_or_truth,
                                   land_mask_file_path = land_mask_file_path,
                                   init_condition      = init_condition,
                                   interval            = interval,
                                   time_step           = time_step,
                                   save_path           = save_path)

                save_path = os.path.join(file_dir, f'kuroshio_steam_init_{init_condition}_timestep_{time_step}_{pre_or_truth}.png')  # 图像保存路径
                plot_kuroshio_stream(file_path      = file_path,
                                     pre_or_truth   = pre_or_truth,
                                     init_condition = init_condition,
                                     time_step      = time_step,
                                     save_path      = save_path)
