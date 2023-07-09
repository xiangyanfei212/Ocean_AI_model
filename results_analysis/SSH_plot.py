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

def plot_global_ssh(
        pre_or_truth:str,        # 可视化预测还是真值，可选："pre"、"truth"
        projection:str,          # 投影方式，可选：'PlateCarree', 'robinson'
        file_path:str,           # 模型预报文件路径
        land_mask_file_path:str, # 海陆掩码文件路径
        interval:int,            # 经纬度间隔, 默认为1
        save_path:str,           # 图像保存路径
        init_condition:int,      # 第几次起报
        time_step:int,           # 第几个预报时间步
    ):

    var_idx = VAR_INDEX_15_LEVELS['SSH']

    # read means and stds, for inverse transform data
    means = np.load(LEVELS_15_STATS['global_means_path'])
    stds  = np.load(LEVELS_15_STATS['global_stds_path'])
    means = means[0, var_idx, 0, 0]
    stds  = stds[0, var_idx, 0, 0]

    # read u, v
    tfid = nc.Dataset(file_path)
    if pre_or_truth == 'pre':
        data = tfid.variables['predicted'][init_condition, time_step, var_idx, :, :] 
    elif pre_or_truth == 'truth':
        data = tfid.variables['ground_truth'][init_condition, time_step, var_idx, :, :] 
    data = data * stds + means
    data = data[::interval,::interval]
    data = np.flipud(data)

    # set longitude and latitude
    longitude = np.linspace(-180,180,1441)[0:-1]  # 地图投影, 以180°为中心
    longitude = longitude[::interval]

    latitude = np.linspace(-90,90,721)[0:-1]
    latitude = latitude[::interval]
    longitude2d, latitude2d = np.meshgrid(longitude, latitude)
    

    if projection == 'PlateCarree':
        fig = plt.figure(figsize=(12,10))
        proj = ccrs.PlateCarree(central_longitude = 180) #central_longitude设置地图中心经度
        ax = fig.add_axes([0, 1, 0.6, 0.3],projection = proj) #设置图片位置和大小（左下宽高）
        ax.set_extent([0, 360.000001, -90, 90.000001], crs=ccrs.PlateCarree())#以下为经纬度设置
        
        xticks = np.arange(0, 360.1, 60)
        yticks = np.arange(-90, 90.1, 30)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, linestyle=':', color='k', alpha=0.8)###绘制经纬线###
        ax.set_xticks(xticks, crs=ccrs.PlateCarree()) ###绘制经纬度标记###
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelsize=10)
        
        # land
        land = cfeature.NaturalEarthFeature('physical', 'land', scale='110m', edgecolor='black', facecolor='white',lw=0.5,zorder=2)#覆盖陆地
        ax.add_feature(land)

        # color
        # selfcolor = ['#F8FF01','#5FCA2A','#14861C','#0A6E6E','#2D1EB9','#43056C']
        # cmap1 = mcolors.ListedColormap(selfcolor)
        cmap = plt.get_cmap('GnBu')

        # cf = ax.contourf(longitude2d, latitude2d, data, levels=np.arange(-1,1.21,0.2),cmap=cmap,extend='both',zorder=0)
        cf = ax.pcolormesh(longitude2d, latitude2d, data, cmap=cmap, 
                           vmin=-1, vmax=1.21, transform=ccrs.PlateCarree(central_longitude=180))

        rect = [0.05,0.91,0.5,0.02]
        cbar_ax = fig.add_axes(rect)    
        c = plt.colorbar(cf, cax=cbar_ax,extend="both",orientation='horizontal', aspect=20, pad=0.1)
        c.set_ticks(np.arange(-1,1.21,0.2)) #自定义数字间隔
        c.ax.tick_params(which='major',direction='in',labelsize=10,length=15) #色卡设置
        #c.set_label('SSH',loc="center",fontsize=12,rotation=360)

        fig.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        

    elif projection == 'robinson':

        fig = plt.figure(figsize=(12,10))

        # Use Robinson projection
        proj = ccrs.Robinson(central_longitude=180)
        
        ax = fig.add_axes([0, 1, 0.6, 0.3], projection=proj)
        ax.set_global()
        
        land = cfeature.NaturalEarthFeature('physical', 'land', scale='110m', edgecolor='black', facecolor='white',lw=0.5,zorder=2)
        ax.add_feature(land)
        
        # The range of Colorbar
        # selfcolor = ['#F8FF01','#5FCA2A','#14861C','#0A6E6E','#2D1EB9','#43056C']
        # cmap1 = mcolors.ListedColormap(selfcolor)
        cmap = plt.get_cmap('GnBu')
        
        # Use pcolormesh() function to fill grid boxes with colors
        cf = ax.pcolormesh(longitude2d, latitude2d, data, cmap=cmap, vmin=-1, vmax=1.21, transform=ccrs.PlateCarree(central_longitude=180))
        
        rect = [0.05,0.91,0.5,0.02]
        cbar_ax = fig.add_axes(rect)
        c=plt.colorbar(cf, cax=cbar_ax,extend="both",orientation='horizontal', aspect=20, pad=0.1)
        c.set_ticks(np.arange(-1,1.21,0.2)) #自定义数字间隔
        c.ax.tick_params(which='major',direction='in',labelsize=10,length=15) #色卡设置
        #c.set_label('SSH',loc="center",fontsize=12,rotation=360)

        fig.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()

def plot_global_ssh_diff_between_pred_truth(
        projection:str,          # 投影方式，可选：'PlateCarree', 'robinson'
        file_path:str,           # 模型预报文件路径
        interval:int,            # 经纬度间隔, 默认为1
        save_path:str,           # 图像保存路径
        init_condition:int,      # 第几次起报
        time_step:int,           # 第几个预报时间步
    ):

    var_idx = VAR_INDEX_15_LEVELS['SSH']

    # read means and stds, for inverse transform data
    means = np.load(LEVELS_15_STATS['global_means_path'])
    stds  = np.load(LEVELS_15_STATS['global_stds_path'])
    means = means[0, var_idx, 0, 0]
    stds  = stds[0, var_idx, 0, 0]

    # read u, v
    tfid = nc.Dataset(file_path)
    pred_data  = tfid.variables['predicted'][init_condition, time_step, var_idx, :, :] 
    truth_data = tfid.variables['ground_truth'][init_condition, time_step, var_idx, :, :] 
    pred_data  = pred_data * stds + means
    truth_data = truth_data * stds + means

    # diff
    data = pred_data - truth_data

    data = data[::interval,::interval]
    data = np.flipud(data)

    # set longitude and latitude
    longitude = np.linspace(-180,180,1441)[0:-1]  # 地图投影, 以180°为中心
    longitude = longitude[::interval]

    latitude = np.linspace(-90,90,721)[0:-1]
    latitude = latitude[::interval]
    longitude2d, latitude2d = np.meshgrid(longitude, latitude)
    
    vmin = -0.5
    vmax = 0.5
    v_interval = 0.1

    if projection == 'PlateCarree':
        fig = plt.figure(figsize=(12,10))
        proj = ccrs.PlateCarree(central_longitude = 180) #central_longitude设置地图中心经度
        ax = fig.add_axes([0, 1, 0.6, 0.3],projection = proj) #设置图片位置和大小（左下宽高）
        ax.set_extent([0, 360.000001, -90, 90.000001], crs=ccrs.PlateCarree())#以下为经纬度设置
        
        xticks = np.arange(0, 360.1, 60)
        yticks = np.arange(-90, 90.1, 30)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, linestyle=':', color='k', alpha=0.8)###绘制经纬线###
        ax.set_xticks(xticks, crs=ccrs.PlateCarree()) ###绘制经纬度标记###
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelsize=10)
        
        # land
        land = cfeature.NaturalEarthFeature('physical', 'land', scale='110m', edgecolor='black', facecolor='white',lw=0.5,zorder=2)#覆盖陆地
        ax.add_feature(land)

        # color
        cmap = plt.get_cmap('RdBu')
        cf = ax.pcolormesh(longitude2d, latitude2d, data, cmap=cmap, 
                           vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(central_longitude=180))

        rect = [0.05,0.91,0.5,0.02]
        cbar_ax = fig.add_axes(rect)    
        c = plt.colorbar(cf, cax=cbar_ax,extend="both",orientation='horizontal', aspect=20, pad=0.1)
        c.set_ticks(np.arange(vmin, vmax+v_interval, v_interval))  # 自定义数字间隔
        c.ax.tick_params(which='major', direction='in', labelsize=10, length=15) #色卡设置

        fig.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        return
        

    elif projection == 'robinson':

        fig = plt.figure(figsize=(12,10))

        # Use Robinson projection
        proj = ccrs.Robinson(central_longitude=180)
        
        ax = fig.add_axes([0, 1, 0.6, 0.3], projection=proj)
        ax.set_global()
        
        land = cfeature.NaturalEarthFeature('physical', 'land', scale='110m', edgecolor='black', facecolor='white',lw=0.5,zorder=2)
        ax.add_feature(land)
        
        # The range of Colorbar
        cmap = plt.get_cmap('RdBu')
        
        # Use pcolormesh() function to fill grid boxes with colors
        cf = ax.pcolormesh(longitude2d, latitude2d, data, cmap=cmap, 
                            vmin=vmin, vmax=vmax, 
                            transform=ccrs.PlateCarree(central_longitude=180))
        
        rect = [0.05,0.91,0.5,0.02]
        cbar_ax = fig.add_axes(rect)
        c=plt.colorbar(cf, cax=cbar_ax, extend="both", orientation='horizontal', aspect=20, pad=0.1)
        c.set_ticks(np.arange(vmin, vmax, 0.2)) #自定义数字间隔
        c.ax.tick_params(which='major', direction='in', labelsize=10, length=15) #色卡设置
        #c.set_label('SSH',loc="center",fontsize=12,rotation=360)

        fig.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        return


def plot_kuroshio_ssh(
        pre_or_truth:str,        # 可视化预测还是真值，可选："pre"、"truth"
        file_path:str,           # 模型预报文件路径
        save_path:str,           # 图像保存路径
        init_condition:int,      # 第几次起报
        time_step:int,           # 第几个预报时间步
):
    # Kuroshio Stream Contourf & Quiver

    var_idx = VAR_INDEX_15_LEVELS['SSH']

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
        data = tfid.variables['predicted'][init_condition, time_step, var_idx, :, :] 
    elif pre_or_truth == 'truth':
        data = tfid.variables['ground_truth'][init_condition, time_step, var_idx, :, :] 
    data = np.flipud(data)

    # read means and stds, inverse transform data
    means = np.load(LEVELS_15_STATS['global_means_path'])
    stds  = np.load(LEVELS_15_STATS['global_stds_path'])
    means = means[0, var_idx, 0, 0]
    stds  = stds[0, var_idx, 0, 0]
    data = data * stds + means

    data_kuroshio = data[lat_indices[0]:lat_indices[-1],lon_indices[0]:lon_indices[-1]]

    fig = plt.figure(figsize=(12,10))
    proj = ccrs.PlateCarree(central_longitude=180)#central_longitude设置地图中心经度
    ax = fig.add_axes([0, 1, 0.6, 0.3],projection = proj)#设置图片位置和大小（左下宽高）
    
    ax.set_extent([135, 175.000001, 10, 50.000001], crs=ccrs.PlateCarree())#以下为经纬度设置
    xticks = np.arange(135,175.1, 10)
    yticks = np.arange(10, 50.1, 10)
    #gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, linestyle=':', color='k', alpha=0.8)###绘制经纬线###
    ax.set_xticks(xticks, crs=ccrs.PlateCarree()) ###绘制经纬度标记###
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    lon_formatter = LongitudeFormatter(number_format='.0f', degree_symbol=r'$^o$')
    lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol=r'$^o$')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=10)
    
    land = cfeature.NaturalEarthFeature('physical', 'land', scale='110m', edgecolor='black', facecolor='white',lw=0.5,zorder=2)#覆盖陆地
    ax.add_feature(land)

    cmap = plt.get_cmap('GnBu')
    cf = ax.contourf(longitude2d_kuroshio, latitude2d_kuroshio, data_kuroshio, 
                        levels=np.arange(-0.5,1.5,0.35), cmap=cmap, extend='both', zorder=0)

    selfcolor = ['#F8FF01','#5FCA2A','#14861C','#0A6E6E','#2D1EB9','#43056C']
    cmap1 = mcolors.ListedColormap(selfcolor)
    cf1 = ax.contour(longitude2d_kuroshio, latitude2d_kuroshio, data_kuroshio, 
                        levels=np.arange(-0.5,1.5,0.35), cmap=cmap1, extend='both', zorder=1)
    ml = ax.clabel(cf1, inline=True, fmt='%.2f', 
                   manual=[(-37,45),(-30,40),(-36,35),(-23,30),(-32,20),(-9,35)], 
                   use_clabeltext=True,fontsize = 10) #等值线数值
    for m in ml:
        #m.set_bbox({'fc': 'white','color':'white'})#给每根线加上框 #边框白色
        m.set_bbox({'fc': 'white'})#边框黑色
    
    rect = [0.154, 0.91, 0.29, 0.02]
    cbar_ax = fig.add_axes(rect)
    c = plt.colorbar(cf, cax=cbar_ax, extend="both", orientation='horizontal', aspect=20, pad=0.1)
    c.set_ticks(np.arange(-0.5, 1.5, 0.35)) #自定义数字间隔
    c.ax.tick_params(which='major', direction='in', labelsize=10, length=15) #色卡设置
    #c.set_label('SSH',loc="center",fontsize=12,rotation=360

    fig.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close()
    return


def plot_kuroshio_ssh_diff_between_pred_truth(file_path, 
                                              init_condition,
                                              time_step,
                                              save_path):

    var_idx = VAR_INDEX_15_LEVELS['SSH']

    # set longitude and latitude
    longitude = np.linspace(-180,180,1441)[0:-1]  # 地图投影, 以180°为中心
    latitude = np.linspace(-90,90,721)[0:-1]
    box_range = [-60, 10, 0, 60]
    lon_indices = np.where((longitude >= box_range[0]) & (longitude <= box_range[1]))[0]
    lat_indices = np.where((latitude  >= box_range[2]) & (latitude  <= box_range[3]))[0]

    longitude_kuroshio = longitude[lon_indices[0]:lon_indices[-1]]
    latitude_kuroshio  = latitude[lat_indices[0]:lat_indices[-1]]
    longitude2d_kuroshio, latitude2d_kuroshio = np.meshgrid(longitude_kuroshio, latitude_kuroshio)

    # read means and stds, inverse transform data
    means = np.load(LEVELS_15_STATS['global_means_path'])
    stds  = np.load(LEVELS_15_STATS['global_stds_path'])
    means = means[0, var_idx, 0, 0]
    stds  = stds[0, var_idx, 0, 0]

    # read data
    tfid = nc.Dataset(file_path)
    pred_data  = tfid.variables['predicted'][init_condition, time_step, var_idx, :, :] 
    truth_data = tfid.variables['ground_truth'][init_condition, time_step, var_idx, :, :] 
    pred_data  = pred_data * stds + means
    truth_data = truth_data * stds + means
    data = pred_data - truth_data
    data = np.flipud(data)

    data_kuroshio = data[lat_indices[0]:lat_indices[-1],lon_indices[0]:lon_indices[-1]]

    fig = plt.figure(figsize=(12,10))
    proj = ccrs.PlateCarree(central_longitude=180)#central_longitude设置地图中心经度
    ax = fig.add_axes([0, 1, 0.6, 0.3],projection = proj)#设置图片位置和大小（左下宽高）
    
    ax.set_extent([135, 175.000001, 10, 50.000001], crs=ccrs.PlateCarree())#以下为经纬度设置
    xticks = np.arange(135,175.1, 10)
    yticks = np.arange(10, 50.1, 10)
    #gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, linestyle=':', color='k', alpha=0.8)###绘制经纬线###
    ax.set_xticks(xticks, crs=ccrs.PlateCarree()) ###绘制经纬度标记###
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    lon_formatter = LongitudeFormatter(number_format='.0f', degree_symbol=r'$^o$')
    lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol=r'$^o$')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=10)
    
    land = cfeature.NaturalEarthFeature('physical', 'land', scale='110m', edgecolor='black', facecolor='white',lw=0.5,zorder=2)#覆盖陆地
    ax.add_feature(land)

    cmap = plt.get_cmap('RdBu')
    vmin = -0.5
    vmax = 0.5
    v_interval = 0.1

    cf = ax.contourf(longitude2d_kuroshio, latitude2d_kuroshio, data_kuroshio, 
                     levels=np.arange(vmin, vmax+v_interval, v_interval), 
                     cmap=cmap, extend='both', zorder=0)


    rect = [0.154, 0.91, 0.29, 0.02]
    cbar_ax = fig.add_axes(rect)
    c = plt.colorbar(cf, cax=cbar_ax, extend="both", orientation='horizontal', aspect=20, pad=0.1)
    c.set_ticks(np.arange(vmin, vmax+0.1, 0.1)) #自定义数字间隔
    c.ax.tick_params(which='major', direction='in', labelsize=10, length=15) #色卡设置
    #c.set_label('SSH',loc="center",fontsize=12,rotation=360

    fig.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close()
    return

if __name__ == '__main__':

    file_dir = '/home/bingxing2/home/scx6115/Ocean_AI_model/exp_15_levels/Masked_AE_Ocean/20230628-131150/2_steps_finetune'
    file_path = os.path.join(file_dir, 'autoregressive_predictions.h5') # 模型预报文件路径
    land_mask_file_path = LEVELS_15_STATS['land_mask_file'] # 海陆掩码文件路径
    projection = 'PlateCarree' # 投影方式，可选："robinson"，"PlateCarree"
    pre_or_truth = 'truth'    # 可视化预报还是真值
    init_condition = 1      # 第几次起报
    time_step = 5
    interval  = 1           # downscale, 经纬度间隔, 默认为1

    for init_condition in [1, 3, 5, 7, 9]:
        for time_step in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]:
            # plot the difference between truth and prediction
            save_path = os.path.join(file_dir, f'global_ssh_{projection}_init_{init_condition}_timestep_{time_step}_pre_truth_diff.png') 
            plot_global_ssh_diff_between_pred_truth( file_path       = file_path,
                                                     projection      = projection,
                                                     init_condition  = init_condition,
                                                     interval        = interval,
                                                     time_step       = time_step,
                                                     save_path       = save_path)

            save_path = os.path.join(file_dir, f'kuroshio_ssh_init_{init_condition}_timestep_{time_step}_pre_truth_diff.png') 
            plot_kuroshio_ssh_diff_between_pred_truth( file_path      = file_path,
                                                       init_condition = init_condition,
                                                       time_step      = time_step,
                                                       save_path      = save_path)

            for pre_or_truth in ['pre', 'truth']:
                save_path = os.path.join(file_dir, f'global_ssh_{projection}_init_{init_condition}_timestep_{time_step}_{pre_or_truth}.png') 
                plot_global_ssh(file_path           = file_path,
                                projection          = projection,
                                pre_or_truth        = pre_or_truth,
                                land_mask_file_path = land_mask_file_path,
                                init_condition      = init_condition,
                                interval            = interval,
                                time_step           = time_step,
                                save_path           = save_path)


                save_path = os.path.join(file_dir, f'kuroshio_ssh_init_{init_condition}_timestep_{time_step}_{pre_or_truth}.png') 
                plot_kuroshio_ssh(file_path      = file_path,
                                  pre_or_truth   = pre_or_truth,
                                  init_condition = init_condition,
                                  time_step      = time_step,
                                  save_path      = save_path)

