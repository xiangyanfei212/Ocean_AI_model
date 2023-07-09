import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from icecream import ic

global_means_path = "/home/bingxing2/home/scx6115/Ocean_AI_model/sample_01/train/global_means.npy"
global_stds_path =  "/home/bingxing2/home/scx6115/Ocean_AI_model/sample_01/train/global_stds.npy"
land_mask_file = "/home/bingxing2/home/scx6115/Ocean_AI_model/sample_01/land_mask.h5"
VAR_INDEX_15_levels = {
        "T0":   0,   "T6":   1,   "T10":  2,   "T20":  3,   "T30":  4, 
        "T50":  5,   "T70":  6,   "T100": 7,   "T125": 8,   "T150": 9, 
        "T200": 10,  "T250": 11,  "T300": 12,  "T400": 13,  "T500": 14,
        "S0":   15,  "S6":   16,  "S10":  17,  "S20":  18,  "S30":  19, 
        "S50":  20,  "S70":  21,  "S100": 22,  "S125": 23,  "S150": 24, 
        "S200": 25,  "S250": 26,  "S300": 27,  "S400": 28,  "S500": 29,
        "U0":   30,  "U6":   31,  "U10":  32,  "U20":  33,  "U30":  34, 
        "U50":  35,  "U70":  36,  "U100": 37,  "U125": 38,  "U150": 39, 
        "U200": 40,  "U250": 41,  "U300": 42,  "U400": 43,  "U500": 44,
        "V0":   45,  "V6":   46,  "V10":  47,  "V20":  48,  "V30":  49, 
        "V50":  50,  "V70":  51,  "V100": 52,  "V125": 53,  "V150": 54, 
        "V200": 55,  "V250": 56,  "V300": 57,  "V400": 58,  "V500": 59,
        "SSH": 60,
} 

VAR_INDEX_15_levels = {'T': np.arange(15), 
                       'S': np.arange(15, 30),
                       'U': np.arange(30, 45),
                       'V': np.arange(45, 60)}

def read_pre_tru_from_h5(file_path, var_name, time_step):
    """
    var_name: T, S, U, V
    """

    var_idx = VAR_INDEX_15_levels[var_name]

    means = np.load(global_means_path)
    stds = np.load(global_stds_path)

    means = means[0, var_idx, 0, 0]
    stds = stds[0, var_idx, 0, 0]
    ic(means.shape, stds.shape)
    means = np.expand_dims(means, axis=[0,2,3])
    stds = np.expand_dims(stds, axis=[0,2,3])
    ic(means.shape, stds.shape)

    tfid = nc.Dataset(file_path)
    pre = tfid.variables['predicted'][0, time_step, var_idx, :, :] 
    tru = tfid.variables['ground_truth'][0, time_step, var_idx, :, :]  # truth data

    pre = pre * stds + means
    tru = tru * stds + means

    return pre, tru



def profilePlot(in_file, time_step, var_name, pro_type, pro_loc, save_path):

    lon = np.arange(0, 360, 0.25)    # lontitude
    lat = np.arange(90, -90, -0.25)  # latitude
    depth = [0, 6, 10, 20, 30, 50, 70, 100, 125, 150, 200, 250, 300, 400, 500]

    pre, tru = read_pre_tru_from_h5(in_file, var_name, time_step)
    ic(pre.shape, tru.shape)

    fontsize=10

    fig = plt.figure(figsize=(8,3), dpi=300, constrained_layout=True) # init a figure
    plt.rcParams['axes.facecolor']='lightgrey' # set the background color to lightgrey

    ax = fig.add_axes([0,0,1,1])  # add sub figure
    ax.invert_yaxis() # invert the y axis
    # ax.set_yticks([1000,925,850,700,500,300]) 


    if pro_type == 'lon':
        x = lon

        lat_idx = lat.tolist().index(pro_loc)

        pre = pre[:,:,lat_idx]
        tru = tru[:,:,lat_idx]
        print('lon profile:', pre.shape, tru.shape)

        xlabel = 'Longitude'
        ax.set_xticks(np.arange(0, 360, 30))
        ax.set_xticklabels([r'180$^\degree$', r'150$^\degree$W',r'120$^\degree$W',
                            r'90$^\degree$W', r'60$^\degree$W',r'30$^\degree$W',
                            r'0$^\degree$',   r'30$^\degree$E',r'60$^\degree$E',
                            r'90$^\degree$E', r'120$^\degree$E',r'150$^\degree$E'],
                            fontsize=fontsize)

    if pro_type == 'lat':
        x = lat 
        lon_idx = lon.tolist().index(pro_loc)

        pre = pre[:,:,:,lon_idx]
        tru = tru[:,:,:,lon_idx]
        print('lat profile:', pre.shape, tru.shape)

        xlabel = 'Latitude'
        ax.set_xticks(np.arange(-90, 91, 30))
        ax.set_xticklabels([r'90$^\degree$S', r'60$^\degree$S', r'30$^\degree$S',
                            r'0$^\degree$', r'30$^\degree$N', r'60$^\degree$N', r'90$^\degree$N'],
                            fontsize=fontsize)


    ax.set_xlabel(xlabel, fontsize = fontsize)
    ax.set_ylabel('Depth (m)', fontsize = fontsize)
    ax.tick_params(axis='both', which='both', labelsize=fontsize)

    apartNum = 5 
    # levels = np.arange(0, 30, 2)
    ic(x.shape)
    ac=ax.contourf(x, depth, np.squeeze(tru), apartNum, cmap='jet', extend='both', alpha=0.75)
    # 画出n条线，并将颜色设置为黑色

    contour = plt.contour(x, depth, np.squeeze(tru), apartNum, colors='k',linewidths=0.5)

    # 等高线上标明温度的值，颜色是黑色
    plt.clabel(contour, fmt = '%.0f', fontsize=fontsize, colors='k')   
    cb=fig.colorbar(ac, extend='both', shrink=1, label='Temperature', pad=0.01)
    cb.ax.tick_params(axis='both', which='both', length=1, labelsize=fontsize)


    title = 'test'
    ax.set_title(title, fontsize=fontsize+3)

    print('Saving...')
    plt.savefig('tmp.jpg', bbox_inches = 'tight')



if __name__ == '__main__':

    model_name = 'afno'
    in_dir = '../exp_15_levels/afno/20230628-131420'
    profilePlot(in_file = os.path.join(in_dir, 'autoregressive_predictions.h5'), 
                time_step = 5,
                var_name = 'T',
                pro_type = 'lon', 
                pro_loc = 30, 
                save_path = '')





