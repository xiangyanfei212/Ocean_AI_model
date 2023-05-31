import os
import sys
import h5py
import argparse
import numpy as np
import xarray as xr

from icecream import ic

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

HYCOM_MONTHLY_DIR = '/work/home/acrzcyisbk/HYCOM_monthly'
ERA5_MONTHLY_DIR = '/work/home/acrzcyisbk/ERA5_monthly'

OCEAN_FEATURES = ['water_temp', 'salinity', 'water_u', 'water_v', 'surf_el']
ATMOS_FEATURES = ['t2m', 'u10', 'v10', 'msl', 'sp']
LABEL = ['water_temp', 'salinity', 'water_u', 'water_v', 'surf_el']
# '10m_u_component_of_wind' 
# var='10m_v_component_of_wind'
# var='2m_temperature'
# var='mean_sea_level_pressure'
# var='sea_surface_temperature'
# var='surface_pressure']

def read_nc(var, year, month, depths=None, degree=1):
    if var == 'water_temp_salinity':
        file_path = os.path.join(HYCOM_MONTHLY_DIR, 'ts3z', f'degree_{degree}', f'GLBv0.08_53X_archMN.{year}_{str(month).zfill(2)}_ts3z.nc')
        mean_file_path = os.path.join(HYCOM_MONTHLY_DIR, 'ts3z', f'degree_{degree}', f'GLBv0.08_53X_archMN.{year}_01_{year}_12_ts3z.nc')

        ds = xr.open_dataset(file_path)
        if depths:
            # print(f'select depth: {depths} m')
            ds = ds.loc[dict(depth=depths)]
        t = ds['water_temp'].values
        s = ds['salinity'].values
        ds.close()

        ds = xr.open_dataset(mean_file_path)
        if depths:
            # print(f'select depth: {depths} m')
            ds = ds.loc[dict(depth=depths)]
        mean_t = ds['water_temp'].values
        mean_s = ds['salinity'].values
        ds.close()
        return t+mean_t, s+mean_s
    if var == 'water_u_v':
        file_path = os.path.join(HYCOM_MONTHLY_DIR, 'uv3z', f'degree_{degree}', f'GLBv0.08_53X_archMN.{year}_{str(month).zfill(2)}_uv3z.nc')
        mean_file_path = os.path.join(HYCOM_MONTHLY_DIR, 'uv3z', f'degree_{degree}', f'GLBv0.08_53X_archMN.{year}_01_{year}_12_uv3z.nc')

        ds = xr.open_dataset(file_path)
        if depths:
            # print(f'select depth: {depths} m')
            ds = ds.loc[dict(depth=depths)]
        u = ds['water_u'].values
        v = ds['water_v'].values
        ds.close()

        ds = xr.open_dataset(mean_file_path)
        if depths:
            # print(f'select depth: {depths} m')
            ds = ds.loc[dict(depth=depths)]
        mean_u = ds['water_u'].values
        mean_v = ds['water_v'].values
        ds.close()
        return u+mean_u,v+mean_v

    if var == 'ssh':
        file_path = os.path.join(HYCOM_MONTHLY_DIR, 'ssh', f'degree_{degree}', f'GLBv0.08_53X_archMN.{year}_{str(month).zfill(2)}_ssh.nc')
        mean_file_path = os.path.join(HYCOM_MONTHLY_DIR, 'ssh', f'degree_{degree}', f'GLBv0.08_53X_archMN.{year}_01_{year}_12_ssh.nc')

        ds = xr.open_dataset(file_path)
        ssh = ds['surf_el'].values
        ds.close()

        ds = xr.open_dataset(mean_file_path)
        mean_ssh = ds['surf_el'].values
        ds.close()
        return ssh+mean_ssh

    if var == 'u10':
        file_path = os.path.join(ERA5_MONTHLY_DIR, '10m_u_component_of_wind', f'degree_{degree}', f'10m_u_component_of_wind_{year}{str(month).zfill(2)}.nc') 
        ds = xr.open_dataset(file_path)
        u10 = ds['u10'].values
        return u10
    if var == 'v10':
        file_path = os.path.join(ERA5_MONTHLY_DIR, '10m_v_component_of_wind', f'degree_{degree}', f'10m_v_component_of_wind_{year}{str(month).zfill(2)}.nc') 
        ds = xr.open_dataset(file_path)
        u10 = ds['v10'].values
        return u10
    if var == 't2m':
        file_path = os.path.join(ERA5_MONTHLY_DIR, '2m_temperature', f'degree_{degree}', f'2m_temperature_{year}{str(month).zfill(2)}.nc') 
        ds = xr.open_dataset(file_path)
        t2m = ds['t2m'].values
        return t2m
    if var == 'msl':
        file_path = os.path.join(ERA5_MONTHLY_DIR, 'mean_sea_level_pressure', f'degree_{degree}', f'mean_sea_level_pressure_{year}{str(month).zfill(2)}.nc') 
        ds = xr.open_dataset(file_path)
        msl = ds['msl'].values
        return msl
    if var == 'sp':
        file_path = os.path.join(ERA5_MONTHLY_DIR, 'surface_pressure', f'degree_{degree}', f'surface_pressure_{year}{str(month).zfill(2)}.nc') 
        ds = xr.open_dataset(file_path)
        sp = ds['sp'].values
        return sp

def get_nan_indices(data):
    nan_indices = np.where(np.isnan(data))
    return nan_indices


def fill_land_with_zonal_depth_mean(data):

    land_indices = get_nan_indices(data)
    zonal_mean, depth_mean = get_zonal_depth_mean(data)
    
    # Fisrt, fill land(nan) with zonal_mean, but there are still NaN
    data[land_indices] = zonal_mean[land_indices]
    nan_indices = get_nan_indices(data)

    # Second, fill NaN with depth_mean
    data[nan_indices] = depth_mean[nan_indices]

    assert len(np.where(np.isnan(data))[0]) == 0

    return data

def get_zonal_depth_mean(data):
    """
    data: [Time, Depth, Lat, Lon] or [Time, Lat, Lon]
    """

    if data.ndim == 3:
        zonal_mean = np.nanmean(data, axis=1, keepdims=True)
        zonal_mean = np.repeat(zonal_mean, data.shape[1], axis=1)

        depth_mean = np.nanmean(data, axis=(0,1), keepdims=True)
        depth_mean = np.repeat(depth_mean, data.shape[0], axis=0)
        depth_mean = np.repeat(depth_mean, data.shape[1], axis=1)

    if data.ndim == 4:
        zonal_mean = np.nanmean(data, axis=2, keepdims=True)
        zonal_mean = np.repeat(zonal_mean, data.shape[2], axis=2)

        depth_mean = np.nanmean(data, axis=(1,2), keepdims=True)
        depth_mean = np.repeat(depth_mean, data.shape[1], axis=1)
        depth_mean = np.repeat(depth_mean, data.shape[2], axis=2)

    assert len(np.where(np.isnan(depth_mean))[0]) == 0
    assert data.shape == zonal_mean.shape
    assert data.shape == depth_mean.shape

    return zonal_mean, depth_mean

def gen_sample(
               # ocean_feature_vars, 
               # atmos_feature_vars, 
               # label_vars, 
               depths,
               degree, year, save_dir):
    for m in np.arange(1, 12):
        ic(m)

        t,s = read_nc('water_temp_salinity', year, m, depths, degree)
        u,v = read_nc('water_u_v', year, m, depths, degree)
        ssh = read_nc('ssh', year, m, degree)

        t = fill_land_with_zonal_depth_mean(t)
        s = fill_land_with_zonal_depth_mean(s)
        u = fill_land_with_zonal_depth_mean(u)
        v = fill_land_with_zonal_depth_mean(v)
        ssh = fill_land_with_zonal_depth_mean(ssh)
        ic(t.shape, ssh.shape)

        u10 = read_nc('u10', year, m, degree)
        v10 = read_nc('v10', year, m, degree)
        t2m = read_nc('t2m', year, m, degree)
        msl = read_nc('msl', year, m, degree)
        sp  = read_nc('sp',  year, m, degree)
        assert len(np.where(np.isnan(u10))[0]) == 0
        assert len(np.where(np.isnan(v10))[0]) == 0
        assert len(np.where(np.isnan(t2m))[0]) == 0
        assert len(np.where(np.isnan(msl))[0]) == 0
        assert len(np.where(np.isnan(sp))[0]) == 0

        ic(u.shape, u10.shape)
        t = np.squeeze(t)
        s = np.squeeze(s)
        u = np.squeeze(u)
        v = np.squeeze(v)
        data_monthly = np.concatenate([t, s, u, v, ssh, u10, v10, t2m, msl, sp],
                                       axis=0)
        # channel 0-24: t(depths=6),s(depths=6),u(depths=6),v(depths=6),ssh
        # channel 26-29: u10, v10, t2m, msl, sp
        ic(data_monthly.shape)
        
        data_monthly = np.expand_dims(data_monthly, axis=0)

        save_path = os.path.join(save_dir, f'{year}.h5')
        print(f'saving to {save_path}')
        if m == 1:
            f = h5py.File(save_path, 'w')
            f.create_dataset("fields", data=data_monthly, maxshape=[None, 30, 170, 360])
            f.close()
        else:
            f = h5py.File(save_path, 'a')
            f["fields"].resize((f["fields"].shape[0] + 1), axis = 0)
            f["fields"][-1:] = data_monthly
            f.close()

def get_min_max_value(sample_dir, year_range):

    for i, year in enumerate(year_range):
        ic(year)
        f = h5py.File(os.path.join(sample_dir, f'{year}.h5'), 'r')
        data = f['fields']

        if i == 0:
            max_global = np.max(data, axis=(0,2,3), keepdims=False)
            min_global = np.min(data, axis=(0,2,3), keepdims=False)
            ic(max_global, max_global.shape)
            ic(min_global, min_global.shape)
        else:
            max_local = np.max(data, axis=(0,2,3), keepdims=False)
            min_local = np.min(data, axis=(0,2,3), keepdims=False)

            max_global[max_global < max_local] = max_local[max_global < max_local]
            min_global[min_global > min_local] = min_local[min_global > min_local]

        f.close()

    f = h5py.File(os.path.join(sample_dir, f'min_max_val_{year_range[0]}_{year_range[-1]}.h5'), 'w')
    f.create_dataset("max_value", data=max_global, maxshape=[30])
    f.create_dataset("min_value", data=min_global, maxshape=[30])
    f.close()

if __name__ == '__main__':

    degree = 1
    depths = [0, 50, 100, 300, 500, 1000]
    save_dir = '/work/home/acrzcyisbk/Ocean_AI_model/sample_02'
    year = 2000
    for year in np.arange(2010, 2011):
        gen_sample(depths, degree, year, save_dir)

    # years = np.arange(1995, 1998)
    # sample_dir = '../sample_02'
    # get_min_max_value(sample_dir, years)

