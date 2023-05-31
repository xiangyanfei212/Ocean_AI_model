import os
import h5py
import xarray as xr
import numpy as np
from icecream import ic

def normalize_orography(in_file, out_file, mode:str): 
    # mode: zscore or maxmin

    if mode == 'zscore':
        ds = xr.open_dataset(in_file)
        orog = ds['Band1'].values
        omean = np.mean(orog)
        ic(omean)
        ostd = np.std(orog)
        ic(ostd)
        
        orog -= omean
        orog /= ostd

        ic(np.min(orog), np.max(orog))
        
        f = h5py.File(out_file, 'w')
        f.create_dataset("orog", data=orog)
        f.close()
        ds.close()

    if mode == 'maxmin':
        ds = xr.open_dataset(in_file)
        orog = ds['Band1'].values
        omax = np.max(orog)
        ic(omax)
        omin = np.min(orog)
        ic(omin)
        
        orog = (orog - omin) / (omax - omin)
        ic(np.min(orog), np.max(orog))
        
        f = h5py.File(out_file, 'w')
        f.create_dataset("orog", data=orog)
        f.close()
        ds.close()

if __name__ == '__main__':
    # in_file = '/work/home/acrzcyisbk/ETOPO/global_topo_1degree.nc'
    # mode = 'maxmin' # maxmin
    # out_file = f'../sample_02/orography_norm_{mode}.h5'
    # normalize_orography(in_file, out_file, mode)

    in_file = '/work/home/acrzcyisbk/ETOPO/global_topo_025degree.nc'
    mode = 'zscore' # maxmin
    out_file = f'../sample_03/orography_norm_{mode}.h5'
    normalize_orography(in_file, out_file, mode)
