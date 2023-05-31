import os
import numpy as np
import pandas as pd 
import h5py
from icecream import ic
import matplotlib
import matplotlib.pyplot as plt

VAR_INDEX = {
    "T0":0,  "T50":1,  "T100":2,  "T300":3,  "T500":4,  "T1000":5,
    "S0":6,  "S50":7,  "S100":8,  "S300":9,  "S500":10, "S1000":11,
    "U0":12, "U50":13, "U100":14, "U300":15, "U500":16, "U1000":17,
    "V0":18, "V50":19, "V100":20, "V300":21, "V500":22, "V1000":23,
    "ssh": 24,
} 
TIME_STEPS = np.arange(30)


def convert_mean_weighted_acc_rmse_to_csv(h5_path, out_dir):

    f = h5py.File(in_h5_path, 'r')
    ic(f.keys())

    acc = f['acc'][:]
    rmse = f['rmse'][:]
    ic(acc.shape)
    ic(rmse.shape)

    acc_df = pd.DataFrame(acc[0], index=time_steps, columns=VAR_INDEX.keys())
    rmse_df = pd.DataFrame(rmse[0], index=time_steps, columns=VAR_INDEX.keys())

    acc_df.to_csv(os.path.join(out_dir, 'mean_weighted_acc.csv'))
    rmse_df.to_csv(os.path.join(out_dir, 'mean_weighted_rmse.csv'))

def plot_shaded_curve(h5_path, var_name, prediction_length, out_dir):
    f = h5py.File(h5_path, 'r')
    ic(f.keys())
    acc = f['acc'][:] # (#init_conditions, #prediction_length, #vars)
    rmse = f['rmse'][:]

    var_idx = VAR_INDEX[var_name]
    acc = acc[:, :prediction_length, var_idx] # (#init_conditions, #prediction_length)
    rmse = rmse[:, :prediction_length, var_idx]

    acc_mean = np.mean(acc, axis=0) # (#prediction_length)
    rmse_mean = np.mean(rmse, axis=0)

    acc_025 = np.quantile(acc, .25, axis=0)
    acc_075 = np.quantile(acc, .75, axis=0)
    rmse_025 = np.quantile(rmse, .25, axis=0)
    rmse_075 = np.quantile(rmse, .75, axis=0)

    x = np.arange(prediction_length)
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5), dpi=200)

    ax1.plot(x, acc_mean, color='orangered', linestyle='-', linewidth=1)
    ax1.fill_between(x, acc_025, acc_075, color='peachpuff', edgecolor=None)
    ax1.grid(linestyle='--', linewidth=1, alpha=0.3)

    ax1.set_ylabel('ACC (latitude weighted)')
    ax1.set_xlabel("Fourcast time (days)")
    ax1.set_title(var_name)
    
    ax2.plot(x, rmse_mean, color='orangered', linestyle='-', linewidth=1)
    ax2.fill_between(x, rmse_025, rmse_075, color='peachpuff', edgecolor=None)
    ax2.grid(linestyle='--', linewidth=1, alpha=0.3)

    ax2.set_ylabel('RMSE (latitude weighted)')
    ax2.set_xlabel("Fourcast time (days)")
    ax2.set_title(var_name)

    plt.tight_layout()
    out_fig_path = os.path.join(out_dir, f"shaded_acc_rmse_{var_name}.png")
    print(f'Saving fig to {out_fig_path}')
    plt.savefig(out_fig_path, dpi=200)
    plt.show()

if __name__ == '__main__':
    # in_dir = '/work/home/acrzcyisbk/Ocean_AI_model/exp/afno_backbone/20230524-163725'
    in_dir = '/work/home/acrzcyisbk/Ocean_AI_model/exp/Masked_AE_Ocean/20230528-Mask_AE_v2_landmask/'
    in_h5_file = os.path.join(in_dir, 'autoregressive_predictions.h5')
    prediction_length = 30
    plot_shaded_curve(in_h5_file, 'ssh', prediction_length, out_dir=in_dir)
    plot_shaded_curve(in_h5_file, 'T1000', prediction_length, out_dir=in_dir)
    
    # convert_mean_weighted_acc_rmse_to_csv(in_h5_path, out_dir=in_dir)

