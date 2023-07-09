import os
import numpy as np
import pandas as pd 
import h5py
from icecream import ic
import matplotlib
import matplotlib.pyplot as plt
from var_channels import *


def convert_mean_weighted_acc_rmse_to_csv(h5_path, out_dir):

    f = h5py.File(in_h5_path, 'r')
    ic(f.keys())

    acc = f['acc'][:]
    rmse = f['rmse'][:]
    ic(acc.shape)
    ic(rmse.shape)

    acc_df = pd.DataFrame(acc[0], index=time_steps, columns=VAR_INDEX_6_LEVELS.keys())
    rmse_df = pd.DataFrame(rmse[0], index=time_steps, columns=VAR_INDEX_6_LEVELS.keys())

    acc_df.to_csv(os.path.join(out_dir, 'mean_weighted_acc.csv'))
    rmse_df.to_csv(os.path.join(out_dir, 'mean_weighted_rmse.csv'))

def plot_single_shaded_curve(levels:int, h5_path, var_name, prediction_length, out_dir):
    # acc/rmse : (#init_conditions, #prediction_length, #vars)

    print(f'variable: {var_name}')
    if levels == 6:
        var_idx = VAR_INDEX_6_LEVELS[var_name]
    if levels == 15:
        var_idx = VAR_INDEX_15_LEVELS[var_name]

    f = h5py.File(h5_path, 'r')
    ic(f.keys())
    acc = f['acc'][:, :prediction_length, var_idx] # (#init_conditions, #prediction_length)
    rmse = f['rmse'][:, :prediction_length, var_idx] # 

    acc_mean = np.mean(acc, axis=0) # (#prediction_length)
    rmse_mean = np.mean(rmse, axis=0)

    # mean metrics for 30-days forecasting 
    acc_mean_ = np.mean(acc_mean)
    rmse_mean_ = np.mean(rmse_mean)
    # print(f'mean metrics for 30-days forecasting: acc={acc_mean_}, rmse={rmse_mean_}')

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
    ax1.set_xlabel("Forecast time (days)")
    ax1.set_title(var_name)
    
    ax2.plot(x, rmse_mean, color='orangered', linestyle='-', linewidth=1)
    ax2.fill_between(x, rmse_025, rmse_075, color='peachpuff', edgecolor=None)
    ax2.grid(linestyle='--', linewidth=1, alpha=0.3)

    ax2.set_ylabel('RMSE (latitude weighted)')
    ax2.set_xlabel("Forecast time (days)")
    ax2.set_title(var_name)

    plt.tight_layout()
    out_fig_path = os.path.join(out_dir, f"shaded_acc_rmse_{var_name}.png")
    print(f'Saving fig to {out_fig_path}')
    plt.savefig(out_fig_path, dpi=200)
    plt.show()

def plot_multi_shaded_curve(levels:int, h5_path_list:list, model_name_list:list, var_name, prediction_length, out_dir):
    # acc/rmse : (#init_conditions, #prediction_length, #vars)

    colors = ['#924B43', '#B1CE46', '#5F97D2', '#9394E7', '#63E398', '#EF7A6D']

    print(f'variable: {var_name}')
    if levels == 6:
        var_idx = VAR_INDEX_6_LEVELS[var_name]
    if levels == 15:
        var_idx = VAR_INDEX_15_LEVELS[var_name]

    if var_name == 'SSH':
        acc_y_lim = [0.7, 1]
        rmse_y_lim = [0, 0.6]
    if var_name[0] == 'T':
        acc_y_lim = [0.65, 1]
        rmse_y_lim = [0, 10]
    if var_name[0] == 'S' and var_name != 'SSH':
        acc_y_lim = [0.4, 1]
        rmse_y_lim = [0, 2.5]
    if var_name[0] == 'U' or var_name[0] == 'V':
        acc_y_lim = [0, 1]
        rmse_y_lim = [0, 0.3]

    x = np.arange(prediction_length)
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5), dpi=200)

    for i, (h5_path, model_name) in enumerate(zip(h5_path_list, model_name_list)):
        print('model:', model_name)
        print('reading ', h5_path)

        f = h5py.File(h5_path, 'r')
        ic(f.keys())
        acc = f['acc'][:, :prediction_length, var_idx] # (#init_conditions, #prediction_length)
        rmse = f['rmse'][:, :prediction_length, var_idx] # 

        acc_mean = np.mean(acc, axis=0) # (#prediction_length)
        rmse_mean = np.mean(rmse, axis=0)

        # mean metrics for 30-days forecasting 
        acc_mean_ = np.mean(acc_mean)
        rmse_mean_ = np.mean(rmse_mean)
        print(f'mean metrics for 30-days forecasting: acc={acc_mean_}, rmse={rmse_mean_}')

        acc_025 = np.quantile(acc, .25, axis=0)
        acc_075 = np.quantile(acc, .75, axis=0)
        rmse_025 = np.quantile(rmse, .25, axis=0)
        rmse_075 = np.quantile(rmse, .75, axis=0)

        ax1.plot(x, acc_mean, color=colors[i], linestyle='-', linewidth=1, label=model_name)
        ax1.fill_between(x, acc_025, acc_075, color=colors[i], edgecolor=None, alpha=0.5)
        ax1.grid(linestyle='--', linewidth=1, alpha=0.3)
        ax1.set_ylim(acc_y_lim[0], acc_y_lim[1])

        ax2.plot(x, rmse_mean, color=colors[i], linestyle='-', linewidth=1, label=model_name)
        ax2.fill_between(x, rmse_025, rmse_075, color=colors[i], edgecolor=None, alpha=0.5)
        ax2.grid(linestyle='--', linewidth=1, alpha=0.3)
        ax2.set_ylim(rmse_y_lim[0], rmse_y_lim[1])

        f.close()

    ax1.legend()
    ax1.set_ylabel('ACC (latitude weighted)')
    ax1.set_xlabel("Forecast time (days)")
    ax1.set_title(var_name)
    
    ax2.legend()
    ax2.set_ylabel('RMSE (latitude weighted)')
    ax2.set_xlabel("Forecast time (days)")
    ax2.set_title(var_name)

    plt.tight_layout()
    out_fig_path = os.path.join(out_dir, f"shaded_acc_rmse_{var_name}.png")
    print(f'Saving fig to {out_fig_path}')
    plt.savefig(out_fig_path, dpi=200)
    plt.show()

if __name__ == '__main__':

    prediction_length = 30

    ##########################################################
    levels = 6
    in_dir_1 = '/home/bingxing2/home/scx6115/Ocean_AI_model/exp_6_levels/afno_backbone/20230609-155507'
    in_h5_file_1 = os.path.join(in_dir_1, 'autoregressive_predictions.h5')

    in_dir_2 = '/home/bingxing2/home/scx6115/Ocean_AI_model/exp_6_levels/Masked_AE_Ocean/20230610'
    in_h5_file_2 = os.path.join(in_dir_2, 'autoregressive_predictions.h5')

    in_dir_3 = '/home/bingxing2/home/scx6115/Ocean_AI_model/exp_6_levels/Masked_AE_fusion/20230613-203822'
    in_h5_file_3 = os.path.join(in_dir_3, 'autoregressive_predictions.h5')

    # plot_single_shaded_curve(levels, in_h5_file, 'ssh',  prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file, 'T0',   prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file, 'T100', prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file, 'S0',   prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file, 'S100', prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file, 'U0',   prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file, 'U100', prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file, 'V0',   prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file, 'V100', prediction_length, out_dir=in_dir)
    
    # for var in ['SSH', 'T0', 'T50', 'T100', 'S0', 'S50', 'S100', 'U0', 'U50', 'U100', 'V0', 'V50', 'V100']:
    #     plot_multi_shaded_curve(levels, [in_h5_file_1, in_h5_file_2, in_h5_file_3], ['AFNO', 'MAE_ocean', 'MAE_fusion'], var, prediction_length, out_dir='../exp_6_levels')

    # convert_mean_weighted_acc_rmse_to_csv(in_h5_path, out_dir=in_dir)
    ##########################################################

    ##########################################################
    levels = 15
    prediction_length = 30
    in_dir = '../exp_15_levels/afno/20230628-131420'
    in_h5_file_1 = os.path.join(in_dir, 'autoregressive_predictions.h5')

    in_dir = '../exp_15_levels/Masked_AE_Ocean/20230628-131150/'
    in_h5_file_2 = os.path.join(in_dir, 'autoregressive_predictions.h5')

    in_dir = '../exp_15_levels/Masked_AE_Ocean/20230629-222717/'
    in_h5_file_3 = os.path.join(in_dir, 'autoregressive_predictions.h5')

    in_dir = '/home/bingxing2/home/scx6115/Ocean_AI_model/exp_15_levels/Masked_AE_Ocean/20230628-131150/2_steps_finetune'
    in_h5_file_4 = os.path.join(in_dir, 'autoregressive_predictions.h5')

    in_dir = '../exp_15_levels/Masked_AE_Ocean/20230701-220232/'
    in_h5_file_5 = os.path.join(in_dir, 'autoregressive_predictions.h5')

    # plot_single_shaded_curve(levels, in_h5_file_1, 'SSH',  prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file_1, 'T0',   prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file_1, 'T100', prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file_1, 'T500', prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file_1, 'S0',   prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file_1, 'S100', prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file_1, 'S500', prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file_1, 'U0',   prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file_1, 'U100', prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file_1, 'U500', prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file_1, 'V0',   prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file_1, 'V100', prediction_length, out_dir=in_dir)
    # plot_single_shaded_curve(levels, in_h5_file_1, 'V500', prediction_length, out_dir=in_dir)

    for var in ['SSH', 'T0', 'T50', 'T100', 'S0', 'S50', 'S100', 'U0', 'U50', 'U100', 'V0', 'V50', 'V100']:
    # for var in ['S0', 'S50', 'S100']:

        plot_multi_shaded_curve(levels, 
                                [in_h5_file_1, in_h5_file_2, in_h5_file_3, in_h5_file_4, in_h5_file_5], 
                                ['AFNO', 'MAE_ocean_20230628', 'MAE_ocean_20230629', 'MAE_ocean_20230628_2_steps_finetune', 'MAE_Ocean_20230701'], 
                                var, 
                                prediction_length, 
                                out_dir='../exp_15_levels')
    
    ##########################################################
