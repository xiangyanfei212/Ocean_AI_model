#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The code was authored by the following people:
#
#Jaideep Pathak - NVIDIA Corporation
#Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
#Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
#Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
#Ashesh Chattopadhyay - Rice University 
#Morteza Mardani - NVIDIA Corporation 
#Thorsten Kurth - NVIDIA Corporation 
#David Hall - NVIDIA Corporation 
#Zongyi Li - California Institute of Technology, NVIDIA Corporation 
#Kamyar Azizzadenesheli - Purdue University 
#Pedram Hassanzadeh - Rice University 
#Karthik Kashinath - NVIDIA Corporation 
#Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation
# Yanfei Xiang


import os
import sys
import time
import glob
import h5py
import wandb
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import OrderedDict
from numpy.core.numeric import False_

import torch
import torchvision
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels, unweighted_acc_torch_channels, weighted_acc_masked_torch_channels

from utils import logging_utils
logging_utils.config_logger()

fld = 'ssh'
DECORRELATION_TIME = 60 # TODO: interval time between initial condition time?
idxes = {
    "T0":0,  "T50":1,  "T100":2,  "T300":3,  "T500":4,  "T1000":5,
    "S0":6,  "S50":7,  "S100":8,  "S300":9,  "S500":10, "S1000":11,
    "U0":12, "U50":13, "U100":14, "U300":15, "U500":16, "U1000":17,
    "V0":18, "V50":19, "V100":20, "V300":21, "V500":22, "V1000":23,
    "ssh": 24,
} 

def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return (x + noise)

def load_model(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val  
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def setup(params):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # get data loader
    valid_data_loader, valid_dataset = get_data_loader(params, params.test_data_path, dist.is_initialized(), train=False)

    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y

    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)

    if params["orography"]:
        params['N_in_channels'] = n_in_channels + 1
    else:
        params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels

    if params.normalization == 'zscore': 
        params.means = np.load(params.global_means_path)
        params.stds = np.load(params.global_stds_path)
    if params.normalization == 'minmax': 
        params.mins = np.load(params.global_mins_path)
        params.maxs = np.load(params.global_maxs_path)

    if params.nettype == 'afno':
        from networks.afnonet import AFNONet as model
    elif params.nettype == 'model_v2':
        from networks.model_v2 import Model_iter as model
    elif params.nettype == 'Masked_AE_Ocean':
        from networks.Masked_AE_Ocean import Masked_AFNO as model
    else:
        raise Exception("not implemented")

    checkpoint_file  = params['best_checkpoint_path']
    if params.log_to_screen:
        logging.info('Loading trained model checkpoint from {}'.format(checkpoint_file))
    model = model(params).to(device) 
    model = load_model(model, params, checkpoint_file)
    model = model.to(device)

    # load the validation data
    files_paths = glob.glob(params.test_data_path + "/*.h5")
    files_paths.sort()
    # which year
    yr = 0
    if params.log_to_screen:
        logging.info('Loading inference data')
        logging.info('Inference data from {}'.format(files_paths[yr]))
    valid_data_full = h5py.File(files_paths[yr], 'r')['fields']

    return valid_data_full, model

    
def autoregressive_inference(params, ic, valid_data_full, model): 
    ic = int(ic) 
    
    # initialize global variables
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    exp_dir = params['experiment_dir'] 
    dt = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    n_history = params.n_history
    img_shape_x = params.img_shape_x
    img_shape_y = params.img_shape_y
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)

    # initialize memory for image sequences and RMSE/ACC
    valid_loss = torch.zeros((prediction_length, n_out_channels))
    acc = torch.zeros((prediction_length, n_out_channels))
    acc_unweighted = torch.zeros((prediction_length, n_out_channels))

    # compute metrics in a coarse resolution too if params.interp is nonzero
    valid_loss_coarse = torch.zeros((prediction_length, n_out_channels))
    acc_coarse = torch.zeros((prediction_length, n_out_channels))
    acc_coarse_unweighted = torch.zeros((prediction_length, n_out_channels))

    seq_real = torch.zeros((prediction_length, n_out_channels, img_shape_x, img_shape_y))
    seq_pred = torch.zeros((prediction_length, n_out_channels, img_shape_x, img_shape_y))

    # extract valid data from first year
    valid_data = valid_data_full[ic:(ic+prediction_length*dt+n_history*dt):dt, :, 0:720]
    logging.info(f'valid_data_full: {valid_data_full.shape}')
    logging.info(f'valid_data: {valid_data.shape}')
    
    if params.normalization == 'zscore': 
        valid_data = (valid_data - params.means)/params.stds
    if params.normalization == 'minmax': 
        valid_data = (valid_data - params.mins) / (params.maxs - params.mins)
    valid_data = torch.as_tensor(valid_data)

    # orography
    if params.orography and params.normalization == 'zscore': 
        orography_path = params.orography_norm_zscore_path
    if params.orography and params.normalization == 'maxmin': 
        orography_path = params.orography_norm_maxmin_path
    if params.orography:
        orog = torch.as_tensor(np.expand_dims(np.expand_dims(h5py.File(orography_path, 'r')['orog'][0:720], axis = 0), axis = 0))
        logging.info("orography loaded; shape:{}".format(orog.shape))

    # autoregressive inference
    if params.log_to_screen:
        logging.info('Begin autoregressive inference')

    with torch.no_grad():
        for i in range(valid_data.shape[0]): 
            if i==0: # start of sequence, t0 --> t0'
                first = valid_data[0:n_history+1]
                future = valid_data[n_history+1]
                for h in range(n_history+1):
                    seq_real[h] = first[h*n_in_channels : (h+1)*n_in_channels, 0:n_out_channels] # extract history from 1st 
                    seq_pred[h] = seq_real[h]
                if params.perturb:
                    first = gaussian_perturb(first, level=params.n_level, device=device) # perturb the ic
                if params.orography:
                    first = first.to(device, dtype=torch.float)
                    orog  = orog.to(device, dtype=torch.float)
                    future_pred = model(torch.cat((first, orog), axis=1))
                else:
                    first = first.to(device, dtype=torch.float)
                    future_pred = model(first)
            else: # (t1) --> (t+1)', (t+1)' --> (t+2)', (t+2)' --> (t+3)' ....
                if i < prediction_length-1:
                    future = valid_data[n_history+i+1]
                if params.orography:
                    future_pred = future_pred.to(device, dtype=torch.float)
                    future_force = torch.unsqueeze(future[n_out_channels:], dim=0).to(device, dtype=torch.float)
                    orog  = orog.to(device, dtype=torch.float)
                    future_pred = model(torch.cat((future_pred, future_force, orog), axis=1)) #autoregressive step
                else:
                    future_force = future[n_out_channels:].to(device, dtype=torch.float)
                    future_pred = model(torch.cat((future_pred, future_force), axis=1)) #autoregressive step

            if i < prediction_length - 1: # not on the last step
                seq_pred[n_history+i+1] = future_pred.cpu()
                seq_real[n_history+i+1] = future[:n_out_channels]
                history_stack = seq_pred[i+1:i+2+n_history]

            future_pred = history_stack

            pred = torch.unsqueeze(seq_pred[i], 0)
            tar  = torch.unsqueeze(seq_real[i], 0)
            print('pred:', pred.shape, 'tar:', tar.shape)

            if params.land_mask:
                # 0:land, 1:ocean
                with h5py.File(params.land_mask_path, 'r') as _f: 
                    logging.info(f"Loading land mask data from {params.land_mask_path}")
                    mask_data = torch.as_tensor(_f['fields'][:], dtype=bool)

                # 将 mask 张量中为 True 的元素在 input 张量中对应位置的元素替换为指定的 value 值
                pred = torch.masked_fill(input=pred, mask=~mask_data, value=0)
                tar  = torch.masked_fill(input=tar,  mask=~mask_data, value=0)

            # Compute metrics 
            if params.normalization == 'zscore': 
                valid_loss[i] = weighted_rmse_torch_channels(pred, tar) * params.stds[:,:n_out_channels,0,0]
            acc[i] = weighted_acc_torch_channels(pred, tar)
            acc_unweighted[i] = unweighted_acc_torch_channels(pred, tar)

            if params.interp > 0:
                pred = downsample(pred, scale=params.interp)
                tar = downsample(tar, scale=params.interp)
                if params.normalization == 'zscore': 
                    valid_loss_coarse[i] = weighted_rmse_torch_channels(pred, tar) * params.stds[:,:n_out_channels]
                acc_coarse[i] = weighted_acc_torch_channels(pred, tar)
                acc_coarse_unweighted[i] = unweighted_acc_torch_channels(pred, tar)

            if params.log_to_screen:
                idx = idxes[fld] 
                logging.info('Predicted timestep {} of {}. {} RMSE: {}, ACC: {}'.format(i, prediction_length, fld, valid_loss[i, idx], acc[i, idx]))

            if params.interp > 0:
                logging.info('[COARSE] Predicted timestep {} of {}. {} RMSE: {}, ACC: {}'.format(i, prediction_length, fld, valid_loss_coarse[i, idx],
                        acc_coarse[i, idx]))

        
    seq_real = seq_real.numpy()
    seq_pred = seq_pred.numpy()
    valid_loss = valid_loss.numpy()
    acc = acc.numpy()
    acc_unweighted = acc_unweighted.numpy()
    acc_coarse = acc_coarse.numpy()
    acc_coarse_unweighted = acc_coarse_unweighted.numpy()
    valid_loss_coarse = valid_loss_coarse.numpy()

    return (np.expand_dims(seq_real[n_history:], 0), 
            np.expand_dims(seq_pred[n_history:], 0), 
            np.expand_dims(valid_loss,0), 
            np.expand_dims(acc, 0),
            np.expand_dims(acc_unweighted, 0), 
            np.expand_dims(valid_loss_coarse, 0), 
            np.expand_dims(acc_coarse, 0),
            np.expand_dims(acc_coarse_unweighted, 0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='../config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--vis", default=True)
    parser.add_argument("--override_dir", default=None, type = str, help = 'Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--interp", default=0, type=float)
    parser.add_argument("--weights", default=None, type=str, help = 'Path to model weights, for use with override_dir option')
    
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['world_size'] = 1
    params['interp'] = args.interp
    params['global_batch_size'] = params.batch_size

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    vis = args.vis

    # Set up directory
    expDir = os.path.join('../', params.exp_dir, args.config, str(args.run_num))
    logging.info(f'expDir: {expDir}')
    params['experiment_dir'] = expDir 
    params['best_checkpoint_path'] = args.weights if args.override_dir is not None else os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['resuming'] = False
    params['local_rank'] = 0

    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out.log'))
    logging_utils.log_versions()
    params.log()

    # n_ics = params['n_initial_conditions']
    n_samples_per_year = 250 # 365 
    
    if params["ics_type"] == 'default':
        num_samples = n_samples_per_year-params.prediction_length
        stop = num_samples
        ics = np.arange(0, stop, DECORRELATION_TIME)
        n_ics = len(ics)
    elif params["ics_type"] == "datetime": # TODO: this function has not been debug
        date_strings = params["date_strings"]
        ics = []
        if params.perturb: 
            # like ensemble forecast, 
            # for perturbations use a single date and create n_ics perturbations
            n_ics = params["n_perturbations"]
            date = date_strings[0]
            date_obj = datetime.strptime(date,'%Y-%m-%d %H:%M:%S') 
            day_of_year = date_obj.timetuple().tm_yday - 1
            hour_of_day = date_obj.timetuple().tm_hour
            hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
            for ii in range(n_ics):
                ics.append(int(hours_since_jan_01_epoch/6))
        else:
            for date in date_strings:
                date_obj = datetime.strptime(date,'%Y-%m-%d %H:%M:%S') 
                day_of_year = date_obj.timetuple().tm_yday - 1
                hour_of_day = date_obj.timetuple().tm_hour
                hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
                ics.append(int(hours_since_jan_01_epoch/6))
        n_ics = len(ics)

    logging.info("Inference for {} initial conditions".format(n_ics))

    try:
      autoregressive_inference_filetag = params["inference_file_tag"]
    except:
      autoregressive_inference_filetag = ""
    if params.interp > 0:
        autoregressive_inference_filetag = "_coarse"

    # get data and models
    valid_data_full, model = setup(params)

    # initialize lists for image sequences and RMSE/ACC
    valid_loss = []
    valid_loss_coarse = []

    acc = []
    acc_unweighted = []

    acc_coarse = []
    acc_coarse_unweighted = []

    seq_pred = []
    seq_real = []

    # run autoregressive inference for multiple initial conditions
    for i, ic in enumerate(ics):
        logging.info("Initial condition {} of {}".format(i+1, n_ics))
        sr, sp, vl, a, au, vc, ac, acu = autoregressive_inference(params, ic, valid_data_full, model)

        if i ==0 or len(valid_loss) == 0:
            seq_real = sr
            seq_pred = sp
            valid_loss = vl
            valid_loss_coarse = vc
            acc = a
            acc_coarse = ac
            acc_coarse_unweighted = acu
            acc_unweighted = au
        else:
            seq_real = np.concatenate((seq_real, sr), 0)
            seq_pred = np.concatenate((seq_pred, sp), 0)
            valid_loss = np.concatenate((valid_loss, vl), 0)
            valid_loss_coarse = np.concatenate((valid_loss_coarse, vc), 0)
            acc = np.concatenate((acc, a), 0)
            acc_coarse = np.concatenate((acc_coarse, ac), 0)
            acc_coarse_unweighted = np.concatenate((acc_coarse_unweighted, acu), 0)
            acc_unweighted = np.concatenate((acc_unweighted, au), 0)

    prediction_length = seq_real[0].shape[0]
    n_out_channels = seq_real[0].shape[1]
    img_shape_x = seq_real[0].shape[2]
    img_shape_y = seq_real[0].shape[3]

    # save predictions and loss
    if params.log_to_screen:
        logging.info("Saving files at {}".format(os.path.join(params['experiment_dir'], 'autoregressive_predictions' + autoregressive_inference_filetag + '.h5')))
    with h5py.File(os.path.join(params['experiment_dir'], 'autoregressive_predictions'+ autoregressive_inference_filetag +'.h5'), 'a') as f:
        if vis:
            try:
                print('seq_real:', seq_real.shape)
                print(n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y)
                f.create_dataset("ground_truth", data = seq_real, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
            except: 
                del f["ground_truth"]
                f.create_dataset("ground_truth", data = seq_real, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
                f["ground_truth"][...] = seq_real

            try:
                print(f'seq_pred: ', seq_pred.shape)
                f.create_dataset("predicted", data = seq_pred, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
            except:
                del f["predicted"]
                f.create_dataset("predicted", data = seq_pred, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
                f["predicted"][...]= seq_pred

        try:
            f.create_dataset("rmse", data = valid_loss, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        except:
            del f["rmse"]
            f.create_dataset("rmse", data = valid_loss, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
            f["rmse"][...] = valid_loss

        try:
            f.create_dataset("acc", data = acc, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        except:
            del f["acc"]
            f.create_dataset("acc", data = acc, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
            f["acc"][...] = acc   

        try:
            f.create_dataset("rmse_coarse", data = valid_loss_coarse, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        except:
            del f["rmse_coarse"]
            f.create_dataset("rmse_coarse", data = valid_loss_coarse, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
            f["rmse_coarse"][...] = valid_loss_coarse

        try:
            f.create_dataset("acc_coarse", data = acc_coarse, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        except:
            del f["acc_coarse"]
            f.create_dataset("acc_coarse", data = acc_coarse, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
            f["acc_coarse"][...] = acc_coarse

        try:
            f.create_dataset("acc_unweighted", data = acc_unweighted, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        except:
            del f["acc_unweighted"]
            f.create_dataset("acc_unweighted", data = acc_unweighted, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
            f["acc_unweighted"][...] = acc_unweighted     

        try:
            f.create_dataset("acc_coarse_unweighted", data = acc_coarse_unweighted, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        except:
            del f["acc_coarse_unweighted"]
            f.create_dataset("acc_coarse_unweighted", data = acc_coarse_unweighted, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
            f["acc_coarse_unweighted"][...] = acc_coarse_unweighted     
        
        f.close()
