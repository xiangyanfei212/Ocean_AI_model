"""
BSD 3-Clause License

Copyright (c) 2022, FourCastNet authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

-------------------------------------------------------------------------------

Project: Ocean AI Model
Modified by: Yanfei Xiang

Description:
This script is used for distributed training of deep learning models for ocean data analysis.
It includes functionality for:
- Loading datasets (training, validation, and test sets)
- Distributed data parallel training
- Mixed precision training (AMP)
- Model checkpointing and resuming
- Integration with Weights & Biases (WandB) for experiment tracking

Key Modifications:
1. Updated to support ocean simulations.
1. Added region-weighted LpLoss function to handle specific ocean regions.
2. Integrated land masking functionality to filter out land data.
3. Updated to support new model architecture.

"""

import os
import sys
import time
import h5py
import json
import torch
import wandb
import pickle
import logging
import argparse
import cProfile
import numpy as np
import matplotlib.pyplot as plt

from icecream import ic
from apex import optimizers
from shutil import copyfile
from collections import OrderedDict

import torchvision
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torchsummary import summary
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel

from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.darcy_loss import LossScaler, LpLoss, channel_wise_LpLoss, LpLoss_region_weighted 
from utils.data_loader_multifiles import get_data_loader
from utils.weighted_acc_rmse import weighted_acc, weighted_rmse, weighted_rmse_torch

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

os.environ["WANDB_API_KEY"] = "" # your wandb API KEY
os.environ["WANDB_MODE"] = "online" # optional: online, dryrun, offline
# if 'WANDB_MODE' is set to 'dryrun', first 'cd wandb', then run 'wandb sync + $dir', the log will updated to the wandb website.

class Trainer():
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad) 

    def __init__(self, params, world_rank):

        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

        # Init wandb
        if params.log_to_wandb:
            wandb.init(config=params, 
                       name=params.name, 
                       group=params.group, 
                       project=params.project,
                       entity=params.entity)

        # Init gpu
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.device = torch.device('cuda', local_rank)
        logging.info('device: %s' % self.device)

        # Load data
        logging.info('rank %d, begin data loader init' % world_rank)
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(
                params, 
                params.train_data_path,
                dist.is_initialized(),
                train=True)
        self.valid_data_loader, self.valid_dataset, self.valid_sampler = get_data_loader(
                params, 
                params.valid_data_path,
                dist.is_initialized(), 
                train=True)

        if params.loss_region_weighted:
            elf.loss_obj = LpLoss_region_weighted(
                    params.region_idx, 
                    params.region_weight,
                    scale = params.loss_scale)
        elif params.loss_channel_wise:
            self.loss_obj = channel_wise_LpLoss(scale = params.loss_scale)
        else:
            self.loss_obj = LpLoss()

        # loss scaler
        self.mse_loss_scaler = LossScaler()

        logging.info('rank %d, data loader initialized' % world_rank)

        # Load model
        if params.nettype == 'afno':
            from networks.afnonet import AFNONet as model
        elif params.nettype == 'Masked_AE_Ocean':
            from networks.Masked_AE_Ocean import Masked_Ocean as model
        elif params.nettype == 'Masked_AE_fusion':
            from networks.Masked_AE_fusion import Masked_AFNO as model
        else:
            raise Exception("not implemented")
        self.model = model(params).to(self.device)

        if params.log_to_wandb:
            wandb.watch(self.model)

        if params.optimizer_type == 'FusedAdam':
            self.optimizer = optimizers.FusedAdam(self.model.parameters(), lr = params.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = params.lr)

        if params.enable_amp == True:
            self.gscaler = amp.GradScaler()

        if dist.is_initialized():
            self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[params.local_rank],
                    output_device=[params.local_rank],
                    find_unused_parameters=True
            )

        self.iters = 0
        self.startEpoch = 0

        if (params.multi_steps_finetune == 1) and (params.resuming):
            logging.info("Loading checkpoint %s" % params.checkpoint_path)
            self.restore_checkpoint(params.checkpoint_path)

        if params.multi_steps_finetune > 1:
            logging.info("Starting from pretrained one-step model at %s"%params.pretrained_ckpt_path)
            self.restore_checkpoint(params.pretrained_ckpt_path)
            self.iters = 0
            self.startEpoch = 0
            logging.info("Adding %d epochs specified in config file for refining pretrained model"%params.finetune_max_epochs)
            params['max_epochs'] = params.finetune_max_epochs

        self.epoch = self.startEpoch

        # Dynamical Learning rate
        if params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, 
                    factor=0.2, 
                    patience=5, 
                    mode='min'
            )
        elif params.scheduler == 'CosineAnnealingLR': 
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=params.max_epochs,
                    last_epoch=self.startEpoch - 1
            )
        else:
            self.scheduler = None

        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))

    def switch_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")

        best_valid_loss = 1.e6
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)
                self.valid_sampler.set_epoch(epoch)

            start = time.time()
            tr_time, data_time, step_time, train_logs = self.train_one_epoch() 
            valid_time, valid_logs = self.validate_one_epoch()

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
                if self.epoch >= self.params.max_epochs:
                    logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                    exit()

            if self.params.log_to_wandb:
                for pg in self.optimizer.param_groups:
                    lr = pg['lr']
                wandb.log({'lr': lr})

            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    # checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path)
                    if valid_logs['valid_loss'] <= best_valid_loss:
                        logging.info('Val loss improved from {} to {}'.format(best_valid_loss, valid_logs['valid_loss']))
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        best_valid_loss = valid_logs['valid_loss']

            if self.params.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                logging.info('train data time={}, train per epoch time={}, train per step time={}, valid time={}'.format(data_time, tr_time, step_time, valid_time))
                logging.info('Train loss: {}. Valid loss: {}'.format(train_logs['train_loss'], valid_logs['valid_loss']))

    def land_mask_func(self, x, y, land_mask_path):
        # 0:land, 1:ocean
        with h5py.File(land_mask_path, 'r') as _f: 
            # logging.info(f"Loading land mask data from {self.params.land_mask_path}")
            mask_data = torch.as_tensor(_f['fields'])
            mask_data = mask_data[0,self.params.out_channels].to(x.device, dtype=torch.bool)
        x = torch.masked_fill(input=x, mask=~mask_data, value=0)
        y = torch.masked_fill(input=y, mask=~mask_data, value=0)
        return x, y 

    def train_one_epoch(self):
        self.epoch += 1
        tr_time = 0
        data_time = 0
        self.model.train()

        if self.epoch <= 3:
            mask_ratio = 0
        elif self.epoch <= 20:
            mask_ratio = 0.75
        elif self.epoch <= 40:
            mask_ratio = 0.25
        else:
            mask_ratio = 0

        steps_in_one_epoch = 0
        for i, data in enumerate(self.train_data_loader, 0):
            self.iters += 1
            steps_in_one_epoch += 1 

            data_start = time.time()

            if params.fusion_3d_2d:
                inp_2d, inp_3d, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
            else:
                (inp, tar) = data
              
            if self.params.orography and self.params.multi_steps_finetune > 1:
                orog = torch.unsqueeze(inp[:,-1], dim=1)

            data_time += time.time() - data_start

            tr_start = time.time()
            self.model.zero_grad()

            if self.params.multi_steps_finetune == 2:
                orog = orog.to(self.device, dtype = torch.float)
                with amp.autocast(self.params.enable_amp):
                    # step 1
                    inp_step_1 = inp.to(self.device, dtype = torch.float32)
                    gen_step_1  = self.model(inp_step_1).to(self.device, dtype = torch.float32)
                    tar_step_1  = tar[:, 0, self.params.out_channels].to(self.device, dtype=torch.float)

                    if self.params.land_mask:
                        gen_step_1, tar_step_1 = self.land_mask_func(gen_step_1, tar_step_1, self.params.land_mask_path)

                    loss_step_1, cw_loss_step_1 = self.loss_obj(gen_step_1, tar_step_1)
                    tar_step_1  = tar_step_1.cpu()
                    del tar_step_1

                    # step 2
                    if self.params.orography:
                        atmos_force = tar[:, 1, self.params.atmos_channels].to(self.device, dtype=torch.float)
                        inp_step_2 = torch.cat( (gen_step_1, atmos_force, orog), axis = 1).to(self.device, dtype = torch.float32)
                        gen_step_2 = self.model(inp_step_2).to(self.device, dtype = torch.float32)
                    else:
                        atmos_force = tar[:, 1, self.params.atmos_channels].to(self.device, dtype=torch.float)
                        gen_step_2 = self.model( torch.cat((gen_step_1, atmos_force), axis = 1) ).to(self.device, dtype = torch.float32)

                    tar_step_2  = tar[:, 1, self.params.out_channels].to(self.device, dtype=torch.float32)

                    if self.params.land_mask:
                        gen_step_2, tar_step_2 = self.land_mask_func(gen_step_2, tar_step_2, self.params.land_mask_path)

                    loss_step_2, cw_loss_step_2 = self.loss_obj(gen_step_2, tar_step_2)
                    tar_step_2  = tar_step_2.cpu()
                    del tar_step_2

                    loss = loss_step_1 + loss_step_2
                    cw_loss = cw_loss_step_1 + cw_loss_step_2

            elif self.params.multi_steps_finetune == 3:
                with amp.autocast(self.params.enable_amp):
                    # step 1
                    gen_step_1  = self.model(inp.to(self.device, dtype=torch.float32))
                    tar_step_1  = tar[:, 0, self.params.out_channels].to(self.device, dtype=torch.float)
                    if self.params.land_mask:
                        gen_step_1, tar_step_1 = self.land_mask_func(gen_step_1, tar_step_1, self.params.land_mask_path)
                    loss_step_1 = self.loss_obj(gen_step_1, tar_step_1)
                    tar_step_1  = tar_step_1.cpu()
                    del inp, tar_step_1
                    torch.cuda.empty_cache()

                    # step 2
                    if self.params.orography:
                        atmos_force = tar[:, 0, -5:]
                        inp_step_2 = torch.cat( (gen_step_1.cpu(), atmos_force, orog), axis = 1)
                        gen_step_2 = self.model( inp_step_2.to(self.device, dtype=torch.float) )
                        del  gen_step_1, atmos_force, inp_step_2
                    else:
                        atmos_force = tar[:, 0, -5:].to(self.device, dtype=torch.float)
                        gen_step_2 = self.model( torch.cat( (gen_step_1, atmos_force), axis = 1) )

                    tar_step_2  = tar[:, 1, self.params.out_channels].to(self.device, dtype=torch.float)
                    if self.params.land_mask:
                        gen_step_2, tar_step_2 = self.land_mask_func(gen_step_2, tar_step_2, self.params.land_mask_path)
                    loss_step_2 = self.loss_obj(gen_step_2, tar_step_2)
                    tar_step_2  = tar_step_2.cpu()
                    del tar_step_2
                    torch.cuda.empty_cache()

                    # step 3
                    if self.params.orography:
                        atmos_force = tar[:, 1, -5:]
                        inp_step_3 = torch.cat( (gen_step_2.cpu(), atmos_force, orog), axis = 1)
                        gen_step_3 = self.model( inp_step_3.to(self.device, dtype=torch.float) )
                        del gen_step_2, atmos_force, inp_step_3
                    else:
                        atmos_force = tar[:, 1, -5:].to(self.device, dtype=torch.float)
                        gen_step_3 = self.model( torch.cat( (gen_step_2, atmos_force), axis = 1) )

                    tar_step_3  = tar[:, 2, 0:self.params.N_out_channels].to(self.device, dtype=torch.float)
                    if self.params.land_mask:
                        gen_step_3, tar_step_3 = self.land_mask_func(gen_step_3, tar_step_3, self.params.land_mask_path)
                    loss_step_3 = self.loss_obj(gen_step_3, tar_step_3)
                    tar_step_3  = tar_step_3.cpu()
                    del tar_step_3
                    torch.cuda.empty_cache()

                    loss = loss_step_1 + loss_step_2 + loss_step_3
            else:
                with amp.autocast(self.params.enable_amp):
                    if params.fusion_3d_2d:
                        gen = self.model(inp_2d, inp_3d, mask_ratio=mask_ratio)
                    else:
                        inp = inp.to(self.device, dtype = torch.float)
                        tar = tar.to(self.device, dtype = torch.float)
                        gen = self.model(inp)
                    gen.to(self.device, dtype=torch.float)

                    if self.params.land_mask:
                        gen, tar = self.land_mask_func(gen, tar, self.params.land_mask_path)

                    if self.params.use_loss_scaler_from_metnet3:
                        gen = self.mse_loss_scaler(gen)

                    loss, cw_loss = self.loss_obj(gen, tar)
                
            if self.params.enable_amp:
                self.gscaler.scale(loss).backward()
                self.gscaler.step(self.optimizer)
            else:
                loss.backward()
                self.optimizer.step()

            if self.params.enable_amp:
                self.gscaler.update()

            tr_time += time.time() - tr_start

        logs = {'train_loss': loss}

        for vi, v in enumerate(self.params.out_variables):
            logs[f'{v}_train_loss'] = cw_loss[vi]

        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach())
                logs[key] = float(logs[key] / dist.get_world_size())

        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        # time of one step in epoch
        step_time = tr_time / steps_in_one_epoch

        return tr_time, data_time, step_time, logs

    def validate_one_epoch(self):

        logging.info('validating...')
        self.model.eval()

        valid_buff  = torch.zeros((3+self.params.N_out_channels), dtype=torch.float32, device=self.device)
        valid_loss  = valid_buff[0].view(-1) # 0
        valid_l1    = valid_buff[1].view(-1) # 0
        valid_steps = valid_buff[-1].view(-1) # 0

        valid_start = time.time()
        sample_idx = np.random.randint(len(self.valid_data_loader))
        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader, 0):

                if params.fusion_3d_2d:
                    inp_2d, inp_3d, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
                    gen = self.model(inp_2d, inp_3d, mask_ratio=0)
                else:
                    inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
                    gen = self.model(inp)
                gen.to(self.device, dtype=torch.float)

                # only compare to the first time-step
                if self.params.multi_steps_finetune > 1:
                    tar = tar[:, 0, self.params.out_channels] 

                if self.params.land_mask:
                    gen, tar = self.land_mask_func(gen, tar, self.params.land_mask_path)

                valid_loss_, cw_valid_loss = self.loss_obj(gen, tar)
                valid_loss += valid_loss_
                valid_l1   += nn.functional.l1_loss(gen, tar)

                for vi, v in enumerate(self.params.out_variables):
                    valid_buff[vi+2] += cw_valid_loss[vi]

                valid_steps += 1.

                # save fields for vis before log norm
                os.makedirs(params['experiment_dir'] + "/" + str(i), exist_ok =True)
                save_image(torch.cat((gen[0, 0],
                                      torch.zeros((self.valid_dataset.img_shape_x, 4)).to(self.device, dtype=torch.float),
                                      tar[0, 0]), axis=1),
                           os.path.join(params['experiment_dir'], str(i), f's0_epoch_{self.epoch}.png'))

        if dist.is_initialized():
            dist.all_reduce(valid_buff)

        # divide by number of steps
        valid_buff[0:-1] = valid_buff[0:-1] / valid_buff[-1] # loss/steps, l1/steps
        valid_buff_cpu = valid_buff.detach().cpu().numpy()

        valid_time = time.time() - valid_start
        
        logs = {'valid_loss': valid_buff_cpu[0],
                'valid_l1':   valid_buff_cpu[1]}
        for vi, v in enumerate(self.params.out_variables):
            logs[f'{v}_valid_loss'] = valid_buff_cpu[vi+2]

        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        return valid_time, logs

    def load_model(self, model_path):
        if self.params.log_to_screen:
            logging.info('Loading the model weights from {}'.format(model_path))

        checkpoint = torch.load(model_path, map_location='cuda:{}'.format(self.params.local_rank))

        if dist.is_initialized():
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            new_model_state = OrderedDict()
            model_key = 'model_state' if 'model_state' in checkpoint else 'state_dict'
            for key in checkpoint[model_key].keys():
                if 'module.' in key:  # model was stored using ddp which prepends module
                    name = str(key[7:])
                    new_model_state[name] = checkpoint[model_key][key]
                else:
                    new_model_state[key] = checkpoint[model_key][key]
            self.model.load_state_dict(new_model_state)
            self.model.eval()

    def save_checkpoint(self, checkpoint_path, model=None):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """

        if not model:
            model = self.model

        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)

    def restore_checkpoint(self, checkpoint_path):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']
        if self.params.resuming:  
        # restore checkpoint is used for finetuning as well as resuming. 
        # If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/config.yaml', type=str)  
    parser.add_argument("--multi_steps_finetune", default=1, type=int)  
    parser.add_argument("--finetune_max_epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--wandb_group", default='025_daily', type=str)
    parser.add_argument("--config", default='Masked_AE_Ocean', type=str)
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--local_rank", default=-1, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config, True)
    params['epsilon_factor'] = args.epsilon_factor
    params['multi_steps_finetune'] = args.multi_steps_finetune
    params['finetune_max_epochs']  = args.finetune_max_epochs

    params['world_size'] = 1
    if 'WORLD_SIZE' in os.environ:
        params['world_size'] = int(os.environ['WORLD_SIZE'])
    print('world_size :', params['world_size'])

    print('Initialize distributed process group...')
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    params['local_rank'] = local_rank  # GPU ID

    torch.backends.cudnn.benchmark = True
    world_rank = dist.get_rank()

    params['global_batch_size'] = args.batch_size
    params['batch_size'] = int(args.batch_size // params['world_size'])  # batch size must be divisible by the number of gpu's
    params['enable_amp'] = args.enable_amp  # Automatic Mixed Precision Training

    # Set up directory
    if params['multi_steps_finetune'] > 1:
        pretrained_expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
        params['pretrained_ckpt_path'] = os.path.join(pretrained_expDir, 'training_checkpoints/best_ckpt.tar')

        multi_steps = params['multi_steps_finetune']

        expDir = os.path.join(pretrained_expDir, f'{multi_steps}_steps_finetune')
        if world_rank == 0:
            os.makedirs(expDir, exist_ok=True)
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'), exist_ok=True)

        params['experiment_dir'] = os.path.abspath(expDir)
        params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar') 
        params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

        params['resuming'] = True
    else:
        expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
        if world_rank == 0:
            os.makedirs(expDir, exist_ok =True)
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'), exist_ok =True)
            copyfile(os.path.abspath(args.yaml_config), os.path.join(expDir, 'config.yaml'))

        params['experiment_dir'] = os.path.abspath(expDir)
        params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar') 
        params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

        # Do not comment this line out please:
        args.resuming = True if os.path.isfile(params.checkpoint_path) else False
        params['resuming'] = args.resuming

    # wandb setting 
    params['entity'] = "ocean_ai_model"  # team name 
    params['project'] = "ai4science"     # project name
    params['group'] = args.wandb_group   # "025_daily_15_levels"
    params['name'] = args.config + '_' + str(args.run_num)  # experiment name

    if world_rank == 0:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'train.log'))
        logging_utils.log_versions()
        params.log()

    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']

    params['in_channels'] = np.array(params['in_channels'])
    params['out_channels'] = np.array(params['out_channels'])
    params['N_out_channels'] = len(params['out_channels'])
    if params.orography:
        params['N_in_channels'] = len(params['in_channels']) + 1
    else:
        params['N_in_channels'] = len(params['in_channels']) 

    if world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)

    trainer = Trainer(params, world_rank)
    trainer.train()
    logging.info('DONE ---- rank %d' % world_rank)
