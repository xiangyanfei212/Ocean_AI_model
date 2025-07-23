import glob
import h5py
import math
import torch
import random
import logging
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from utils.img_utils import reshape_fields, reshape_downstream_fields


def get_data_loader(params, files_pattern, distributed, train):
    dataset = GetDataset(params, files_pattern, train)
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    # DistributedSampler:
    #   Allocate a part of the dataset to each process/gpu,
    #   and avoid duplication of data between different processes

    dataloader = DataLoader(
        dataset,
        batch_size=int(params.batch_size),
        num_workers=params.num_data_workers,
        shuffle=False,  # (sampler is None),
        sampler=sampler if train else None,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )  # pin_memory能加快内存的Tensor转义到GPU的显存的速度

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


class GetDataset(Dataset):
    def __init__(self, params, location, train):
        self.params = params
        self.location = location
        self.train = train
        self.orography = params.orography
        self.normalize = params.normalize
        self.dt = params.dt
        self.n_history = params.n_history
        self.in_channels = np.array(params.in_channels)
        self.out_channels = np.array(params.out_channels)
        self.atmos_channels = np.array(params.atmos_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)

        self._get_files_stats()
        self.add_noise = params.add_noise if train else False
        self.fusion_3d_2d = params.fusion_3d_2d

    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)

        with h5py.File(self.files_paths[0], "r") as _f:
            logging.info("Getting file stats from {}".format(self.files_paths[0]))

            # self.n_samples_per_year = _f['fields'].shape[0] - 1
            self.n_samples_per_year = (
                _f["fields"].shape[0] - self.params.multi_steps_finetune
            )

            # original image shape (before padding)
            self.img_shape_x = (
                _f["fields"].shape[2] - 1
            )  # just get rid of one of the pixels
            self.img_shape_y = _f["fields"].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]

        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info(
            "Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
                self.location,
                self.n_samples_total,
                self.img_shape_x,
                self.img_shape_y,
                self.n_in_channels,
            )
        )
        logging.info("Delta t: {} days".format(1 * self.dt))
        logging.info(
            "Including {} days of past history in training at a frequency of {} days".format(
                1 * self.dt * self.n_history, 1 * self.dt
            )
        )

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], "r")
        self.files[year_idx] = _file["fields"]

        if self.orography and self.params.normalization == "zscore":
            _orog_file = h5py.File(self.params.orography_norm_zscore_path, "r")
        if self.orography and self.params.normalization == "maxmin":
            _orog_file = h5py.File(self.params.orography_norm_maxmin_path, "r")
        self.orography_field = _orog_file["orog"]

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx = int(global_idx / self.n_samples_per_year)  # which year
        local_idx = int(global_idx % self.n_samples_per_year)  # which sample in a year

        if self.files[year_idx] is None:
            self._open_file(year_idx)

        # If there are not enough historical time steps available in the features, shift to future time steps.
        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        # If the sample is the final one for the year, predict the current time step. Otherwise, predict the next time step.

        step = 0 if local_idx >= self.n_samples_per_year - self.dt else self.dt

        if self.orography:
            orog = self.orography_field
            if np.shape(orog)[0] == 721:
                orog = orog[0:720]
            # logging.info(f'orog: {orog.shape}')
        else:
            orog = None

        # logging.info(f'year_idx: {year_idx}, local_idx:{local_idx}, dt:{self.dt}, step:{step}, n_history:{self.n_history}')
        if self.fusion_3d_2d:
            inp = reshape_fields(
                self.files[year_idx][
                    (local_idx - self.dt * self.n_history) : (local_idx + 1) : self.dt,
                    self.in_channels,
                ],
                "inp",
                self.params,
                self.train,
                self.normalize,
                orog,
                self.add_noise,
            )
            tar = reshape_fields(
                self.files[year_idx][local_idx + step, self.out_channels],
                "tar",
                self.params,
                self.train,
                self.normalize,
                orog,
            )

            inp_3d_t = inp[:6, :, :]
            inp_3d_s = inp[6:12, :, :]
            inp_3d_u = inp[12:18, :, :]
            inp_3d_v = inp[18:24, :, :]
            inp_3d_t = inp_3d_t.unsqueeze(0)
            inp_3d_s = inp_3d_s.unsqueeze(0)
            inp_3d_u = inp_3d_u.unsqueeze(0)
            inp_3d_v = inp_3d_v.unsqueeze(0)
            inp_3d = torch.cat([inp_3d_t, inp_3d_s, inp_3d_u, inp_3d_v], axis=0)
            inp_3d = inp_3d.permute(0, 2, 3, 1)
            inp_2d = inp[24:, :, :]
            # print('inp: ', inp.shape)
            # print('inp_3d:', inp_3d.shape)
            # print('inp_2d:', inp_2d.shape)
            del inp, inp_3d_t, inp_3d_s, inp_3d_u, inp_3d_v
            return inp_2d, inp_3d, tar

        if self.params.multi_steps_finetune == 1:
            inp = reshape_fields(
                self.files[year_idx][
                    (local_idx - self.dt * self.n_history) : (local_idx + 1) : self.dt,
                    self.in_channels,
                ],
                "inp",
                self.params,
                self.train,
                self.normalize,
                orog,
                self.add_noise,
            )
            tar = reshape_fields(
                self.files[year_idx][local_idx + step, self.out_channels],
                "tar",
                self.params,
                self.train,
                self.normalize,
                orog,
            )
        else:
            inp = reshape_fields(
                self.files[year_idx][
                    (local_idx - self.dt * self.n_history) : (local_idx + 1) : self.dt,
                    self.in_channels,
                ],
                "inp",
                self.params,
                self.train,
                self.normalize,
                orog,
                self.add_noise,
            )
            tar = reshape_fields(
                self.files[year_idx][
                    local_idx
                    + step : local_idx
                    + step
                    + self.params.multi_steps_finetune,
                    self.in_channels,
                ],
                "inp",
                self.params,
                self.train,
                self.normalize,
                orog,
            )

        return inp, tar


def get_downstream_data_loader(
    params, backbone_files_pattern, downstream_files_pattern, distributed, train
):

    if params.downstream_config == "DownScalingNet":
        dataset = GetDataset_Kuroshio_Downscaling(
            params, backbone_files_pattern, downstream_files_pattern, train
        )
    if params.downstream_config == "WaveNet":
        dataset = GetDataset_Wave(
            params, backbone_files_pattern, downstream_files_pattern, train
        )
    if params.downstream_config == "BiochemicalNet":
        dataset = GetDataset_Biochemical(
            params, backbone_files_pattern, downstream_files_pattern, train
        )

    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None

    dataloader = DataLoader(
        dataset,
        batch_size=int(params.batch_size),
        num_workers=params.num_data_workers,
        shuffle=False,  # (sampler is None),
        sampler=sampler if train else None,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )  # pin_memory能加快内存的Tensor转义到GPU的显存的速度

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


class GetDataset_Wave(Dataset):
    def __init__(self, params, backbone_data_location, downstream_data_location, train):

        self.train = train
        self.params = params
        self.backbone_data_location = backbone_data_location
        self.downstream_data_location = downstream_data_location

        self.orography = params.orography
        self.normalize = params.normalize

        self.dt = params.dt
        self.n_history = params.n_history  # 0

        # backbone data channels
        self.in_channels = np.array(params.in_channels)
        self.out_channels = np.array(params.out_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)

        # downstream data channels
        self.downstream_in_channels = np.array(params.downstream_in_channels)
        self.downstream_out_channels = np.array(params.downstream_out_channels)
        self.downstream_force_channels = np.array(params.downstream_force_channels)
        self.downstream_n_in_channels = len(self.downstream_in_channels)
        self.downstream_n_out_channels = len(self.downstream_out_channels)

        self._get_backbone_files_stats()
        self._get_downstream_task_files_stats()

        self.add_noise = params.add_noise if train else False

    def _get_backbone_files_stats(self):
        self.files_paths = glob.glob(self.backbone_data_location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)

        with h5py.File(self.files_paths[0], "r") as _f:
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f["fields"].shape[0] - self.dt

            # original image shape (before padding)
            self.img_shape_x = (
                _f["fields"].shape[2] - 1
            )  # just get rid of one of the pixels
            self.img_shape_y = _f["fields"].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]

        logging.info(
            "Found backbone data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
                self.backbone_data_location,
                self.n_samples_total,
                self.img_shape_x,
                self.img_shape_y,
                self.n_in_channels,
            )
        )
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Delta t: {} days".format(self.dt))
        logging.info(
            "Including {} days of past history in training at a frequency of {} days".format(
                self.dt * self.n_history, self.dt
            )
        )

    def _get_downstream_task_files_stats(self):

        self.downstream_files_paths = glob.glob(self.downstream_data_location + "/*.h5")
        self.downstream_files_paths.sort()
        self.downstream_n_years = len(self.downstream_files_paths)

        with h5py.File(self.downstream_files_paths[0], "r") as _f:
            logging.info(
                "Getting downstream task file stats from {}".format(
                    self.downstream_files_paths[0]
                )
            )
            self.downstream_n_samples_per_year = _f["fields"].shape[0] - 1

            # original image shape (before padding)
            self.downstream_img_shape_x = (
                _f["fields"].shape[2] - 1
            )  # just get rid of one of the pixels
            self.downstream_img_shape_y = _f["fields"].shape[3]

        self.downstream_n_samples_total = (
            self.downstream_n_years * self.downstream_n_samples_per_year
        )
        self.downstream_files = [None for _ in range(self.downstream_n_years)]

        logging.info(
            "Found downstream task data at path {}. Number of examples: {}. Image Shape: {} x {}".format(
                self.downstream_data_location,
                self.downstream_n_samples_total,
                self.downstream_img_shape_x,
                self.downstream_img_shape_y,
            )
        )

        logging.info(
            "Number of downstream samples per year: {}".format(
                self.downstream_n_samples_per_year
            )
        )
        logging.info("Delta t: {} days".format(self.dt))
        logging.info(
            "Including {} days of past history in training at a frequency of {} days".format(
                self.dt * self.n_history, self.dt
            )
        )

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], "r")
        self.files[year_idx] = _file["fields"]

        if self.orography and self.params.normalization == "zscore":
            _orog_file = h5py.File(self.params.orography_norm_zscore_path, "r")
        if self.orography and self.params.normalization == "maxmin":
            _orog_file = h5py.File(self.params.orography_norm_maxmin_path, "r")
        self.orography_field = _orog_file["orog"]

    def _open_downstream_file(self, year_idx):
        _file = h5py.File(self.downstream_files_paths[year_idx], "r")
        self.downstream_files[year_idx] = _file["fields"]

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx = int(global_idx / self.n_samples_per_year)  # which year
        local_idx = int(global_idx % self.n_samples_per_year)  # which sample in a year

        if self.files[year_idx] is None:
            self._open_file(year_idx)
            self._open_downstream_file(year_idx)

        # If there are not enough historical time steps available in the features, shift to future time steps.
        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        # If the sample is the final one for the year, predict the current time step. Otherwise, predict the next time step.
        step = 0 if local_idx >= self.n_samples_per_year - self.dt else self.dt

        if self.orography:
            orog = self.orography_field
            if np.shape(orog)[0] == 721:
                orog = orog[0:720]
        else:
            orog = None

        inp = reshape_fields(
            self.files[year_idx][
                (local_idx - self.dt * self.n_history) : (local_idx + 1) : self.dt,
                self.in_channels,
            ],
            "inp",
            self.params,
            self.train,
            self.normalize,
            orog,
            self.add_noise,
        )
        tar = reshape_fields(
            self.files[year_idx][local_idx + step, self.out_channels],
            "tar",
            self.params,
            self.train,
            self.normalize,
            orog,
        )
        inp_downstream = reshape_downstream_fields(
            self.downstream_files[year_idx][
                (local_idx - self.dt * self.n_history) : (local_idx + 1) : self.dt,
                self.downstream_out_channels,
            ],
            "tar",
            self.params,
            self.normalize,
        )
        inp_wind_downstream = reshape_downstream_fields(
            self.downstream_files[year_idx][
                local_idx + step, self.downstream_force_channels
            ],
            "force",
            self.params,
            self.normalize,
        )
        inp_downstream = np.concatenate(
            (np.expand_dims(inp_downstream, 0), inp_wind_downstream), axis=0
        )

        tar_downstream = reshape_downstream_fields(
            self.downstream_files[year_idx][local_idx + step, self.downstream_out_channels],
            "tar",
            self.params,
            self.normalize,
        )

        return inp, tar, inp_downstream, tar_downstream


class GetDataset_Kuroshio_Downscaling(Dataset):
    def __init__(self, params, backbone_data_location, downstream_data_location, train):

        self.train = train
        self.params = params
        self.backbone_data_location = backbone_data_location
        self.downstream_data_location = downstream_data_location

        self.orography = params.orography
        self.normalize = params.normalize

        self.dt = params.dt
        self.n_history = params.n_history  # 0

        # backbone data channels
        self.in_channels = np.array(params.in_channels)
        self.out_channels = np.array(params.out_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)

        # downstream data channels
        self.downstream_in_channels = np.array(params.downstream_in_channels)
        self.downstream_out_channels = np.array(params.downstream_out_channels)
        self.downstream_n_in_channels = len(self.downstream_in_channels)
        self.downstream_n_out_channels = len(self.downstream_out_channels)

        self._get_backbone_files_stats()
        self._get_downstream_task_files_stats()

        self.add_noise = params.add_noise if train else False

    def _get_backbone_files_stats(self):
        self.files_paths = glob.glob(self.backbone_data_location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)

        with h5py.File(self.files_paths[0], "r") as _f:
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f["fields"].shape[0] - 1

            # original image shape (before padding)
            self.img_shape_x = _f["fields"].shape[2]
            self.img_shape_y = _f["fields"].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]

        logging.info(
            "Found backbone data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
                self.backbone_data_location,
                self.n_samples_total,
                self.img_shape_x,
                self.img_shape_y,
                self.n_in_channels,
            )
        )
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Delta t: {} days".format(self.dt))
        logging.info(
            "Including {} days of past history in training at a frequency of {} days".format(
                self.dt * self.n_history, self.dt
            )
        )

    def _get_downstream_task_files_stats(self):

        self.downstream_files_paths = glob.glob(self.downstream_data_location + "/*.h5")
        self.downstream_files_paths.sort()
        self.downstream_n_years = len(self.downstream_files_paths)

        with h5py.File(self.downstream_files_paths[0], "r") as _f:
            logging.info(
                "Getting downstream task file stats from {}".format(
                    self.downstream_files_paths[0]
                )
            )
            self.downstream_n_samples_per_year = _f["fields_0p08"].shape[0] - 1

            # original image shape (before padding)
            self.downstream_0p25_img_shape_x = _f["fields_0p25"].shape[2]
            self.downstream_0p25_img_shape_y = _f["fields_0p25"].shape[3]

            self.downstream_0p08_img_shape_x = _f["fields_0p08"].shape[2]
            self.downstream_0p08_img_shape_y = _f["fields_0p08"].shape[3]

        self.downstream_n_samples_total = (
            self.downstream_n_years * self.downstream_n_samples_per_year
        )
        self.downstream_files_0p08 = [None for _ in range(self.downstream_n_years)]
        self.downstream_files_0p25 = [None for _ in range(self.downstream_n_years)]

        logging.info(
            "Found downstream task data at path {}. Number of examples: {}. 0p25 Image Shape: {} x {}. 0p08 Image Shape: {} x {}".format(
                self.downstream_data_location,
                self.downstream_n_samples_total,
                self.downstream_0p25_img_shape_x,
                self.downstream_0p25_img_shape_y,
                self.downstream_0p08_img_shape_x,
                self.downstream_0p08_img_shape_y,
            )
        )

        logging.info(
            "Number of downstream samples per year: {}".format(
                self.downstream_n_samples_per_year
            )
        )
        logging.info("Delta t: {} days".format(self.dt))
        logging.info(
            "Including {} days of past history in training at a frequency of {} days".format(
                self.dt * self.n_history, self.dt
            )
        )

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], "r")
        self.files[year_idx] = _file["fields"]

        if self.orography and self.params.normalization == "zscore":
            _orog_file = h5py.File(self.params.orography_norm_zscore_path, "r")
        if self.orography and self.params.normalization == "maxmin":
            _orog_file = h5py.File(self.params.orography_norm_maxmin_path, "r")
        self.orography_field = _orog_file["orog"]

    def _open_downstream_file(self, year_idx):
        _file = h5py.File(self.downstream_files_paths[year_idx], "r")
        self.downstream_files_0p08[year_idx] = _file["fields_0p08"]
        self.downstream_files_0p25[year_idx] = _file["fields_0p25"]

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx = int(global_idx / self.n_samples_per_year)  # which year
        local_idx = int(global_idx % self.n_samples_per_year)  # which sample in a year

        if self.files[year_idx] is None:
            self._open_file(year_idx)
            self._open_downstream_file(year_idx)

        # If there are not enough historical time steps available in the features, shift to future time steps.
        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        # If the sample is the final one for the year, predict the current time step. Otherwise, predict the next time step.
        step = 0 if local_idx >= self.n_samples_per_year - self.dt else self.dt

        if self.orography:
            orog = self.orography_field
            if np.shape(orog)[0] == 721:
                orog = orog[0:720]
        else:
            orog = None

        inp = reshape_fields(
            self.files[year_idx][
                (local_idx - self.dt * self.n_history) : (local_idx + 1) : self.dt,
                self.in_channels,
            ],
            "inp",
            self.params,
            self.train,
            self.normalize,
            orog,
            self.add_noise,
        )
        tar = reshape_fields(
            self.files[year_idx][local_idx + step, self.out_channels],
            "tar",
            self.params,
            self.train,
            self.normalize,
            orog,
        )
        inp_downstream = reshape_downstream_fields(
            self.downstream_files_0p25[year_idx][
                local_idx + step, self.downstream_in_channels
            ],
            "inp",
            self.params,
            self.normalize,
        )
        tar_downstream = reshape_downstream_fields(
            self.downstream_files_0p08[year_idx][
                local_idx + step, self.downstream_out_channels
            ],
            "tar",
            self.params,
            self.normalize,
        )

        return inp, tar, inp_downstream, tar_downstream


class GetDataset_Biochemical(Dataset):
    def __init__(self, params, backbone_data_location, downstream_data_location, train):

        self.train = train
        self.params = params
        self.backbone_data_location = backbone_data_location
        self.downstream_data_location = downstream_data_location

        self.orography = params.orography
        self.normalize = params.normalize

        self.dt = params.dt
        self.n_history = params.n_history  # 0

        # backbone data channels
        self.in_channels = np.array(params.in_channels)
        self.out_channels = np.array(params.out_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)

        # downstream data channels
        self.downstream_in_channels = np.array(params.downstream_in_channels)
        self.downstream_out_channels = np.array(params.downstream_out_channels)
        self.downstream_n_in_channels = len(self.downstream_in_channels)
        self.downstream_n_out_channels = len(self.downstream_out_channels)

        self._get_backbone_files_stats()
        self._get_downstream_task_files_stats()

        self.add_noise = params.add_noise if train else False

    def _get_backbone_files_stats(self):
        self.files_paths = glob.glob(self.backbone_data_location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)

        with h5py.File(self.files_paths[0], "r") as _f:
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f["fields"].shape[0] - 1

            # original image shape (before padding)
            self.img_shape_x = _f["fields"].shape[2]
            self.img_shape_y = _f["fields"].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]

        logging.info(
            "Found backbone data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
                self.backbone_data_location,
                self.n_samples_total,
                self.img_shape_x,
                self.img_shape_y,
                self.n_in_channels,
            )
        )
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Delta t: {} days".format(self.dt))
        logging.info(
            "Including {} days of past history in training at a frequency of {} days".format(
                self.dt * self.n_history, self.dt
            )
        )

    def _get_downstream_task_files_stats(self):

        self.downstream_files_paths = glob.glob(self.downstream_data_location + "/*.h5")
        self.downstream_files_paths.sort()
        self.downstream_n_years = len(self.downstream_files_paths)

        with h5py.File(self.downstream_files_paths[0], "r") as _f:
            logging.info(
                "Getting downstream task file stats from {}".format(
                    self.downstream_files_paths[0]
                )
            )
            self.downstream_n_samples_per_year = _f["fields"].shape[0] - 1

            # original image shape (before padding)
            self.downstream_img_shape_x = _f["fields"].shape[2]
            self.downstream_img_shape_y = _f["fields"].shape[3]

        self.downstream_n_samples_total = (
            self.downstream_n_years * self.downstream_n_samples_per_year
        )
        self.downstream_files = [None for _ in range(self.downstream_n_years)]

        logging.info(
            "Found downstream task data at path {}. Number of examples: {}. Image Shape: {} x {}".format(
                self.downstream_data_location,
                self.downstream_n_samples_total,
                self.downstream_img_shape_x,
                self.downstream_img_shape_y,
            )
        )

        logging.info(
            "Number of downstream samples per year: {}".format(
                self.downstream_n_samples_per_year
            )
        )
        logging.info("Delta t: {} days".format(self.dt))
        logging.info(
            "Including {} days of past history in training at a frequency of {} days".format(
                self.dt * self.n_history, self.dt
            )
        )

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], "r")
        self.files[year_idx] = _file["fields"]

        if self.orography and self.params.normalization == "zscore":
            _orog_file = h5py.File(self.params.orography_norm_zscore_path, "r")
        if self.orography and self.params.normalization == "maxmin":
            _orog_file = h5py.File(self.params.orography_norm_maxmin_path, "r")
        self.orography_field = _orog_file["orog"]

    def _open_downstream_file(self, year_idx):
        _file = h5py.File(self.downstream_files_paths[year_idx], "r")
        self.downstream_files[year_idx] = _file["fields"]

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx = int(global_idx / self.n_samples_per_year)  # which year
        local_idx = int(global_idx % self.n_samples_per_year)  # which sample in a year

        if self.files[year_idx] is None:
            self._open_file(year_idx)
            self._open_downstream_file(year_idx)

        # If there are not enough historical time steps available in the features, shift to future time steps.
        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        # If the sample is the final one for the year, predict the current time step. Otherwise, predict the next time step.
        step = 0 if local_idx >= self.n_samples_per_year - self.dt else self.dt

        if self.orography:
            orog = self.orography_field
            if np.shape(orog)[0] == 721:
                orog = orog[0:720]
        else:
            orog = None

        inp = reshape_fields(
            self.files[year_idx][
                (local_idx - self.dt * self.n_history) : (local_idx + 1) : self.dt,
                self.in_channels,
            ],
            "inp",
            self.params,
            self.train,
            self.normalize,
            orog,
            self.add_noise,
        )
        tar = reshape_fields(
            self.files[year_idx][local_idx + step, self.out_channels],
            "tar",
            self.params,
            self.train,
            self.normalize,
            orog,
        )
        inp_downstream = reshape_downstream_fields(
            self.downstream_files[year_idx][
                (local_idx - self.dt * self.n_history) : (local_idx + 1) : self.dt,
                self.downstream_in_channels,
            ],
            "inp",
            self.params,
            self.normalize,
        )
        tar_downstream = reshape_downstream_fields(
            self.downstream_files[year_idx][local_idx + step, self.downstream_out_channels],
            "tar",
            self.params,
            self.normalize,
        )

        return inp, tar, inp_downstream, tar_downstream
