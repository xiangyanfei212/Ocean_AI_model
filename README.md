# Ocean_AI_model

This repository contains the code used for "Poseidon: A Unified Foundation Model for Ocean Dynamics System" \[[paper]()\]

Existing ocean simulation methods face significant bottlenecks in generalization and computational efficiency. To address this challenge, we propose Poseidon, a novel AI foundation model for the global ocean system. The core of Poseidon is a Fourier-based Masked Autoencoder. This model processes patched and serialized ocean state and atmospheric forcing fields through self-supervised pre-training to learn general latent representations of ocean dynamics system. This pre-trained model serves as a unified backbone, efficiently empowering diverse downstream oceanographic tasks with minimal, lightweight fine-tuning. These tasks include: **Sparse Observation Simulation**, which accurately simulates global ocean variable fields from incomplete inputs; **Cross-Disciplinary Coupled Simulation**, which drives ocean biogeochemical process predictions at low cost by coupling the pre-trained physical fields with lightweight modules; **Multi-Modal State Decoding**, which decodes complex ocean phenomena by fusing multi-source information like atmospheric and wave data; and **Context-Aware Downscaling**, which generates high-fidelity regional predictions by combining global features with local low-resolution information. On the 30-day continuous simulation task for global ocean core variables at a $1/4^{\circ}$ resolution and 15 depth layers, Poseidon outperforms state-of-the-art baselines by an average of 17.3% across all key metrics. It also improves inference efficiency by two orders of magnitude. Our work provides a critical paradigm and a solid foundation for building scalable and efficient AI foundation models in the Earth system sciences.

Poseidon is based on the vision transformer architecture with Adaptive Fourier Neural Operator (AFNO) attention proposed in Guibas-Mardani et al. \[[paper](https://openreview.net/pdf?id=EXHG-A3jlM)\], \[[code](https://github.com/NVlabs/AFNO-transformer)\]. # TODO: add mask


## Training:

### Backbone model

The backbone model is trained on a subset of [HYCOM analysis(reanalysis)](https://www.hycom.org/dataserver/gofs-3pt1/analysis), which is a state-of-the-art global eddy-resolving ocean analysis(reanalysis) product generated via the Global Ocean Forecast System 3.1 (GOFS-3.1). 

Our backbone model is designed to predict key ocean variables, including sea surface height (SSH), temperature (T), salinity (S), and the u- and v-components of ocean velocity (U and V), on a daily basis. We preprocess the HYCOM analysis(reanalysis) datasets from 2000 to 2021 at a spatial resolution of $1/4^{\circ}$. We selected 15 depth levels (0 m, 6 m, 10 m, 20 m, 30 m, 50 m, 70 m, 100 m, 125 m, 150 m, 200 m, 250 m, 300 m, 400 m, and 500 m) for T, S, U, and V. We sample daily subset at 12:00. 

In addition, external forcing and boundary conditions are incorporated, including topographic data from ETOPO~\cite{NOAA_ETOPO2022} and five atmospheric forcing variables from ERA5~\cite{hersbach2020era5}.

The variables used in backbone are as follows:

| Variable   | Layer | Dimension       | Details                                                                 | Source |
|------------|-------|-----------------|-------------------------------------------------------------------------|--------|
| T          | 15    | (15, 720, 1440) | Sea temperature (°C)                                                   | HYCOM  |
| S          | 15    | (15, 720, 1440) | Sea Salinity (PSU)                                                     | HYCOM  |
| U          | 15    | (15, 720, 1440) | Sea stream zonal velocity (m/s)                                        | HYCOM  |
| V          | 15    | (15, 720, 1440) | Sea stream meridional velocity (m/s)                                   | HYCOM  |
| SSH        | 1     | (1, 720, 1440)  | Sea surface height (m)                                                 | HYCOM  |
| u₁₀        | 1     | (1, 720, 1440)  | Eastward component of the wind at 10m above surface (m/s)              | ERA5   |
| V₁₀        | 1     | (1, 720, 1440)  | Northward component of the wind at 10m above surface (m/s)             | ERA5   |
| T2m        | 1     | (1, 720, 1440)  | Air temperature at 2m above the surface (°C)                           | ERA5   |
| MSL        | 1     | (1, 720, 1440)  | Mean surface level (m)                                                 | ERA5   |
| SP         | 1     | (1, 720, 1440)  | Surface pressure (hPa)                                                 | ERA5   |
| Topology   | 1     | (1, 720, 1440)  | Digital elevation and ocean bathymetry (m)                             | ETOPO  |

We divide the dataset into three subsets: training, validation, and testing. The training dataset spans the period from 2000 to 2019, the validation dataset consists of data from 2020, and the testing dataset includes out-of-sample data from 2021. 

[Pre-processed Training Data]()

The data directory is organized as follows:

```
sample_backbone
|---train
|    |  2000.h5
|    |  2001.h5
|    |  ...
|    |  ...
|    |  2019.h5
|
|---valid
|    |  2020.h5
|
|---test
|    |  2021.h5
|
|---stats
|    |  time_means.npy
|    |  global_means.npy  
|    |  global_stds.npy  
|    |  land_mask.h5
|    |  orography.h5 
```

Training configurations can be set up in [config/AFNO.yaml](config/backbone.yaml).

An example launch script for distributed data parallel training on the slurm based HPC cluster perlmutter is provided in ```submit_backbone_train.sh```. Please follow the pre-training and fine-tuning procedures as described in the pre-print.

### Downstream task

The variables and datasets source used in downstream tasks are as follows:

| Variable | Resolution      | Details                                                   | Task          | Source |
|----------|-----------------|-----------------------------------------------------------|---------------|--------|
| SST      | 1/12°          | Sea surface temperature (°C)                              | Downscaling   | HYCOM  |
| SSS      | 1/12°          | Sea surface salinity (PSU)                                | Downscaling   | HYCOM  |
| SSU      | 1/12°          | Sea surface stream zonal velocity (m/s)                  | Downscaling   | HYCOM  |
| SSV      | 1/12°          | Sea surface stream meridional velocity (m/s)             | Downscaling   | HYCOM  |
| SSH      | 1/12°          | Sea surface height (m)                                    | Downscaling   | HYCOM  |
| SWH      | 1/2°           | Significant wave height (m)                               | Wave          | ERA5   |
| U₁₀      | 1/2°           | 10m U wind component (m/s)                                | Wave          | ERA5   |
| V₁₀      | 1/2°           | 10m V wind component (m/s)                                | Wave          | ERA5   |
| Tca      | 1°             | Total chlorophyll a concentration (mg/m³)                | Biochemistry  | NASA   |
| Chl      | 1°             | Chlorophyte concentration (mg/m³)                        | Biochemistry  | NASA   |
| Dia      | 1°             | Diatom concentration (mg/m³)                             | Biochemistry  | NASA   |
| Coc      | 1°             | Coccolithophores concentration (mg/m³)                   | Biochemistry  | NASA   |
| Cya      | 1°             | Cyanobacteria concentration (mg/m³)                      | Biochemistry  | NASA   |
| Irn      | 1°             | Iron concentration (nano mole/L)                         | Biochemistry  | NASA   |
| Nit      | 1°             | Nitrate concentration (micro mole/L)                     | Biochemistry  | NASA   |
| MLD      | 1°             | Mixed layer depth (m)                                     | Biochemistry  | NASA   |

An example launch script for distributed data parallel training on the slurm based HPC cluster perlmutter is provided in ```submit_downstream_train.sh```.

#### Regional Downscaling

The regional downscaling module focuses on the Kuroshio region, defined by the geographical range of $124.7^{\circ}$E to $180^{\circ}$ and $21.28^{\circ}$N to $45^{\circ}$N. The high-resolution training labels are from the HYCOM analysis and reanalysis data set at a spatial resolution of $1/12^{\circ}$.

The data directory is organized as follows:

```
sample_Kuroshio_downscaling
|---train
|    |  2000.h5
|    |  2001.h5
|    |  ...
|    |  ...
|    |  2013.h5
|
|---valid
|    |  2014.h5
|
|---test
|    |  2015.h5
|
|---stats
|    |  global_means.npy  
|    |  global_stds.npy  
|    |  land_mask.h5
|    |  topo_kuroshio_0p08.h5
```

#### Wave Decoding

The goal of this module is to predict significant wave height (SWH) using physical ocean variables and atmospheric drivers. SWH data and 10m U/V wind components are taken from the ERA5 dataset~\cite{hersbach2020era5}. 

The data directory is organized as follows:

```
sample_wave
|---train
|    |  2000.h5
|    |  2001.h5
|    |  ...
|    |  ...
|    |  2013.h5
|
|---valid
|    |  2014.h5
|
|---test
|    |  2015.h5
|
|---stats
|    |  global_means.npy  
|    |  global_stds.npy  
|    |  land_mask.h5
```

#### Biochemistry Coupling

The Biochemistry Coupling Module is trained on eight biochemical variables obtained from the NASA Ocean Biochemical Model~\cite{gregg2017nasa}, which integrates satellite chlorophyll data with ocean circulation-biochemical coupled numerical models. These variables include total chlorophyll a concentration, chlorophyte concentration, diatom concentration, coccolithophore concentration, cyanobacteria concentration, iron concentration, nitrate concentration, and mixed layer depth. Each variable is interpolated onto a global regular grid with dimensions of $180 \times 360$ using the bilinear interpolation method.

The data directory is organized as follows:

```
sample_biochemical
|---train
|    |  2000.h5
|    |  2001.h5
|    |  ...
|    |  ...
|    |  2013.h5
|
|---valid
|    |  2014.h5
|
|---test
|    |  2015.h5
|
|---stats
|    |  global_means.npy  
|    |  global_stds.npy  
|    |  land_mask.h5
```

## Inference:

In order to run Poseidon’s backbone and downstream model in inference mode you will need to have the following files on hand.

1. The path to the out of training sample hdf5 file.
2. The model weights hosted at [Trained Model Weights]()
3. The pre-computed normalization statistics hosted at [additional]().

Run inference for backbone using

```
nohup python -u inference_backbone.py \
    --config='Masked_AE_Ocean' \  # Model configuration
    --exp_dir='../exps' \  # Path to the experiment directory (./[exp_dir]/[run_num])
    --run_num='' \
    --finetune_dir='2_steps_finetune' \  # Directory containing fine-tuned weights
    --prediction_length=31 \  # Prediction length in timesteps
    --decorrelation_time=30 \  # Decorrelation time interval
    --n_samples_per_year=365 \  # Number of samples per year for evaluation
    --ics_type='datetime' \  # Type of initial conditions (datetime-based)
    --date_strings='01/01/2021-00:00:00,01/02/2021-00:00:00,01/03/2021-00:00:00,01/04/2021-00:00:00,01/05/2021-00:00:00,01/06/2021-00:00:00,01/07/2021-00:00:00,01/08/2021-00:00:00,01/09/2021-00:00:00,01/10/2021-00:00:00,01/11/2021-00:00:00,01/12/2021-00:00:00' \  
    --year=2021 \  # Year for which predictions are being generated
    > logs/inference_025_backbone_20240223-100516.log 2>&1 &
```

Run inference for downstream using

```
python inference_biochmical.py \     # you can modify the inference code (inference_biochmical.py, inference_wave.py, inference_kuroshio_downscaling.py)
    --exp_dir='' \                   # Path to the experiment directory
    --prediction_length=31 \         # Length of predictions (e.g., 31 days)
    --decorrelation_time=30 \        # Time interval for decorrelation
    --n_samples_per_year=365         # Number of samples to evaluate per year
```

## References:

ERA5 data \[ Hersbach, H. et al., (2018) \] was downloaded from the Copernicus Climate Change Service (C3S) Climate Data Store.

```
Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2018): ERA5 hourly data on pressure levels from 1959 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). , 10.24381/cds.bd0915c6

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2018): ERA5 hourly data on single levels from 1959 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). , 10.24381/cds.adbb2d47
```

If you find this work useful, cite it using:
```
@article{pathak2022fourcastnet,
  title={Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators},
  author={Pathak, Jaideep and Subramanian, Shashank and Harrington, Peter and Raja, Sanjeev and Chattopadhyay, Ashesh and Mardani, Morteza and Kurth, Thorsten and Hall, David and Li, Zongyi and Azizzadenesheli, Kamyar and Hassanzadeh, Pedram and Kashinath, Karthik and Anandkumar, Animashree},
  journal={arXiv preprint arXiv:2202.11214},
  year={2022}
}
```






















