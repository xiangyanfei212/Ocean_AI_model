# Poseidon: A Unified Foundation Model for Ocean Dynamics System

This repository contains the code and resources for Poseidon, a novel AI foundation model for the global ocean system. Poseidon is designed to overcome the limitations of existing ocean simulation methods in terms of generalization and computational efficiency. By leveraging a Fourier-based Masked Autoencoder architecture, Poseidon learns general latent representations of ocean dynamics, enabling diverse oceanographic tasks with minimal fine-tuning.

## Overview
Poseidon consists of a pre-trained backbone model and downstream modules that address key oceanographic challenges, including:

1. Sparse Observation Simulation: Simulating global ocean variable fields from incomplete inputs.
2. Cross-Disciplinary Coupled Simulation: Driving ocean biogeochemical process predictions at low computational cost.
3. Multi-Modal State Decoding: Decoding complex ocean phenomena by fusing multi-source information like atmospheric and wave data.
4. Context-Aware Downscaling: Generating high-fidelity regional predictions by combining global features with local low-resolution information.

With its efficient design, Poseidon outperforms state-of-the-art baselines by 17.3% across key metrics for 30-day continuous simulations at a $1/4^{\circ}$ resolution. Furthermore, it achieves two orders of magnitude improvement in inference efficiency.

## Architecture
### Pre-Training Stage
Poseidon uses a Fourier-based Masked Autoencoder to learn general representations of ocean dynamics by reconstructing complete ocean states (e.g., sea temperature, salinity, velocity, and sea surface height) from masked, patched inputs of initial ocean conditions and atmospheric forcing.

### Downstream Tasks
The pre-trained backbone serves as a unified model for various downstream tasks:

1. Zero-shot Sparse Prediction: Reconstruct full fields from incomplete observations.
2. Cross-Disciplinary Coupling: Drive biogeochemical models at low computational cost.
3. Multi-Modal Decoding: Fuse atmospheric and wave data for complex state estimation.
4. Context-Aware Downscaling: Generate high-fidelity regional predictions.


## Dependencies
The following dependencies are required to run Poseidon:

- Python >= 3.8
- PyTorch >= 1.8.0
- NumPy
- h5py
- tqdm
- matplotlib
Additional libraries as specified in ```requirements.txt```
Install dependencies using:
```
pip install -r requirements.txt
```

## Data Sources
### Backbone Training Data
Poseidon is trained on a subset of the [HYCOM analysis (reanalysis)](https://www.hycom.org/dataserver/gofs-3pt1/analysis), a state-of-the-art global ocean analysis product from the Global Ocean Forecast System 3.1 (GOFS-3.1). 
In addition, external forcing and boundary conditions are incorporated, including topographic data from [ETOPO](https://doi.org/10.25921/fd45-gt74) and five atmospheric forcing variables from [ERA5](https://doi.org/10.24381/cds.bd0915c6).

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

### Downstream Tasks

1. Regional Downscaling: High-resolution training labels from HYCOM for the Kuroshio region.
2. Wave Decoding: Significant Wave Height (SWH) and wind components from [ERA5](https://doi.org/10.24381/cds.bd0915c6).
3. Biochemistry Coupling: Biogeochemical variables from the [NASA Ocean Biochemical Model](https://doi.org/10.5067/BHCFDIICIOU5).

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

## Data Organization
### Backbone Training Data

```
sample_backbone
|---train
|    |  2000.h5
|    |  2001.h5
|    |  ...
|---valid
|    |  2020.h5
|---test
|    |  2021.h5
|---stats
|    |  time_means.npy
|    |  global_means.npy  
|    |  global_stds.npy  
|    |  land_mask.h5
|    |  orography.h5
```


### Downstream Tasks
1. Regional Downscaling:
```
sample_Kuroshio_downscaling
|---train
|    |  2000.h5
|    |  ...
|---valid
|    |  2014.h5
|---test
|    |  2015.h5
|---stats
|    |  global_means.npy  
|    |  global_stds.npy  
|    |  topo_kuroshio_0p08.h5
```

2. Wave Decoding:
```
sample_wave
|---train
|    |  2000.h5
|    |  ...
|---stats
|    |  global_means.npy  
|    |  global_stds.npy  
|    |  land_mask.h5
```

3. Biochemistry Coupling:
```
sample_biochemical
|---train
|    |  2000.h5
|    |  ...
|---stats
|    |  global_means.npy  
|    |  global_stds.npy  
|    |  land_mask.h5
```

## Training
### Backbone Training
The backbone model can be trained using the provided configuration file (```config/config_backbone.yaml```) and the launch script (```submit_backbone_train.sh```).

### Downstream Tasks
The downstream model can be trained using the configuration file (```config/config_downstream.yaml```) and the launch script (```submit_downstream_train.sh```).

## Inference

### Backbone Model
To run inference with the backbone model:
1. Ensure the test data (```./sample_backbone/test/```) and normalization stats (```./sample_backbone/stats/```) are available.
2. Use the trained model weights hosted at Trained Model Weights.
3. Run the following command:
```
nohup python -u inference_backbone.py \
    --config='Masked_AE_Ocean' \
    --exp_dir='../exps' \
    --run_num='' \
    --finetune_dir='2_steps_finetune' \
    --prediction_length=31 \
    --decorrelation_time=30 \
    --n_samples_per_year=365 \
    --ics_type='datetime' \
    --date_strings='01/01/2021-00:00:00,...' \
    --year=2021 \
    > logs/inference.log 2>&1 &
```

### Downstream Modules
```
python inference_biochmical.py \
    --exp_dir='' \
    --prediction_length=31 \
    --decorrelation_time=30 \
    --n_samples_per_year=365
```
Modify the script based on the task (e.g., wave decoding or regional downscaling).

## Citation
@article{pathak2022fourcastnet,
  title={Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators},
  author={Pathak, Jaideep and Subramanian, Shashank and Harrington, Peter and ...},
  journal={arXiv preprint arXiv:2202.11214},
  year={2022}
}


















