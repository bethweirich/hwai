""" HWAI constants """

# Author: Elizabeth Weirich Benet 
# License: MIT License 2022

# Import packages
import xarray as xr
import numpy as np


#  **------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

# 1. Define dictionary entries (user tuning)
## 1.1. Parameters
### Define target_names
target_names_regr = ['t2m']
target_names_classi = ['hw_bin_1SD', 'hw_bin_15SD']
### Define lead times
lead_times = [1, 2, 3, 4, 5, 6]


## 1.2. Data & time periods
### Specify train_full, train, vali & test time period splits
splits_data = np.array([[1981, 2009], [1981, 2000], [2001, 2009], [2010, 2018]]).T
### Structure train-vali-test splits
splits = xr.DataArray(splits_data,
                        dims = ['edge', 'slice_type'],
                        coords = {'slice_type': ['train_full', 'train', 'vali', 'test'], 'edge': ['start', 'end']})
### Extract overall start and end years
start_year = int(splits.sel(slice_type = 'train_full', edge = 'start'))
end_year = int(splits.sel(slice_type = 'test', edge = 'end'))

### Cross-validation ('none', 'nested')
### 'none' has no loops but uses the train_full-test-train-vali partitions manually specified above instead
### 'nested' has an inner (train-validation) and an outer loop (train_full-test)
cv_type = 'none'
### Folds in outer and inner loop (only valid for nested CV)
if cv_type == 'none': num_outer_folds = None; num_inner_folds = None
elif cv_type == 'nested': num_outer_folds = 5; num_inner_folds = 2


## 1.3. Constants
### Select temporal resolution
timestep = '7D'
### Simple or rolling mean? ('simple', 'rolling')
mean_type = 'rolling'
### Define number of lags you want to have in the predictors e.g. for num_lags = 3, (lead_time, lead_time + 2 weeks before the 
## prediction date) will be taken into account
num_lags = 4
### Select season
initial_month = 5
final_month = 9


## 1.4. Actions
### Calibrate probabilistic classification forecast via Platt scaling (True) or use uncalibrated forecast (False)?
calibrate_linear = False
calibrate_rf = True

### Full run (True) or fast run (False; makes 2-member ensemble for the ML models and for choosing the probability threshold, faster but WRONG)
full_run = True
### Do you want to run preprocessing part 2 again (True) or use the metrics saved in the last run for plotting only (False)?
### prepro2 = False can only be run after a complete run (all lead times) with prepro2 = True 
prepro2 = True
### Do you want to train the ML models and predict the test set again (True) or read the ML model's forecasts of the test set from the last run (False)?
### train_models = False can only be run after a complete run (all lead times) with train_models = True 
### prepro2 contains train_models (i.e., train_models = True only works if prepro2 = True) 
train_models = True

### Do you want to optimize the hyperparameters for the linear models (True) or use default hyperparameters (False)? 
### The linear model hyper-parameter optimization is fast to run (on the order of <1min per model)
optimize_linear_hyperparam = True
### Do you want to optimize hyperparameters for the RF models (True) or use the set of best hyperparameters saved in the last run (False)? 
### The hyper-parameters provided on GitHub are optimized through exhaustive grid search (Table C1 in [1])
### If executer.py is run with optimize_rf_hyperparam = True, these hyper-parameters will be overwritten. The exhaustive RF hyper-parameter optimization 
### might run for several hours on your server.
optimize_rf_hyperparam = False
### Search type ('rdm', 'exhaustive') - only valid if optimize_rf_hyperparam is True 
### 'exhaustive' explores all possible hyperparameter combinations in the grid
### 'rdm' explores a only a subset of the possible hyperparameter combinations
hp_search_type = 'rdm'
### Specify number of hyperparameter combinations to try out (only valid for rdm search)
num_hp_set_candidates = 2

### Metrics to optimize 
#### Regression ('RMSE', 'Corr')
metric_regr = 'RMSE'
#### Classification ('ROC AUC', 'BS')
metric_classi = 'BS'
#### Probability threshold selection ('B': frequency bias closest to 1, 'TS': max threat score)
metric_th_sel = 'B'

### Should the figures show and be saved with a title?
plot_fig_title = True


## 1.5. Verbosity level
###        0 : Nothing printed nor plotted
###        1 : Section & subsection titles (highlighted) printed, tables and plots shown
###        2 : Add subtitles (****), coefficients, importances and best hyperparameters
###        3 : Add comments
###        4 : Add hyperparameter optimization steps, probability threshold optimization plots, ensemble members plots, & print of lists and numbers 
###        5 : Add dataset prints & EOF plots
verbosity = 4


## 1.6. Machinery
### Number of cores (set to -1 to use all available cores in your server)
n_cores = 30


## 1.7. Paths
### Path to raw data
#### For ETHZ IAC users: Before running preprocessing_part1.py, set path_raw_data = '/net/iacftp/ftp/pub_read/weiriche/hwai_raw_data/' below
#### For external users: Before running preprocessing_part1.py, download the raw data (listed in Table 1 in [1]) from the FTP server (see README.md 
#### file) and specifiy the path to the folder containing the files z500_eraint_daymean_1979-2019.nc, sm_era5land_daymean_1981-2019.nc, 
#### sst_hadisst_monmean_1870-2019.nc, rain_eobs_daymean_1950-2020.nc, and t2m_eobs_daymean_1950-2020.nc under path_raw_data:
path_raw_data = None
### Path to preprocessed data
path_data = '../data/'
### Path to metrics
path_metrics = '../model_output/metrics/'
### Path to best hyperparameters
path_hyperparam = '../model_output/hyperparameters/'
### Path to time series
path_time_series = '../model_output/time_series/'
### Path to plots
path_plots = '../model_output/plots/'
### Path to feature importances and regression coefficients
path_features = '../model_output/features/'


## 1.8. Latitude-longitude boxes
### Central Europe (CE) [45°N,55°N], [5°,15°E]
ce = np.array([[55.,45.], [5.,15.]]).T
ce_obs = np.array([[44.875,54.875], [4.875,14.875]]).T
### North Atlantic [40°S,70°N], [90°W,30°E]
na = np.array([[70.,40.], [-90., 30.]]).T
### North western Mediterranean (NWMED) [35°N,45°N], [0°,15°E]
nwmed = np.array([[45.,35.], [0.,15.]]).T
### Cold North Atlantic anomaly (CNAA) [45°N,60°N], [15°W,40°W]
cnaa = np.array([[60.5,45.5], [-40., -15.]]).T
### Structure latitude-longitude boxes
boxes = xr.concat(
                  [xr.DataArray(
                                X,
                                dims = ['edge','axis'],
                                coords = {'axis': ['latitude', 'longitude'], 'edge': ['start', 'end']})
                                for X in (ce, ce_obs, na, nwmed, cnaa)],
                                dim = 'box').assign_coords(box = ['ce', 
                                                                  'ce_obs', 
                                                                  'na', 
                                                                  'nwmed', 
                                                                  'cnaa']
                    )


## 1.9. Long names and units
long_box_names = {'ce': 'Central Europe', 
                  'ce_obs': 'Central Europe E-OBS', 
                  'na': 'North Atlantic', 
                  'nwmed': 'Northwestern Mediterranean', 
                  'cnaa': 'Cold North Atlantic anomaly'
                 }
long_predictor_names = {'t2m_x': 'Temperature',
                            'z': 'Geopotential',
                            'rain': 'Precipitation', 
                            'sm': 'Soil moisture', 
                            'sea': 'Summer East Atlantic pattern',
                            'sst_cnaa': 'Cold North Atlantic anomaly SST', 
                            'sst_nwmed': 'Northwestern Mediterranean SST'
                        }
units = {'t2m': '^0C',
         't2m_x': '^0C',
         'z': 'm^2/s^2',
         'rain': 'mm', 
         'sm': 'm^3/m^3', 
         'sea': 'm^2/s^2',
         'sst_cnaa': '^0C', 
         'sst_nwmed': '^0C'
        }

#  **------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

# 2. Save definitions (fix)
dictionary = { 
              'target_names': target_names_regr + target_names_classi,
              'target_names_regr': target_names_regr,
              'target_names_classi': target_names_classi,
              'lead_times': lead_times,
              'cv_type': cv_type,
              'hp_search_type': hp_search_type,
              'num_hp_set_candidates': num_hp_set_candidates,
              'num_outer_folds': num_outer_folds,
              'num_inner_folds': num_inner_folds,
              'timestep': timestep,
              'timestep_num': int(timestep[:-1]),
              'num_lags': num_lags,
              'initial_month': initial_month,
              'final_month': final_month,
              'splits': splits,
              'start_year': start_year,
              'end_year': end_year,
              'boxes': boxes,
              'mean_type': mean_type,
              'calibrate_linear': calibrate_linear,
              'calibrate_rf': calibrate_rf,
              'full_run': full_run, 
              'prepro2': prepro2,
              'train_models': train_models,
              'optimize_linear_hyperparam': optimize_linear_hyperparam,
              'optimize_rf_hyperparam': optimize_rf_hyperparam,
              'metric_regr': metric_regr,
              'metric_classi': metric_classi,
              'metric_th_sel': metric_th_sel,
              'plot_fig_title': plot_fig_title,
              'n_cores': n_cores,
              'path_raw_data': path_raw_data,     
              'path_data': path_data, 
              'path_metrics': path_metrics,
              'path_hyperparam': path_hyperparam,
              'path_time_series': path_time_series,
              'path_plots': path_plots,
              'path_features': path_features,
              'verbosity': verbosity,
              'long_box_names': long_box_names,
              'long_predictor_names': long_predictor_names,  
              'units': units
               }


