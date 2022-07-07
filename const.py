#!/usr/bin/env python
# coding: utf-8

# # Constants

# In[53]:


# Import packages
import xarray as xr
import numpy as np


# ## 1. Dictionary entries
# ### 1.1. Parameters

# In[54]:


## Define target_names
target_names_regr = ['t2m']
target_names_classi = ['hw_bin_15SD', 'hw_bin_1SD']
## Define lead times (in units of the user-defined timestep)
lead_times = [1, 2, 3, 4, 5, 6]

## Define different balancing possibilities ('undersampling', 'oversampling', 'none')
balance_types = ['undersampling', 'oversampling']


# ### 1.2. Data & time periods

# In[55]:


## Select dataset ('combi', 'ERA5', 'ERA20C')
### combi available [1981, 2018]
### ERA5 available [1950, 2020]
### ERA20C available [1900, 2009]
dataset = 'combi'

## Specify train_full, train, vali & test time period splits for each dataset
splits_combi = np.array([[1981, 2009], [1981, 2000], [2001, 2009], [2010, 2018]]).T
splits_ERA5 = np.array([[1950, 2000], [1950, 1990], [1991, 2000], [2001, 2020]]).T
splits_ERA20C = np.array([[1900, 1989], [1900, 1974], [1975, 1989], [1990, 2009]]).T

## Cross-validation ('none', 'nested')
### 'none' has no loops but uses the train_full-test-train-vali partitions manually specified above instead
### 'nested' has an inner (train-validation) and an outer loop (train_full-test)
cv_type = 'none'
### Folds in outer and inner loop
if cv_type == 'none': num_outer_folds = None; num_inner_folds = None
elif cv_type == 'nested': num_outer_folds = 5; num_inner_folds = 2


# ### 1.3. Constants

# In[56]:


## Select temporal resolution (in days e.g., '5D' or months e.g., '1M')
timestep = '7D'

## Simple or rolling mean? ('simple', 'rolling')
mean_type = 'rolling'

## Define number of lags you want to have in the predictors e.g. for num_lags = 3, (lead_time, lead_time + 2 weeks before the 
## prediction date) will be taken into account
num_lags = 4
## Add lower resolution (x 2,3,4) features for soil moisture and SST predictors
all_lags_and_means = False

## Select season
initial_month = 5
final_month = 9

## Percentile for extreme day definition (only releavant for the 2in7 index)
percentile = 0.95

## Minimum consecutive extreme days to be considered a heat wave (only relevant for the 2in7 index)
min_duration = 3

## Geopotential level in hPa
geopotential_level = '500.'


# ### 1.4. Actions

# In[57]:


## Drop predictors? If True: the 11 predictors out of 18 that show lower lagged correlation (selection has been done manually) are dropped
### Rem: regularized linear models benefit from being fed with all 18 predictors, while RFs work better when given only the 7 most relevant ones
### Feed them with a different dataset?
drop_predictors = True

## Feature selection? If True: feature selection with Random Forest to keep features whose importances sum 0.9
feature_selection = False
min_sum_importances = 0.9

## Dimensionality reduction? If True: Principal Component Analysis (PCA)
pca = False
min_cum_exp_var = 0.9

## Do you want to run preprocessing 2 again or use saved values?
prepro2 = True
## Do you want to train the ML models again (True) or read the ML model's forecasts from memory (False)?
train_models = True

## Do you want to optimize hyperparameters (True) or use the last saved set of hyperparameters to train the ML model (False)? 
optimize_rf_hyperparam = False
## Search type ('rdm', 'exhaustive')
### 'exhaustive' explores all possible hyperparameter combinations in the grid
### 'rdm' explores a only a subset of the possible hyperparameter combinations
hp_search_type = 'exhaustive'
### Specify number of hyperparameter combinations to try out (only valid for rdm search)
num_hp_set_candidates = 10

## Apply regularization?
### Regression
reg_regr = False
### Classification
# True per default

## Metric to optimize 
### Regression ('RMSE', 'Corr', 'RMSE_Corr')
metric_regr = 'RMSE'
### Classification
metric_classi = 'ROC AUC'

## Do you want to explore overfitting? Plots the metric on the validation set against the model complexity
explore_overfitting = False

## Do you want to print a Decission Tree?
show_tree = False

## Should the figures show and be saved with a title?
plot_fig_title = False


# ### 1.5. Verbosity

# In[58]:


## Choose verbosity level
###        0 : Nothing printed nor plotted
###        1 : Section & subsection titles (highlighted) printed, tables and plots shown
###        2 : Add subtitles (****), coefficients, importances and best hyperparameters
###        3 : Add comments
###        4 : Add hyperparameter optimization steps & print of lists and numbers 
###        5 : Add dataset prints & scatter plots & EOF plots
verbosity = 4


# ### 1.6. Machinery

# In[59]:


## Run on... ('CFC', 'Euler')
server = 'CFC'

## Number of cores
if server == 'CFC': n_cores = 30
elif server == 'Euler': n_cores = -1


# ### 1.7. Paths

# In[60]:


if server == 'CFC':
    ## Paths to data sources
    ### Reanalysis
    path_eraint = '/s2s/shared_data/Datasets/ERA-Int/'
    path_era5_land = '/s2s/weiriche/heat_waves/reanalysis/ERA5-Land/'
    path_hadisst = '/s2s/shared_data/Datasets/HadISST/'
    path_bernatj_data = '/home/bernatj/data/'
    path_eobs = '/s2s/shared_data/Datasets/E-OBS/'
    ### Forecasts
    path_ecmwf = '/s2s/weiriche/heat_waves/s2s_forecasts/ECMWF_2020_runs/'

    ## Paths to datasets with predictors
    path_input_orig_pred = '/s2s/weiriche/heat_waves/inputs/predictors/original/'
    path_input_prepro_pred = '/s2s/weiriche/heat_waves/inputs/predictors/preprocessed/'

    ## Path to target heat wave data
    path_input_targ = '/s2s/weiriche/heat_waves/inputs/targets/'

    ## Path to metrics
    path_metrics = '/s2s/weiriche/heat_waves/metrics/'

    ## Path to results
    path_hyperparam = '/s2s/weiriche/heat_waves/results/hyperparameters/'
    path_time_series = '/s2s/weiriche/heat_waves/results/time_series/'

    ## Path to plots
    path_plots_prediction = '/home/weiriche/heat_waves/plots/Prediction/'

elif server == 'Euler':
    # Path to CFC on Euler
    path_cfc_on_euler = '/cluster/home/weiriche/heat_waves_cfc/' 
    
    ## Paths to data sources
    ### Reanalysis (none, since Euler will not be used for the preprocessing, but for running the prediction algorithms)
    path_eraint = None
    path_era5_land = None
    path_hadisst = None
    path_bernatj_data = None
    path_eobs = None
    ### Forecasts
    path_ecmwf = path_cfc_on_euler + 's2s_forecasts/ECMWF_2020_runs/'

    ## Paths to datasets with predictors
    path_input_orig_pred = None
    path_input_prepro_pred = path_cfc_on_euler + 'inputs/predictors/preprocessed/'

    ## Path to target heat wave data
    path_input_targ = path_cfc_on_euler + 'inputs/targets/'

    ## Path to metrics
    path_metrics = path_cfc_on_euler + 'metrics/'

    ## Path to results
    path_hyperparam = path_cfc_on_euler + 'results/hyperparameters/'
    path_time_series = path_cfc_on_euler + 'results/time_series/'

    ## Path to plots
    path_plots_prediction = path_cfc_on_euler + 'plots/Prediction/'


# ### 1.8. Grid point boxes

# In[61]:


# Central Europe (CE) [45°N,55°N], [5°,15°E]
ce = np.array([[55.,45.], [5.,15.]]).T
ce_obs = np.array([[44.875,54.875], [4.875,14.875]]).T

# Central Europe Large (CE_large) [30°N,70°N], [50°W,60°E]
ce_large_left = np.array([[70.,30.], [310.,357.5]]).T
ce_large_right = np.array([[70.,30.], [0.,60.]]).T
ce_obs_large = np.array([[29.875,69.875], [-50.125,59.875]]).T

# Mediterranean Sea (MED) [30°N,45°N], [0°,25°E]
med = np.array([[45.,30.], [0.,25.]]).T
## North western box (NWMED) [35°N,45°N], [0°,15°E]
nwmed = np.array([[45.,35.], [0.,15.]]).T
## South eastern box (SEMED) [30°N,40°N], [15°,35°E]
semed = np.array([[40.,30.], [15.,35.]]).T

# Northern Atlantic (NA)
## Baltic [50°N,65°N], [0°W,25°W]
baltic = np.array([[65.5,50.5], [-24.5, 0.5]]).T
## Cold blob box (CBNA) [45°N,60°N], [15°W,40°W]
cbna = np.array([[60.5,45.5], [-40., -15.]]).T
## North western box (NWNA) [42.5°N,52.5°N], [52.5°W,40°W]
nwna = np.array([[52.5,42.5], [-52.5, -40.]]).T
## South eastern box (SENA) [35°N,42.5°N], [35°W,20°W]
sena = np.array([[42.5,35.], [-35., -20.]]).T

# Pacific
## El Niño Southern Oscillation (ENSO) [5°S,5°N], [120°W, 170°W]
enso = np.array([[5.,-5.], [-170., -120.]]).T
## Pacific Region [10°N, 20°N],[ 180°, 110°W]
pac = np.array([[20.,10.], [180., 250.]]).T
## Caribbean Region [10°N, 25°N] [85°W, 65°W]
carib = np.array([[25.,10.], [275., 295.]]).T


# ### 1.9. Long predictor names

# In[62]:


long_predictor_names = {'t2m_x': 'Temperature',
                            'z': 'Geopotential height',
                            'rain': 'Precipitation', 
                            'sm': 'Soil moisture', 
                            'sm_deeper': 'Deeper soil moisture', 
                            'sm_deepest': 'Deepest soil moisture', 
                            'snao': 'Summer North Atlantic Oscillation', 
                            'sea': 'Summer East Atlantic pattern',
                            'sst_med': 'Mediterranean SST',
                            'sst_nwmed': 'North-western Mediterranean SST', 
                            'sst_semed': 'South-eastern Mediterranean SST', 
                            'sst_baltic': 'Baltic SST', 
                            'sst_cbna': 'Cold blob North Atlantic SST', 
                            'sst_nwna': 'North-western North Atlantic SST', 
                            'sst_sena': 'South-eastern North Atlantic SST',
                            'na_index': 'North Atlantic index',
                            'sst_enso': 'El Ninyo Southern Oscillation', 
                            'pcd_index': 'Pacific Caribbean Dipole index'
                        }
units = {'t2m': '^0C',
         't2m_x': '^0C',
         'z': 'm^2/s^2',
         'rain': 'mm', 
         'sm': 'm^3/m^3', 
         'sm_deeper': 'm^3/m^3', 
         'sm_deepest': 'm^3/m^3', 
         'snao': 'm^2/s^2', 
         'sea': 'm^2/s^2',
         'sst_med': '^0C',
         'sst_nwmed': '^0C', 
         'sst_semed': '^0C', 
         'sst_baltic': '^0C', 
         'sst_cbna': '^0C', 
         'sst_nwna': '^0C', 
         'sst_sena': '^0C',
         'na_index': '^0C',
         'sst_enso': '^0C', 
         'pcd_index': 'm'
                        }


#  **------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

#  **------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

# ## 2. Save definitions

# ### 2.1. Structre train-vali-test splits

# In[2]:


# Build slices into 3D xarray
splits = xr.concat(
    [xr.DataArray(
            X,
            dims = ['edge', 'slice_type'],
            coords = {'slice_type': ['train_full', 'train', 'vali', 'test'], 'edge': ['start', 'end']})
            for X in (splits_combi, splits_ERA5, splits_ERA20C)],
            dim = 'dataset').assign_coords(dataset = ['combi', 'ERA5',  'ERA20C'])


# ### 2.2. Structure boxes

# In[3]:


# Build boxes into 3D xarray
boxes = xr.concat(
    [xr.DataArray(
            X,
            dims = ['edge','axis'],
            coords = {'axis': ['latitude', 'longitude'], 'edge': ['start', 'end']})
            for X in (ce, ce_obs, ce_large_left, ce_large_right, ce_obs_large, med, nwmed, semed, baltic, cbna, nwna, sena, enso, pac, carib)],
            dim = 'box').assign_coords(box = ['ce', 'ce_obs',  'ce_large_left', 'ce_large_right', 'ce_obs_large', 'med', 'nwmed', 'semed', 
                                              'baltic', 'cbna', 'nwna', 'sena', 'enso', 'pac', 'carib'])


# ### 2.3. Adapt paths depending on time period and dataset

# In[66]:


# Compute overall start and end years
start_year = int(splits.sel(dataset = dataset, slice_type = 'train_full', edge = 'start'))
end_year = int(splits.sel(dataset = dataset, slice_type = 'test', edge = 'end'))

# Adapt paths
path_input_prepro_pred = path_input_prepro_pred + dataset + '_dataset/'
path_input_targ = path_input_targ + dataset + '_dataset/'
path_metrics = path_metrics + str(start_year) + '-' + str(end_year) + '_period_' + dataset + '_dataset/'
path_hyperparam = path_hyperparam + str(start_year) + '-' + str(end_year) + '_period_' + dataset + '_dataset/'
path_time_series = path_time_series + str(start_year) + '-' + str(end_year) + '_period_' + dataset + '_dataset/'
path_plots_prediction = path_plots_prediction + str(start_year) + '-' + str(end_year) + '_period_' + dataset + '_dataset/'


# ### 2.4. Make dictionary

# In[67]:


# Construct dictionary
dictionary = { 
  'target_names': target_names_regr + target_names_classi,
  'target_names_regr': target_names_regr,
  'target_names_classi': target_names_classi,
  'lead_times': lead_times,
  'balance_types':balance_types,
  'dataset': dataset,
  'cv_type': cv_type,
  'hp_search_type': hp_search_type,
  'num_hp_set_candidates': num_hp_set_candidates,
  'num_outer_folds': num_outer_folds,
  'num_inner_folds': num_inner_folds,
  'timestep': timestep,
  'timestep_num': int(timestep[:-1]),
  'num_lags': num_lags,
  'all_lags_and_means': all_lags_and_means,
  'initial_month': initial_month,
  'final_month': final_month,
  'splits': splits,
  'start_year': start_year,
  'end_year': end_year,
  'boxes': boxes,
  'mean_type': mean_type,
  'percentile': percentile,
  'min_duration': min_duration,
  'geopotential_level': geopotential_level,
  'drop_predictors': drop_predictors,
  'feature_selection': feature_selection,
  'min_sum_importances': min_sum_importances,
  'pca': pca,
  'min_cum_exp_var': min_cum_exp_var,
  'prepro2': prepro2,
  'train_models': train_models,
  'optimize_rf_hyperparam': optimize_rf_hyperparam,
  'reg_regr': reg_regr,
#  'reg_classi': reg_classi,
  'metric_regr': metric_regr,
  'metric_classi': metric_classi,
  'explore_overfitting': explore_overfitting,
  'show_tree': show_tree,
  'plot_fig_title': plot_fig_title,
  'n_cores': n_cores,
  'path_eraint': path_eraint, 
  'path_era5_land': path_era5_land,
  'path_hadisst': path_hadisst,
  'path_bernatj_data': path_bernatj_data,
  'path_eobs': path_eobs,
  'path_ecmwf': path_ecmwf,
  'path_input_orig_pred': path_input_orig_pred,
  'path_input_prepro_pred': path_input_prepro_pred,
  'path_input_targ': path_input_targ,
  'path_metrics': path_metrics,
  'path_hyperparam': path_hyperparam,
  'path_time_series': path_time_series,
  'path_plots_prediction': path_plots_prediction,
  'verbosity': verbosity,
  'long_predictor_names': long_predictor_names,  
  'units': units,
  'server': server
               }


# In[ ]:




