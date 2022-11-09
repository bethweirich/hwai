""" HWAI data visualization """

# Author: Elizabeth Weirich Benet 
# License: MIT License 2022

# Import packages
import numpy as np
import xarray as xr

# Import TIGRAMITE
from tigramite import data_processing as prepro
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr

# Import own functions
from preprocessing_part2 import select_season
from utils import (compute_ones_percentage, 
                   mask_months)
from plotting import (plot_data_histogram, 
                      plot_lagged_correlation_heatmap, 
                      plot_class_distr, 
                      show_multiple_time_series, 
                      plot_latlon_boxes)

# Import constants
from const import dictionary

# **----------------------------------------------------------------------------------------------------------------------------------------------**

# Function definitions 

def lagged_correlations(predictors, target):
    
    """
      inputs
      -------
        predictors                      xr.Dataset : preprocessed predictor time series
        target                        xr.DataArray : preprocessed 2m-temperature time series


      outputs
      -------
        None. Plots heatmap representing the lagged linear correlation of each predictor with the target. 
        Hatched cells correspond to non-significant correlations at 5% significance level.

    """ 
    
    # Set parameters
    ## Set maxium time lag
    mytau_max = 6
    
    ## Add target to dataset
    dset = xr.merge([target, predictors])
    target_pos = 0
    ## Select time period
    time_slice = slice(str(int(dictionary['splits'].sel(slice_type = 'train_full', edge = 'start'))), 
                             str(int(dictionary['splits'].sel(slice_type = 'test', edge = 'end'))))
    dset = dset.sel(time = time_slice)
    if dictionary['verbosity'] > 2: print('Using period:', time_slice)

    # Save characteristics of the data 
    ## Save duration of time series as T
    T = len(dset.time)
    ## Save predictor names
    predictor_names = list(predictors)
    predictor_names_dict = {predictor_name: dictionary['long_predictor_names'][predictor_name] for predictor_name in predictor_names}
    long_predictor_names = predictor_names_dict.values()
    ## Save number of predictors
    N = len(predictor_names) 

    # Masked dataframe
    ## data_mask is a binary dataset (0: non-masked, 1: masked data)
    ## Create dataset with same structure than data but filled with zeros
    data_mask = abs(dset * 0)
    ## Convert to pandas dataframe to be able to access months
    data_mask = data_mask.to_dataframe()
    ## Fill non-selected months with true mask
    ### Note: 1's: non-summer, true mask ; 0's: summer, false mask
    if dictionary['verbosity'] > 2: print('Masking all months except for the ones between month', dictionary['initial_month'], 'and month', dictionary['final_month'])
    for var in list(dset):
        data_mask[var] = 1 - 1*(mask_months(data_mask.index.month))
    ## Create dataframe
    dataframe = prepro.DataFrame(dset.to_dataframe().values, mask = data_mask.values, datatime = np.arange(T), var_names = ['t2m'] + predictor_names)

    # Perform Conditional Independence Test: Linear Partial Correlation
    parcorr = ParCorr(mask_type = 'y', significance = 'analytic')
    # Initialize the PCMCI method with the produced dataframe and the used Conditional Independence Test
    pcmci = PCMCI(dataframe = dataframe, cond_ind_test = parcorr, verbosity = 1)
    # Chose the level of verbosity
    pcmci.verbosity = 1    
    ## Lagged dependencies
    lagged_dep = pcmci.get_lagged_dependencies(tau_max = mytau_max)
    ## Create dataset
    taus = np.arange (mytau_max + 1)
    correlations = xr.DataArray(lagged_dep['val_matrix'], coords = [['t2m'] + predictor_names, ['t2m'] + predictor_names, taus], dims = ['predictor', 'target', 'tau'])
    p_matrix = xr.DataArray(lagged_dep['p_matrix'], coords = [['t2m'] + predictor_names, ['t2m'] + predictor_names, taus], dims = ['predictor', 'target', 'tau'])

    # Plot
    plot_lagged_correlation_heatmap(target_pos, correlations, p_matrix, long_predictor_names, N, mytau_max)


# **----------------------------------------------------------------------------------------------------------------------------------------------**

# Main

## 1. Load preprocessed predictors & targets
### Daily resolution
predictors_daily = xr.open_dataset(dictionary['path_data'] + 'preprocessed_predictors_7D_rolling_mean.nc') 
targets_daily = xr.open_dataset(dictionary['path_data'] + 'targets_7D_rolling_mean.nc') 
### Weekly resolution
predictors_weekly = xr.open_dataset(dictionary['path_data'] + 'preprocessed_predictors_7D_simple_mean.nc') 
targets_weekly = xr.open_dataset(dictionary['path_data'] + 'targets_7D_simple_mean.nc') 

### 2. Predictor time series
show_multiple_time_series(predictors_daily)

## 3. Lagged correlations between predictors and target variable in selected season
lagged_correlations(predictors_weekly.dropna('time'), targets_weekly['t2m'])

## 4. Data histogram
plot_data_histogram(targets_daily.t2m)

## 5. Class imbalance
tgns = ['hw_bin_1SD', 'hw_bin_15SD']
for tgn in tgns:
    y = targets_daily[tgn]
    plot_class_distr(y.drop('dayofyear'), tgn)
    y = select_season(y, 'full dataset')
    if dictionary['verbosity'] > 1: compute_ones_percentage(y, tgn, show_position = False)
    hw_count = int(np.sum(y.values))
    if dictionary['verbosity'] > 1: 
        print('Number of heatwave events: ', hw_count)
        print('Number of non-heatwave events: ', len(y) - hw_count)
        print('\n')
    
## 6. Latitude-longitude boxes
plot_latlon_boxes()


