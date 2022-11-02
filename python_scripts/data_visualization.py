#!/usr/bin/env python
# coding: utf-8

# # Data visualization

# In[1]:


import matplotlib  
import matplotlib.pyplot as plt
# Show plots
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import xarray as xr
from scipy.stats import pearsonr
import pandas as pd
import os

# Import TIGRAMITE
import tigramite
from tigramite import data_processing as prepro
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr


# In[2]:


# Import dictionary
from ipynb.fs.full.const import dictionary

# Import own functions
from ipynb.fs.defs.preprocessing_part2 import select_season
from ipynb.fs.defs.utils import compute_ones_percentage, is_mjjas
from ipynb.fs.defs.plotting import plot_data_histogram, plot_lagged_correlation_heatmap, plot_class_distr, show_multiple_time_series


# In[4]:


def lagged_correlations(predictors, target):
    
    """
      inputs
      -------
        predictors                      xr.Dataset 
        target                          xr.DataArray


      outputs
      -------
        None. Plots heatmap representing the lagged correlation of each predictor with the target. Hatched cells correspond to non-significant correlations at 5% significance level.

    """ 
    
    # Set parameters
    ## Set maxium time lag
    if dictionary['timestep'] == '1D' or dictionary['timestep'] == '5D':
        mytau_max = 8
    if dictionary['timestep'] == '7D' or dictionary['timestep'] == '1M':
        mytau_max = 6
    
    ## Add target to dataset
    dset = xr.merge([target, predictors])
    target_pos = 0
    ## Select time period
    time_slice = slice(str(int(dictionary['splits'].sel(dataset = dictionary['dataset'], slice_type = 'train_full', edge = 'start'))), 
                             str(int(dictionary['splits'].sel(dataset = dictionary['dataset'], slice_type = 'test', edge = 'end'))))
    dset = dset.sel(time = time_slice)
    print('Using period:', time_slice)

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
    ## Fill non-summer months with true mask
    ### Note: 1's: non-summer, true mask ; 0's: summer, false mask
    for var in list(dset):
        data_mask[var] = 1 - 1*(is_mjjas(data_mask.index.month))
    ## Create dataframe
    dataframe = prepro.DataFrame(dset.to_dataframe().values, mask = data_mask.values, datatime = np.arange(T), var_names = ['t2m'] + predictor_names)

    # Perform Conditional Independence Test: Linear Partial Correlation
    parcorr = ParCorr(mask_type = 'xy', significance = 'analytic')
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

# **----------------------------------------------------------------------------------------------------------------------------------------------**

# **----------------------------------------------------------------------------------------------------------------------------------------------**

# # Main

# In[ ]:


# Load preprocessed predictors & targets
## Daily resolution
predictors_daily = xr.open_dataset(dictionary['path_data'] + 'preprocessed_predictors_daily.nc') 
targets_daily = xr.open_dataset(dictionary['path_data'] + 'targets_daily.nc') 
## Weekly resolution
predictors_weekly = xr.open_dataset(dictionary['path_data'] + 'preprocessed_predictors_weekly.nc') 
targets_weekly = xr.open_dataset(dictionary['path_data'] + 'targets_weekly.nc') 


# ### 1. Predictor time series

# In[ ]:


# Show time series of predictors
show_multiple_time_series(predictors_daily)


# ### 2. Lagged correlations between predictors and target variable in selected season

# In[5]:


# Select subset of predictors
predictors_weekly = predictors_weekly[['t2m_x', 'z', 'rain', 'sm', 'sea', 'sst_nwmed', 'sst_cnaa']]  

# Calculate correlations and plot
lagged_correlations(predictors_weekly_sorted.dropna('time'), targets_weekly)


# ### 3. Histograms 

# In[ ]:


plot_data_histogram(targets_daily.t2m)


# ### 4. Show class imbalance

# In[ ]:


tgns = ['hw_bin_1SD', 'hw_bin_15SD']
for tgn in tgns:
    y = targets_daily[tgn]
    y = select_season(y, 'full dataset')
    compute_ones_percentage(y, tgn, show_position = False)
    plot_class_distr(y.drop('dayofyear'), tgn)

