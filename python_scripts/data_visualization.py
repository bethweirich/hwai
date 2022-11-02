#!/usr/bin/env python
# coding: utf-8

# # Data visualization

# In[1]:


# Import packages
import matplotlib  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
import pandas as pd
## TIGRAMITE
import tigramite
from tigramite import data_processing as prepro
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr


# In[2]:


# Import own functions
## Prepro2
from ipynb.fs.defs.preprocessing_part2 import select_season
## Utils
from ipynb.fs.defs.utils import compute_ones_percentage, is_mjjas
## Plotting
from ipynb.fs.defs.plotting import plot_data_histogram, plot_lagged_correlation_heatmap, plot_class_distr


# In[3]:


# Import dictionary
from ipynb.fs.full.const import dictionary


# In[3]:


def correlation_between_datasets(dset1, dset2):

    """
      inputs
      -------
        dset1                      xr.Dataset : one dataset containing the time series of predictors
        dset2                      xr.Dataset : other dataset containing the time series of predictors


      outputs
      -------
        None. Prints a list with the correlations between the same variable in the two datasets.

    """ 
    
    if dset1.time[0] > dset2.time[0]:
        dset2_cut_start = dset2.sel(time = slice(dset1.time[0], dset2.time[-1]))
        dset1_cut_start = dset1        
    else: 
        dset1_cut_start = dset1.sel(time = slice(dset2.time[0], dset1.time[-1]))
        dset2_cut_start = dset2
        
    if dset1.time[-1] < dset2.time[-1]:
        dset2_cut = dset2_cut_start.sel(time = slice(dset2_cut_start.time[0], dset1.time[-1]))
        dset1_cut = dset1_cut_start        
    else: 
        dset1_cut = dset1_cut_start.sel(time = slice(dset1_cut_start.time[0], dset2.time[-1]))
        dset2_cut = dset2_cut_start

    print('Overlapping time period: ', pd.DatetimeIndex(dset1_cut.to_dataframe().index)[0].year, '-', pd.DatetimeIndex(dset1_cut.to_dataframe().index)[-1].year)
    
    var_names = list(dset1)
    for var_name in var_names:  
        print(var_name, ': ', np.round(pearsonr(dset1_cut[var_name], dset2_cut[var_name]), 2)[0])
    
    print('\n')
    


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
    
    ## Set maxium time lag
    if dictionary['timestep'][-1] == 'M' or dictionary['timestep_num'] > 6: mytau_max = 6
    else: mytau_max = 8
            
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
    parcorr = ParCorr(mask_type = 'y', significance = 'analytic')
    # Initialize the PCMCI method with the produced dataframe and the used Conditional Independence Test
    pcmci = PCMCI(dataframe = dataframe, cond_ind_test = parcorr, verbosity = 1)
    # Chose the level of verbosity
    pcmci.verbosity = 1
    
    # Lagged dependencies
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

# In[5]:


# Load preprocessed predictors & targets
predictors_all_datasets = {}
targets_all_datasets = {}
for dataset in ['combi', 'ERA5', 'ERA20C']:
    predictors_all_datasets[dataset] = xr.open_dataset('/s2s/weiriche/heat_waves/inputs/predictors/preprocessed/' + dataset + '_dataset/input_' + dictionary['timestep'] + '_' + dictionary['mean_type'] + '_mean_sa.nc') 
    targets_all_datasets[dataset] = xr.open_dataset('/s2s/weiriche/heat_waves/inputs/targets/' + dataset + '_dataset/targets_' + dictionary['timestep'] + '_' + dictionary['mean_type'] + '_mean.nc') 


# ### 1. Correlations between datasets 

# In[6]:


print('Correlations between combi dataset and ERA5 dataset: ')
correlation_between_datasets(predictors_all_datasets['combi'], predictors_all_datasets['ERA5'])
print('Correlations between combi dataset and ERA20C dataset: ')
correlation_between_datasets(predictors_all_datasets['combi'], predictors_all_datasets['ERA20C'])
print('Correlations between ERA5 dataset and ERA20C dataset: ')
correlation_between_datasets(predictors_all_datasets['ERA5'], predictors_all_datasets['ERA20C'])


# ### 2. Lagged correlations between predictors and target variable in selected dataset and season

# In[7]:


# Import data with simple mean
all_predictors_simple_mean = xr.open_dataset(dictionary['path_input_prepro_pred'] + 'input_' + dictionary['timestep'] + '_simple_mean_sa.nc') 
predictors_simple_mean = all_predictors_simple_mean.drop('na_index').drop('sst_med').drop('sst_semed').drop('snao').drop('sst_nwna').drop('sst_sena').drop('pcd_index').drop('sst_enso').drop('sst_baltic').drop('sm_deeper').drop('sm_deepest')
target_simple_mean = xr.open_dataset(dictionary['path_input_targ'] + 'targets_' + dictionary['timestep'] + '_simple_mean.nc').t2m.load()
# Calculate correlations and plot
get_ipython().run_line_magic('matplotlib', 'inline')
lagged_correlations(predictors_simple_mean.dropna('time'), target_simple_mean)
lagged_correlations(all_predictors_simple_mean.dropna('time'), target_simple_mean)


# ### 3. Histograms 

# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
plot_data_histogram(targets_all_datasets[dictionary['dataset']].t2m, targets_all_datasets[dictionary['dataset']].hw_bin_2in7, False)


# ### 4. Show class imbalance

# In[9]:


from sklearn.model_selection import KFold
from ipynb.fs.defs.preprocessing_part2 import select_season
tgns = ['hw_bin_1SD', 'hw_bin_15SD', 'hw_bin_2in7']
for tgn in tgns:
    # Full dataset
    y = targets_all_datasets[dictionary['dataset']][tgn]
    y_1st_half = y.drop('dayofyear').isel(time = slice(0, int(len(y.time)/2)))
    y_2nd_half = y.drop('dayofyear').isel(time = slice(int(len(y.time)/2), int(len(y.time)-1)))
    compute_ones_percentage(y, tgn, show_position = False)
    compute_ones_percentage(y_1st_half, tgn + '_1st_half', show_position = False)
    compute_ones_percentage(y_2nd_half, tgn + '_2nd_half', show_position = False)
    plot_class_distr(y.drop('dayofyear'), tgn)
    print('*************************************************************************************************************')
    print('*************************************************************************************************************')
    print('*************************************************************************************************************')


# In[ ]:




