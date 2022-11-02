#!/usr/bin/env python
# coding: utf-8

# ### Executer: Heat Wave Prediction for Central Europe 

# In[ ]:


# Import packages
import time
import xarray as xr
import pandas as pd

##Parallelization
import concurrent.futures
import multiprocessing as mp


# In[ ]:


# Import own functions
## Utils
from ipynb.fs.defs.utils import (allDone,
                                 print_title,
                                 print_balance,
                                 print_lead_time,
                                 print_datetime)

## Metrics
from ipynb.fs.defs.metrics import construct_metrics_dset

## Plotting
from ipynb.fs.defs.plotting import plot_tp, plot_metrics, show_hyperparameters, plot_pred_time_series_all_lead_times
#, show_hyperparameters

## Preprocessing
from ipynb.fs.defs.prediction import prepro_part2_and_prediction, prepro_part2_and_prediction_nested_cv


# In[ ]:


# Import dictionary
from ipynb.fs.full.const import dictionary


# In[ ]:


start = time.time()
print_datetime('executer.py')


# ## Main

# In[ ]:


# Initialize
metrics = {'proba_classi': [], 'classi': [], 'regr': []}
metrics_tgns = {'proba_classi': [], 'classi': [], 'regr': []}

# Do all computations for summer t2m and for the different hw_indices

## REGRESSION
for _target_name in dictionary['target_names_regr']:
    if dictionary['verbosity'] > 0: print_title(_target_name)
    # Initialize    
    metrics_tgn = {'proba_classi': [], 'classi': [], 'regr': []}   
    pred_types = ['regr']
    _balance = 'none'
    if dictionary['prepro2'] is True:
        # Iterate over the different lead times
        for leadtime in dictionary['lead_times']:
            if dictionary['verbosity'] > 0: print_lead_time(leadtime)
            # Preprocessing part 2, reference forecasts & prediction   
            if dictionary['cv_type'] == 'none': prepro_part2_and_prediction(_target_name, _balance, leadtime)
            elif dictionary['cv_type'] == 'nested': prepro_part2_and_prediction_nested_cv(_target_name, _balance, leadtime)
    # Show and save table of hyperparameters
    show_hyperparameters(_target_name)
    # Time series plot combined for all lead times
    #if dictionary['cv_type'] == 'none': plot_pred_time_series_all_lead_times()
    # Metrics
    for pred_type in pred_types:
        metrics[pred_type] = construct_metrics_dset(_target_name, 'regr', 'test', _balance).assign_coords(target_name = _target_name).expand_dims('target_name') 
    # Plot metrics for all lead times together
    plot_metrics(metrics, pred_type)
    
## CLASSIFICATION    
for _target_name in dictionary['target_names_classi']:
    if dictionary['verbosity'] > 0: print_title(_target_name)
    # Initialize    
    metrics_tgn = {'proba_classi': [], 'classi': [], 'regr': []}            
    pred_types = ['proba_classi', 'classi']
    # Balance data (or not) for binary indices 
    for _balance in dictionary['balance_types']:
        if dictionary['verbosity'] > 0: print_balance(_balance)   
        if dictionary['prepro2'] is True: 
            # Iterate over the different lead times
            for leadtime in dictionary['lead_times']:
                if dictionary['verbosity'] > 0: print_lead_time(leadtime)
               # Preprocessing part 2, reference forecasts & prediction   
                if dictionary['cv_type'] == 'none': prepro_part2_and_prediction(_target_name, _balance, leadtime)
                elif dictionary['cv_type'] == 'nested': prepro_part2_and_prediction_nested_cv(_target_name, _balance, leadtime) 
        # Metrics
        for pred_type in pred_types:                
            metrics_dset = construct_metrics_dset(_target_name, pred_type, 'test', _balance)
            metrics_tgn[pred_type].append(metrics_dset.assign_coords(balance = _balance).expand_dims('balance'))     
    for pred_type in pred_types: 
        metrics_tgn[pred_type] = xr.concat(metrics_tgn[pred_type], dim = 'balance')    
        metrics_tgns[pred_type].append(metrics_tgn[pred_type].assign_coords(target_name = _target_name).expand_dims('target_name')) 
    # Show and save table of hyperparameters
    show_hyperparameters(_target_name) 
# Metrics
for pred_type in pred_types: 
    metrics[pred_type] = xr.concat(metrics_tgns[pred_type], dim = 'target_name')       
# Plot metrics for all lead times and targets together
for pred_type in pred_types:
    plot_metrics(metrics, pred_type)     


# In[ ]:


end = time.time()
print('Time needed: ', (end - start)/60, ' min')


# In[ ]:




