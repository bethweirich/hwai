#!/usr/bin/env python
# coding: utf-8

# ### Executer: Heat Wave Prediction for Central Europe 

# In[1]:


# Import packages
import time
import xarray as xr


# In[2]:


# Import own functions
## Utils
from ipynb.fs.defs.utils import (print_title,
                                 print_balance,
                                 print_lead_time,
                                 print_datetime)

## Metrics
from ipynb.fs.defs.metrics import construct_metrics_dset

## Plotting
from ipynb.fs.defs.plotting import plot_tp
from ipynb.fs.defs.plotting import plot_metrics

## Preprocessing
from ipynb.fs.defs.prediction import prepro_part2_and_prediction, prepro_part2_and_prediction_nested_cv


# In[3]:


# Import dictionary
from ipynb.fs.full.const import dictionary


# In[4]:


start = time.time()
print_datetime('executer.py')


# ## Main

# In[ ]:


# Do all computations for summer t2m and for the different hw_indices
for _target_name in dictionary['target_names']:
    if dictionary['verbosity'] > 0: print_title(_target_name)

    ## CLASSIFICATION
    if 'bin' in _target_name:    
        # Balance data (or not) for binary indices  
        metrics_dset = {}
        metrics_proba_dset = {}
        metrics_dset_std = {}
        metrics_proba_dset_std = {}
        for _balance in dictionary['balance_types']:
            if dictionary['verbosity'] > 0: print_balance(_balance)   
            if dictionary['prepro2'] is True: 
                # Iterate over the different lead times
                for leadtime in dictionary['lead_times']:
                    if dictionary['verbosity'] > 0: print_lead_time(leadtime)
                   # Preprocessing part 2, reference forecasts & prediction   
                    if dictionary['cv_type'] == 'none': prepro_part2_and_prediction(_target_name, _balance, leadtime)
                    elif dictionary['cv_type'] == 'nested': prepro_part2_and_prediction_nested_cv(_target_name, _balance, leadtime)  
            metrics_dset[_balance] = construct_metrics_dset(_target_name, 'classi', 'test', _balance)
            metrics_proba_dset[_balance] = construct_metrics_dset(_target_name, 'proba_classi', 'test', _balance)        
        ### Save all balance_types together to a dataset
        #### Concat across balance types
        metrics_dset = xr.concat([metrics_dset[dictionary['balance_types'][0]], metrics_dset[dictionary['balance_types'][1]]], 'balance_type')
        metrics_proba_dset = xr.concat([metrics_proba_dset[dictionary['balance_types'][0]], metrics_proba_dset[dictionary['balance_types'][1]]], 'balance_type')     
        ### Assign coords
        metrics_dset = metrics_dset.assign_coords({'balance_type': dictionary['balance_types']})
        metrics_proba_dset = metrics_proba_dset.assign_coords({'balance_type': dictionary['balance_types']})    
        ### Concat across metrics
        metrics_dset = xr.concat([metrics_dset, metrics_proba_dset], 'metric')
        # Plot bar chart with TPR and FPR
        plot_tp(metrics_proba_dset, _target_name)

    ## REGRESSION    
    elif _target_name == 't2m':
        _balance = 'none'
        if dictionary['prepro2'] is True:
            # Iterate over the different lead times
            for leadtime in dictionary['lead_times']:
                if dictionary['verbosity'] > 0: print_lead_time(leadtime)
                # Preprocessing part 2, reference forecasts & prediction   
                if dictionary['cv_type'] == 'none': prepro_part2_and_prediction(_target_name, _balance, leadtime)
                elif dictionary['cv_type'] == 'nested': prepro_part2_and_prediction_nested_cv(_target_name, _balance, leadtime)
        metrics_dset = construct_metrics_dset(_target_name, 'regr', 'test', _balance)

    # Plot metrics for all lead times together
    plot_metrics(metrics_dset, _target_name)


# In[ ]:


end = time.time()
print('Time needed: ', (end - start)/60, ' min')


# In[ ]:




