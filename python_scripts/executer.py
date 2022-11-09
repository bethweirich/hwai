""" HWAI executer """

# Author: Elizabeth Weirich Benet 
# License: MIT License 2022

# Import packages
import time
import xarray as xr

# Import own functions
## Utils
from utils import (print_title,
                   print_lead_time,
                   print_datetime)
## Metrics
from metrics import construct_metrics_dset
## Plotting
from plotting import (plot_metrics, 
                      show_hyperparameters, 
                      plot_pred_time_series_all_lead_times)
## Preprocessing
from prediction import (prepro_part2_and_prediction, 
                        prepro_part2_and_prediction_nested_cv)

# Import constants
from const import dictionary

# **----------------------------------------------------------------------------------------------------------------------------------------------**

# Main

start = time.time()
print_datetime('executer.py')

# Initialize
metrics = {'proba_classi': [], 'classi': [], 'regr': []}
metrics_tgns = {'proba_classi': [], 'classi': [], 'regr': []}

# Loop over all targets
## REGRESSION
for _target_name in dictionary['target_names_regr']:
    print_title(_target_name)
    # Initialize    
    metrics_tgn = {'proba_classi': [], 'classi': [], 'regr': []}   
    pred_types = ['regr']
    if dictionary['prepro2'] is True:
        # Iterate over the different lead times
        for leadtime in dictionary['lead_times']:
            print_lead_time(leadtime)
            # Preprocessing part 2, reference forecasts & prediction   
            if dictionary['cv_type'] == 'none': prepro_part2_and_prediction(_target_name, leadtime)
            elif dictionary['cv_type'] == 'nested': prepro_part2_and_prediction_nested_cv(_target_name, leadtime)
    # Show and save table of hyperparameters
    show_hyperparameters(_target_name)
    # Time series plot combined for all lead times
    if dictionary['cv_type'] == 'none': plot_pred_time_series_all_lead_times()
    # Metrics
    for pred_type in pred_types:
        metrics[pred_type] = construct_metrics_dset(_target_name, 'regr', 'test').assign_coords(target_name = _target_name).expand_dims('target_name') 
    # Plot metrics for all lead times together
    plot_metrics(metrics, pred_type)
    
## CLASSIFICATION    
for _target_name in dictionary['target_names_classi']:
    print_title(_target_name)
    # Initialize    
    metrics_tgn = {'proba_classi': [], 'classi': [], 'regr': []}            
    pred_types = ['proba_classi', 'classi']
    if dictionary['prepro2'] is True: 
        # Iterate over the different lead times
        for leadtime in dictionary['lead_times']:
            print_lead_time(leadtime)
            # Preprocessing part 2, reference forecasts & prediction   
            if dictionary['cv_type'] == 'none': prepro_part2_and_prediction(_target_name, leadtime)
            elif dictionary['cv_type'] == 'nested': prepro_part2_and_prediction_nested_cv(_target_name, leadtime) 
    # Show and save table of hyperparameters
    show_hyperparameters(_target_name) 
    # Metrics
    for pred_type in pred_types:                
        metrics_tgn[pred_type] = construct_metrics_dset(_target_name, pred_type, 'test').assign_coords(target_name = _target_name).expand_dims('target_name')
        metrics_tgns[pred_type].append(metrics_tgn[pred_type]) 
# Metrics
for pred_type in pred_types: 
    metrics[pred_type] = xr.concat(metrics_tgns[pred_type], dim = 'target_name')       
# Plot metrics for all lead times and targets together
for pred_type in pred_types:
    plot_metrics(metrics, pred_type)     


end = time.time()
print('Time needed: ', (end - start)/60, ' min')
