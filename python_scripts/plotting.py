#!/usr/bin/env python
# coding: utf-8

# # Plotting

# In[1]:


# Import packages
import xarray as xr
import pandas as pd
from pandas import Series
import numpy as np
import matplotlib                             
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
import seaborn as sns
import os
import datetime
import itertools

## Models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree


# In[11]:


# Import own functions
## Metrics
from ipynb.fs.defs.metrics import compute_score, tpr_from_dset, fpr_from_dset
## Utils
from ipynb.fs.defs.utils import filter_dates_like, final_day_of_month, MidpointNormalize, format_fn_y, hatch_non_significant_cells


# In[3]:


# Import dictionary
from ipynb.fs.full.const import dictionary


# In[3]:


def show_multiple_time_series(dset):
    
    """
      inputs
      -------
        dset                      xr.Dataset : containing multiple time series


      outputs
      -------
        None. Plots the time series. 

    """ 
   
    var_names = list(dset)
    for var_name in var_names:
        plt.figure(figsize = (20,3))
        plt.plot(dset.time, dset[var_name])
        plt.axhline(0, color = 'mediumturquoise', linewidth = 2, linestyle = '--')
        plt.title(var_name)
        plt.show()
        plt.close()
        


# In[4]:


def show_individual_time_series(data):
    
    """
      inputs
      -------
        data                      xr.DataArray : containing one single time series


      outputs
      -------
        None. Plots one time series. 

    """ 
    
    plt.figure(figsize = (20,3))
    plt.plot(data)
    plt.axhline(0, color = 'mediumturquoise', linewidth = 2, linestyle = '--')
    plt.show()
    plt.close()
    


# In[5]:


def show_snao_sea_patterns(snao_eof_, sea_eof_, z500_snao_, z500_sea_):

    """
      inputs
      -------
        snao_eof_                      xr.Dataset : lat-lon map for SNAO EOF pattern
        sea_eof_                       xr.Dataset : lat-lon map for SEA EOF pattern
        z500_snao_                     xr.Dataset : lat-lon map for daily composite of SNAO EOF pattern
        z500_sea_                      xr.Dataset : lat-lon map for daily composite of SEA EOF pattern


      outputs
      -------
        None. Plots the SNAO and SEA patterns and their daily composite. 

    """ 
    
    print('SNAO Eof')
    snao_eof_.plot.contourf(vmin = -0.1, vmax = 0.1, levels = 9, cmap = 'coolwarm')
    plt.show()
    plt.close()
    print('SEA Eof')
    sea_eof_.plot.contourf(vmin = -0.12, vmax = 0.12, levels = 9, cmap = 'coolwarm')
    plt.show()
    plt.close()
    print('SNAO Eof daily composite')
    z500_snao_.plot.contourf(vmin = -60, vmax = 60, levels = 7, cmap = 'coolwarm')
    plt.show()
    plt.close()
    print('SEA Eof daily composite')
    z500_sea_.plot.contourf(vmin = -180, vmax = 180, levels = 7, cmap = 'coolwarm')
    plt.show()
    plt.close()
    


# In[6]:


def show_first_x_eofs(eofs_, x):

    """
      inputs
      -------
        eofs_                      xr.Dataset : containing x EOFs i.e. lat-lon maps
        x                                 int : # EOFs you want to plot


      outputs
      -------
        None. Plots the x first EOFs.

    """ 
    
    for i in range(0,x):
        eof = eofs_.sel(mode = i)
        print('Eof number ', i + 1)
        eof.plot.contourf(vmin = -0.1, vmax = 0.1, levels = 9, cmap = 'coolwarm')
        plt.show()
        plt.close()
        


# In[ ]:


def plot_pred_time_series(y, predictions_mlr, predictions_rfr, persistence, climatology, ecmwf, lead_time, tgn, outer_split_num_ = None):
    
    """
    inputs
    ------
    y                    pd.Series : time series of the true target
    predictions_mlr      np.array  : time series of the predicted target (same length and time as y) by the MLR model
    predictions_rfr      np.array  : time series of the predicted target (same length and time as y) by the RFR model
    persistence         pd.Series  : time series of the target's persistence forecast (same length and time as y)
    climatology          np.array  : time series of the target's climatology forecast (same length and time as y)
    ecmwf             xr.DataArray : time series of the target's ECMWF forecast (same length and time as y)
    lead_time                  int : Lead time of prediction
    tgn                        str : name of target variable 
    " outer_split_num_          int : counter for outer splits "


    outputs
    -------
    None.  Displays time series cutting only the summer months plot
    
    """

    predictions_mlr_df = pd.DataFrame(data = predictions_mlr, index = y.index).sort_index()  
    predictions_rfr_df = pd.DataFrame(data = predictions_rfr, index = y.index).sort_index()  
    
    ## Define the ranges for the dates
    drange = [[datetime.date(i, dictionary['initial_month'], 1),datetime.date(i, dictionary['final_month'], 30)] for i in range(y.index[0].year, y.index[-1].year + 1)]
    ## Create as many subplots as there are date ranges
    fig, axes = plt.subplots(ncols = len(drange), sharey = True, figsize = (20,5))
    fig.subplots_adjust(bottom = 0.3, wspace = 0)
    #plt.xlabel('Time (year-month)', fontsize = 16)

    if ecmwf is not None:
        ecmwf = ecmwf[tgn + '_mean']
        ymax = ecmwf.max()
        ymin = ecmwf.min()
    else:
        ymax = y.max()
        ymin = y.min()
        
    ## Loop over subplots and limit each to one date range
    for i, ax in enumerate(axes):
        ax.set_xlim(drange[i][0],drange[i][1])
        #ax.set_ylim(ymin, ymax)
        
        # Ground truth
        ax.plot(y.sort_index(), label = 'Ground Truth', color = 'slategray', linewidth = 2)
        # ML predictions
        ax.plot(predictions_mlr_df, '--', label = 'MLR', color = 'indianred')
        ax.plot(predictions_rfr_df, '--', label = 'RFR', color = 'olivedrab')
        # Persistence forecast: take target value from date - lead time
        ax.plot(persistence, '--', label = 'Persistence', color = 'silver')
        # Climatology forecast: plot the climatology forecast for the target variable
        ax.plot(climatology, '--', label = 'Climatology', color = 'mediumturquoise')
        # ECMWF forecast: plot the ECMWF forecast for the target variable
        if ecmwf is not None: 
            ax.plot(ecmwf, '--', label = 'ECMWF', color = 'cornflowerblue')       
        loc = matplotlib.dates.MonthLocator(np.arange(dictionary['initial_month'], dictionary['final_month'] + 1))
        fmt =  matplotlib.dates.DateFormatter("%Y-%b")
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)
        plt.setp(ax.get_xticklabels(), rotation = 90, fontsize = 16)
        if i != 0:
            ax.tick_params(axis = 'y', which = 'both', length = 0)
        #if i == 0: ax.set_ylabel('Temperature anomalies (°C)', fontsize = 16)           
        ax.yaxis.set_tick_params(labelsize = 16)
        
    ax.legend(bbox_to_anchor = (1.1, 1.5), loc = 'upper right')
    if dictionary['plot_fig_title'] == True:
        fig.suptitle('Predicted and true ' + tgn + ' on test set at ' + str(lead_time) + ' weak(s) lead time', fontsize = 14)
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + 'test/' + tgn + '/pred_time_series/'
    if dictionary['cv_type'] == 'nested': save_name = dir_name + 'pred_time_series_' + tgn + '_lead_time_' + str(lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + 'pred_time_series_' + tgn + '_lead_time_' + str(lead_time) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()


# In[ ]:


def plot_zoomed_in_pred_time_series(y, predictions_mlr, predictions_rfr, persistence, climatology, ecmwf, start_date_test, lead_time, tgn, outer_split_num_ = None):

    """
    inputs
    ------
    y                    pd.Series : time series of the true target
    predictions_mlr       np.array : time series of the predicted target (same length and time as y) by the MLR model
    predictions_rfr       np.array : time series of the predicted target (same length and time as y) by the RFR model
    persistence          pd.Series : time series of the target's persistence forecast (same length and time as y)
    climatology           np.array : time series of the target's climatology forecast (same length and time as y)
    ecmwf             xr.DataArray : time series of the target's ECMWF forecast
    start_date_test            str : start date of test time period
    lead_time                  int : Lead time of prediction
    tgn                        str : name of target variable 
    " outer_split_num_          int : counter for outer splits "

    outputs
    -------
    None.  Displays the zoomed-in plot
    
    """

    predictions_mlr_df = pd.DataFrame(data = predictions_mlr, index = y.index).sort_index()  
    predictions_rfr_df = pd.DataFrame(data = predictions_rfr, index = y.index).sort_index()  
    climatology_df = pd.DataFrame(data = climatology, index = y.index).sort_index()  
    
    # Initialize figure
    plt.figure(figsize = (20,5))
    # Find zoom period
    ## Select 2nd year of test set: 
    zoom_year = str(int(start_date_test[:4]) + 1)
    zoom_period_start = zoom_year + '-' + str(dictionary['initial_month']).zfill(2) + '-01'
    zoom_period_end = zoom_year + '-' + str(dictionary['final_month']).zfill(2) + '-' + str(final_day_of_month(dictionary['final_month'])).zfill(2)
    
    print('zoom_period_start: ', zoom_period_start)
    print('zoom_period_end: ', zoom_period_end)
    # Ground truth
    plt.plot(y.sort_index().loc[zoom_period_start:zoom_period_end], label = 'Ground Truth', color = 'slategray', linewidth = 2)
    # ML predictions
    plt.plot(predictions_mlr_df.loc[zoom_period_start:zoom_period_end], '--', label = 'MLR', color = 'indianred')
    plt.plot(predictions_rfr_df.loc[zoom_period_start:zoom_period_end], '--', label = 'RFR', color = 'olivedrab')
    # Persistence forecast: take target value from date - lead time
    plt.plot(persistence.loc[zoom_period_start:zoom_period_end], '--', label = 'Persistence', color = 'silver')
    # Climatology forecast: plot the climatology forecast for the target variable
    plt.plot(climatology_df.loc[zoom_period_start:zoom_period_end], '--', label = 'Climatology', color = 'mediumturquoise')
    # ECMWF forecast: plot the ECMWF forecast for the target variable
    if ecmwf is not None: 
        ecmwf = ecmwf[tgn + '_mean']
        plt.plot(ecmwf.loc[zoom_period_start:zoom_period_end], '--', label = 'ECMWF', color = 'cornflowerblue')
    
    plt.xlabel('Time (year-month)', fontsize = 12)
    plt.ylabel('Temperature anomalies (°C)', fontsize = 12)
    plt.legend(loc = 'best')
    if dictionary['plot_fig_title'] == True:
        plt.title('Zoomed in: ' + str(zoom_year) + ' for the predicted and true ' + tgn + ' on test set at ' + str(lead_time) + ' weak(s) lead time', fontsize = 14)
    ## Save plot
    dir_name = dictionary['path_plots_prediction']+ 'test/' + tgn + '/zoomed_in_pred_time_series/'    
    if dictionary['cv_type'] == 'nested': save_name = dir_name + 'zoomed_in_pred_time_series_' + tgn + '_lead_time_' + str(lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + 'zoomed_in_pred_time_series_' + tgn + '_lead_time_' + str(lead_time) + '_weeks.pdf'    
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    


# In[ ]:


def plot_cross_corr_old(y, predictions_mlr, predictions_rfr, predictions_ecmwf, persistence, lead_time, tgn, outer_split_num_ = None):
    
    """
    inputs
    ------
    y                     pd.Series : time series of the true target
    predictions_mlr       np.array  : time series of the predicted target (same
                                length and time as y) by the MLR model
    predictions_rfr       np.array  : time series of the predicted target (same
                                length and time as y) by the RFR model
    predictions_ecmwf     np.array  : time series of the predicted target (same
                                length and time as y) by the RFR model
    persistence           np.array  : time series of the target's persistence (same
                                length and time as y)
    lead_time                   int : Lead time of prediction
    tgn                         str : name of target variable 
    " outer_split_num_          int : counter for outer splits "

    outputs
    -------
    None.  Displays the lagged correlations plot
    
    """
    
    maxlagnum = 5 * dictionary['timestep_num']
    plt.figure(figsize = (8,5))
    plt.xcorr(y, predictions_mlr, maxlags = maxlagnum, normed = True, color = 'indianred', label = 'MLR-GT', linewidth = 1, usevlines = False, linestyle = '-')
    plt.xcorr(y, predictions_rfr, maxlags = maxlagnum, normed = True, color = 'olivedrab', label = 'RFR-GT', linewidth = 1, usevlines = False, linestyle = '-')
    if predictions_ecmwf is not None: 
        predictions_ecmwf = predictions_ecmwf[tgn + '_mean']
        plt.xcorr(filter_dates_like(y, predictions_ecmwf), predictions_ecmwf, maxlags = maxlagnum, normed = True, color = 'cornflowerblue', label = 'ECMWF-GT', linewidth = 1, usevlines = False, linestyle = '-')
    plt.xcorr(y, persistence, maxlags = maxlagnum, normed = True, color = 'silver', label = 'Persistence-GT', linewidth = 1, usevlines = False, linestyle = '-')
    ## Vertical line at 0 lag time (prediction)
    plt.axvline(0, color = 'black', label = 'Prediction time', linewidth = 1, linestyle = '-')
    plt.axvline(-lead_time * dictionary['timestep_num'], color = 'black', label = 'Lead time', linewidth = 1, linestyle = '--')
    plt.axhline(0, color = 'grey', linewidth = 1, linestyle = '-')
    if dictionary['plot_fig_title'] == True:
        plt.title('Lagged cross-correlation between predictions and ground truth')
    plt.xticks(np.arange(-maxlagnum, maxlagnum + 1, step = dictionary['timestep_num'])) 
    plt.xlabel('Prediction lead time in days', fontsize = 12)
    plt.ylabel('Correlation', fontsize = 12)
    plt.xlim(-42, 42)
    plt.ylim(-0.2, 1)
    # For multiplot
    #if lead_time == 2:
    plt.legend()
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + 'test/' + tgn + '/cross_correlation/'    
    if dictionary['cv_type'] == 'nested': save_name = dir_name + 'cross_corr_' + tgn + '_lead_time_' + str(lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + 'cross_corr_' + tgn + '_lead_time_' + str(lead_time) + '_weeks.pdf'    
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()


# In[ ]:


def plot_cross_corr(y, predictions_mlr, predictions_rfr, predictions_ecmwf, persistence, lead_time, tgn, outer_split_num_ = None):
    
    """
    inputs
    ------
    y                     pd.Series : time series of the true target
    predictions_mlr       np.array  : time series of the predicted target (same
                                length and time as y) by the MLR model
    predictions_rfr       np.array  : time series of the predicted target (same
                                length and time as y) by the RFR model
    predictions_ecmwf     np.array  : time series of the predicted target (same
                                length and time as y) by the RFR model
    persistence           np.array  : time series of the target's persistence (same
                                length and time as y)
    lead_time                   int : Lead time of prediction
    tgn                         str : name of target variable 
    " outer_split_num_          int : counter for outer splits "

    outputs
    -------
    None.  Displays the lagged correlations plot
    
    """
    
    maxlagnum = 7 * dictionary['timestep_num']
    plt.figure(figsize = (8,5))
    
    # Correlations to Ground Truth & to persistence
    ## MLR
    plt.xcorr(y, predictions_mlr, maxlags = maxlagnum, normed = True, color = 'indianred', label = 'MLR-Ground Truth', linewidth = 1, usevlines = False, linestyle = '-', marker = 'o', markersize = 5)
    plt.xcorr(persistence, predictions_mlr, maxlags = maxlagnum, normed = True, color = 'indianred', label = 'MLR-Persistence', linewidth = 1, usevlines = False, linestyle = '--', marker = 'o', markerfacecolor = 'none', markersize = 5)
    ## RFR
    plt.xcorr(y, predictions_rfr, maxlags = maxlagnum, normed = True, color = 'olivedrab', label = 'RFR-Ground Truth', linewidth = 1, usevlines = False, linestyle = '-', marker = 'o', markersize = 5)
    plt.xcorr(persistence, predictions_rfr, maxlags = maxlagnum, normed = True, color = 'olivedrab', label = 'RFR-Persistence', linewidth = 1, usevlines = False, linestyle = '--', marker = 'o', markerfacecolor = 'none', markersize = 5)
    ## ECMWF
    if predictions_ecmwf is not None: 
        predictions_ecmwf = predictions_ecmwf[tgn + '_mean']
        plt.xcorr(filter_dates_like(y, predictions_ecmwf), predictions_ecmwf, maxlags = maxlagnum, normed = True, color = 'cornflowerblue', label = 'ECMWF-Ground Truth', linewidth = 1, usevlines = False, linestyle = '-', marker = 'o', markersize = 5)  
        plt.xcorr(filter_dates_like(persistence, predictions_ecmwf), predictions_ecmwf, maxlags = maxlagnum, normed = True, color = 'cornflowerblue', label = 'ECMWF-Persistence', linewidth = 1, usevlines = False, linestyle = '--', marker = 'o', markerfacecolor = 'none', markersize = 5)  
        
    ## Vertical line at 0 lag time (prediction)
    plt.axvline(0, color = 'black', label = 'Prediction time', linewidth = 1, linestyle = '-')
    plt.axvline(-lead_time * dictionary['timestep_num'], color = 'black', label = 'Lead time', linewidth = 1, linestyle = '--')
    plt.axhline(0, color = 'grey', linewidth = 1, linestyle = '-')
    if dictionary['plot_fig_title'] == True:
        plt.title('Lagged cross-correlation between predictions and ground truth')
    plt.xticks(np.arange(-maxlagnum, maxlagnum + 1, step = dictionary['timestep_num'])) 
    plt.xlabel('Prediction lag (days)', fontsize = 12)
    plt.ylabel('Correlation', fontsize = 12)
    plt.xlim(-45, 45)
    plt.ylim(-0.2, 1)
    # For multiplot
    #if lead_time == 2:
    plt.legend()
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + 'test/' + tgn + '/cross_correlation/'    
    if dictionary['cv_type'] == 'nested': save_name = dir_name + 'cross_corr_' + tgn + '_lead_time_' + str(lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + 'cross_corr_' + tgn + '_lead_time_' + str(lead_time) + '_weeks.pdf'    
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()


# In[8]:


def plot_roc_curve(fpr, tpr, best_th_ix, subset, tg_name, model_name_, _lead_time, outer_split_num_ = None, inner_split_num_ = None):
    
    """
    inputs
    ------
    fpr                        ndarray : False Positive Rate for each probability threshold               
    tpr                        ndarray : True Positive Rate for each probability threshold  
    best_th_ix                     int : index from best probability threshold (the one that maximises the geometric mean of TPR and FPR)
    subset                         str : 'vali', 'train', 'test'
    tg_name                        str : target name
    model_name_                    str : name of the ML model 
    _lead_time                     int : lead time for prediction
    " outer_split_num_             int : counter for outer splits "
    " inner_split_num_             int : counter for inner splits "
    
    outputs
    -------
    None. Plots the ROC AUC curve with or without best threshold
    
    """           
    
    # Plot the ROC curve for the model
    plt.figure(figsize = (5,5))
    plt.plot([0,1], [0,1], linestyle='--', label = 'No Skill')
    plt.plot(fpr, tpr, label = 'Prediction')
    if best_th_ix != None: 
        if isinstance(fpr, pd.Series) or isinstance(fpr, pd.DataFrame):
            plt.scatter(fpr.iloc[best_th_ix], tpr.iloc[best_th_ix], marker='o', color='black', label='Best')
        else:
            plt.scatter(fpr[best_th_ix], tpr[best_th_ix], marker='o', color='black', label='Best')
    ## Axis labels
    plt.xlabel('FPR', fontsize = 12)
    plt.ylabel('TPR', fontsize = 12)
    plt.legend(loc = 'lower right')
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + subset + '/' + tg_name + '/ROC/' + model_name_ + '/'
    if dictionary['cv_type'] == 'nested': 
        if inner_split_num_ is not None:
            save_name = dir_name + 'ROC_' + model_name_ + '_' + tg_name + '_lead_time_' + str(_lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '_inner_split_' + str(inner_split_num_) + '.pdf'
        else: save_name = dir_name + 'ROC_' + model_name_ + '_' + tg_name + '_lead_time_' + str(_lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + 'ROC_' + model_name_ + '_' + tg_name + '_lead_time_' + str(_lead_time) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name)
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()


# In[9]:


def plot_pr_curve(precision, recall, best_th_ix, subset, tg_name, model_name_, _lead_time, outer_split_num_ = None, inner_split_num_ = None):

    """
    inputs
    ------
    precision                  ndarray : False Positive Rate for each probability threshold               
    recall                     ndarray : True Positive Rate for each probability threshold  
    best_th_ix                     int : index from best probability threshold (the one that maximises the geometric mean of TPR and FPR)
    subset                         str : 'vali', 'train', 'test'
    tg_name                        str : target name
    model_name_                    str : name of the ML model 
    _lead_time                     int : lead time for prediction
    " outer_split_num_             int : counter for outer splits "
    " inner_split_num_             int : counter for inner splits "
    
    outputs
    -------
    None. Plots the PR AUC curve with or without best threshold
    
    """      
           
    # Plot the PR curve for the model
    plt.figure(figsize = (5,5))
    #no_skill = len(y[y == 1]) / len(y)
    #plt.plot([0,1], [no_skill, no_skill], linestyle = '--', label = 'No Skill')
    plt.plot(recall, precision, label = 'Prediction')
    if best_th_ix != None: plt.scatter(recall[best_th_ix], precision[best_th_ix], marker = 'o', color = 'black', label = 'Best')    
    ## Axis labels
    plt.xlabel('Recall', fontsize = 12)
    plt.ylabel('Precision', fontsize = 12)
    plt.legend()
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + subset + '/' + tg_name + '/PR/' + model_name_ + '/'
    if dictionary['cv_type'] == 'nested': 
        if inner_split_num_ is not None:
            save_name = dir_name + 'PR_' + model_name_ + '_' + tg_name + '_lead_time_' + str(_lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '_inner_split_' + str(inner_split_num_) + '.pdf'
        else: save_name = dir_name + 'PR_' + model_name_ + '_' + tg_name + '_lead_time_' + str(_lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + 'PR_' + model_name_ + '_' + tg_name + '_lead_time_' + str(_lead_time) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name)
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    


# In[ ]:


def plot_proba_histogram(pred_proba_, best_threshold, tgn_, _model_name_, _lead_time_, outer_split_num_ = None, inner_split_num_ = None):

    """
    inputs
    ------
    pred_proba_              pd.Series : probabilities of belonging to the class 1 (being a HW)
    best_threshold               float : best probability threshold    
    tgn_                           str : target name
    _model_name_                   str : name of the ML model 
    _lead_time_                    int : lead time for prediction
    " outer_split_num_          int : counter for outer splits "
    " inner_split_num_          int : counter for inner splits "
    
    outputs
    -------
    None. Plots the histogram of heat wave probabilities and marks the best threshold
    
    """    
    
    # Plot histogram
    fig = plt.figure(figsize = (6, 4))
    plt.hist(pred_proba_, bins = 100, facecolor = 'cornflowerblue', alpha = 0.75)
    plt.xlabel('Probability of having a heat wave', fontsize = 12)
    plt.ylabel('Number of events', fontsize = 12)
    ## Vertical line at best_threshold
    plt.axvline(best_threshold, color = 'darkred', label = 'Best threshold', linewidth = 1, linestyle = '-')
    plt.legend(loc = 'best')
    if dictionary['plot_fig_title'] == True:
        plt.title('Histogram of heat wave probabilities')
    plt.grid(True)
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + 'vali/' + tgn_ + '/hist/' + _model_name_ + '/'
    if dictionary['cv_type'] == 'nested': 
        save_name = dir_name + 'hist_' + _model_name_ + '_' + tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks_outer_split_' + str(outer_split_num_) + '_inner_split_' + str(inner_split_num_) + '.pdf'
    else: save_name = dir_name + 'hist_' + _model_name_ + '_' + tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name)
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    


# In[13]:


def plot_class_distr(index, tgn_):

    """
    inputs
    ------
    index                 xr.DataArray : binary weekly heat wave index            
    tgn_                           str : target name
    
    outputs
    -------
    None. Plots the class distribution of heat wave index in time 
    
    """    
    
    fig = plt.figure(figsize = (15, 4))
    #plt.hist(index, bins = 100, facecolor = 'cornflowerblue', alpha = 0.75)
    index.plot(linewidth = 0.3, color = 'black')
    plt.xlabel('# Events', fontsize = 12)
    plt.ylabel('Time', fontsize = 12)
    if dictionary['plot_fig_title'] == True:
        plt.title('Probability class distribution for '+ tgn_)
    plt.grid(True)
    plt.show()
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + 'data_visualization/'
    save_name = dir_name + 'class_distr_' + tgn_ + '.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name)    
    plt.close()
    


# In[3]:


def plot_metrics(metrics_full, tgn):
    
    """
    inputs
    ------
    metrics_full       xr.Dataset: dataset of metrics data with dimensions (forecast x metric x lead_time "x statistics")
    tgn                      str : name of target variable 


    outputs
    -------
    None.  Plots n graphs (one for each metric) to evaluate (and compare) the performance 
    of the models. x-axis: lead time, y-axis: the metric
    
        RMSE: Root Mean Squared Error
        Corr: Correlation
        
        ROC AUC: Receiver Operating Characteristic Area Under Curve
        PR AUC: Precision Recall Area Under Curve
        
        PPV: Precision
        TPR: Recall, hit rate
        F1: F1-score
    
    """

    # Separate mean and std
    if dictionary['cv_type'] == 'nested': 
        metrics = metrics_full.sel(statistics = 'mean')
        metrics_std = metrics_full.sel(statistics = 'std')
    else: metrics = metrics_full
        
    # Prepare plot    
    fig = plt.figure(figsize = (12,4))
    
    # Loop over all metrics except for the confusion matrix
    metrics_names = metrics.metric.values
    ## Remove confusion matrix and rates
    for metric in ['TP', 'FP', 'FN', 'TN', 'TPR', 'FPR']:
        if metric in metrics_names: metrics_names = np.delete(metrics_names, np.where(metrics_names == metric))
    ## Loop over all other metrics    
    for metric, index in zip(metrics_names, range(len(metrics_names))):
        columns = len(metrics_names)
        rows = 1 
        ax = fig.add_subplot(rows, columns, index + 1)  
        # Set limits for y axis for RMSE
        if metric == 'RMSE': 
            ymin = -1
            ymax = 11
            ax.set_ylim([ymin, ymax])
            
        # Define models, forecasts, and colors
        reference_models = ['Ground Truth', 'Climatology', 'Persistence']
        if 'bin' not in tgn: 
            forecasts_std = ['MLR_std', 'RFR_std']
            colors = ['dimgray', 'mediumturquoise', 'silver', 'indianred', 'olivedrab']
        elif 'bin' in tgn:  
            ml_models = list(itertools.product(['RC', 'RFC'], zip(['oversampling', 'undersampling'],['+', '-'], ['-o', '--o'])))
            forecasts_std = ['RC_std', 'RFC_std']
            colors = ['dimgray', 'mediumturquoise', 'silver', 'darkslateblue', 'mediumslateblue', 'tab:orange', 'orange']
        # Add ECMWF
        if 'ECMWF' in metrics.forecast:
                colors.insert(len(reference_models), 'cornflowerblue')
                reference_models.append('ECMWF')
                forecasts_std.append('ECMWF_std')
        # Plot lines and uncertainty       
        if 'bin' not in tgn: 
            forecasts = reference_models + ['MLR', 'RFR']
            for (forecast, color) in zip(forecasts, colors):
                if dictionary['cv_type'] == 'nested': 
                    plt.errorbar(metrics.lead_time, 
                                 metrics.sel(metric = metric, forecast = forecast).value, 
                                 yerr = metrics_std.sel(metric = metric, forecast = forecast).value,                            
                                 fmt = '-o', 
                                 label = forecast, 
                                 color = color, ecolor = color, 
                                 capsize = 4, linewidth = 2, markersize = 4)
                elif dictionary['cv_type'] == 'none':
                    plt.plot(metrics.lead_time, 
                                 metrics.sel(metric = metric, forecast = forecast).value, 
                                 '-o', 
                                 label = forecast, 
                                 color = color, 
                                 linewidth = 2, markersize = 4)  
                    if forecast + '_std' in forecasts_std:
                        plt.errorbar(metrics.lead_time, 
                                     metrics.sel(metric = metric, forecast = forecast).value, 
                                     yerr = metrics.sel(metric = metric, forecast = forecast + '_std').value,                            
                                     fmt = '-o', 
                                     label = None, 
                                     color = color, ecolor = color, 
                                     capsize = 4, linewidth = 2, markersize = 4)                      
        elif 'bin' in tgn:  
            reference_models = list(itertools.product(reference_models, zip(['undersampling'],[''], ['-o'])))
            all_model_characteristics = reference_models + ml_models
            for model_characteristics, color in zip(all_model_characteristics, colors): 
                if dictionary['cv_type'] == 'nested': 
                    plt.errorbar(metrics.lead_time, 
                                 metrics.sel(metric = metric, forecast = model_characteristics[0], balance_type = model_characteristics[1][0]).value, 
                                 yerr = metrics_std.sel(metric = metric, forecast = model_characteristics[0]).value, 
                                 fmt = model_characteristics[1][2], 
                                 label = model_characteristics[0] + '$^{' + model_characteristics[1][1] + '}$', 
                                 color = color, 
                                 capsize = 4, linewidth = 2, markersize = 4)
                elif dictionary['cv_type'] == 'none':
                    plt.plot(metrics.lead_time, 
                                 metrics.sel(metric = metric, forecast = model_characteristics[0], balance_type = model_characteristics[1][0]).value, 
                                 model_characteristics[1][2], 
                                 label = model_characteristics[0] + '$^{' + model_characteristics[1][1] + '}$', 
                                 color = color, 
                                 linewidth = 2, markersize = 4)
                    if model_characteristics[0] + '_std' in forecasts_std:
                        plt.errorbar(metrics.lead_time, 
                                     metrics.sel(metric = metric, forecast = model_characteristics[0], balance_type = model_characteristics[1][0]).value, 
                                     yerr = metrics.sel(metric = metric, forecast = model_characteristics[0] + '_std', balance_type = model_characteristics[1][0]).value,                            
                                     fmt = model_characteristics[1][2], 
                                     label = None, 
                                     color = color, ecolor = color, 
                                     capsize = 4, linewidth = 2, markersize = 4)

                
        # Formatting
        plt.xlabel('Prediction lead time in weeks', fontsize = 14)
        plt.xticks(ticks =  metrics.lead_time)
        plt.ylabel(metric, fontsize = 14)
        plt.grid(axis = 'x', color = '#D3D3D3', linestyle = 'solid')
        plt.grid(axis = 'y', color = '#D3D3D3', linestyle = 'solid')
                
    plt.legend(bbox_to_anchor = (1.52, 1.025), loc = 'upper right', fontsize = 14)
    if dictionary['plot_fig_title'] == True:
        plt.suptitle(tgn + ' prediction', fontsize = 14)
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + 'test/' + tgn + '/metrics_plot/'
    if dictionary['cv_type'] == 'nested': save_name = dir_name + 'metrics_lt_plot_' + tgn + '_nested_CV.pdf'
    else: save_name = dir_name + 'metrics_lt_plot_' + tgn + '.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()


# In[7]:


def plot_tp(metrics_full, tgn):
    
    """
    inputs
    ------
    metrics_full       xr.Dataset: dataset of metrics data with dimensions (forecast x metric x lead_time "x statistics")
    tgn                      str : name of target variable

    outputs
    -------
    None.  Plots a bar chart with TPR and FPR to evaluate (and compare) the performance 
    of the models. x-axis: lead time, y-axis: TPR and FPR
    
        TPR: True Positive Rate
        FPR: False Positive Rate (events that were predicted but not detected) = FP/(TN+FN)
    
    """
    
    # Separate mean and std
    if dictionary['cv_type'] == 'nested': 
        metrics = metrics_full.sel(statistics = 'mean')
        metrics_std = metrics_full.sel(statistics = 'std')
    else: metrics = metrics_full
        
    # Prepare plot    
    ## Initialize figure 
    fig = plt.figure(figsize = (10,5))
    ax = fig.add_subplot(111)
    ## Bar width 
    bar_width = 0.1
    ## x axis
    xlabels = list(metrics.lead_time.values)
    ### The label locations
    x = np.arange(len(xlabels)) 
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation = 0)
    ax.set_xlabel('Prediction lead time in weeks', fontsize = 16)
    ## y axis
    ax.set_ylabel('TPR', fontsize = 16)

    # Plot
    reference_models = ['Ground Truth', 'Climatology', 'Persistence']    
    ml_models = list(itertools.product(['RC', 'RFC'], zip(['oversampling', 'undersampling'], ['+', '-'])))
    forecasts_std = ['RC_std', 'RFC_std']
    colors = ['dimgray', 'mediumturquoise', 'silver', 'darkslateblue', 'mediumslateblue', 'tab:orange', 'orange']
    if 'ECMWF' in metrics.forecast:
        colors.insert(len(reference_models), 'cornflowerblue')
        reference_models.append('ECMWF')
        forecasts_std.append('ECMWF_std')
    ## Correct format
    reference_models = list(itertools.product(reference_models, zip(['undersampling'], [''])))
    bar_shifts = np.linspace(-len(reference_models), len(ml_models), len(reference_models) + len(ml_models) + 1, dtype = int)            
    all_model_characteristics = reference_models + ml_models
    for model_characteristics, color, bar_shift in zip(all_model_characteristics, colors, bar_shifts):
        # Bar plot
        ax.bar(x + bar_shift * bar_width, 
               metrics.sel(forecast = model_characteristics[0]).sel(balance_type = model_characteristics[1][0]).sel(metric = 'TPR').to_array().values.flatten(),
               label = model_characteristics[0] + '$^{' + model_characteristics[1][1] + '}$',
               color = color, 
               width = bar_width)
    for model_characteristics, color, bar_shift, i in zip(all_model_characteristics, colors, bar_shifts, range(len(colors))):
        # Hatch
        ax.bar(x + bar_shift * bar_width, 
               metrics.sel(forecast = model_characteristics[0]).sel(balance_type = model_characteristics[1][0]).sel(metric = 'FPR').to_array().values.flatten(),
               label = 'FPR' if i == 0 else '', 
               edgecolor = 'black', 
               hatch = '//', 
               width = bar_width, 
               facecolor = 'none') 
        # Add error bars representing uncertainty
        if dictionary['cv_type'] == 'nested':             
            if metrics_std.sel(forecast = model_characteristics[0]).sel(balance_type = model_characteristics[1][0]).sel(metric = 'TPR').to_array().values.flatten().any() > 0.0001:
                ax.errorbar(x + bar_shift * bar_width, 
                            metrics.sel(forecast = model_characteristics[0]).sel(balance_type = model_characteristics[1][0]).sel(metric = 'TPR').to_array().values.flatten(),
                            yerr = metrics_std.sel(forecast = model_characteristics[0]).sel(balance_type = model_characteristics[1][0]).sel(metric = 'TPR').to_array().values.flatten(),
                            fmt = 'o', color = 'black', capsize = 4, linewidth = 1, markersize = 2)   
        elif dictionary['cv_type'] == 'none': 
            #if model_characteristics[0] + '_std' in forecasts_std and metrics.sel(forecast = model_characteristics[0] + '_std').sel(balance_type = model_characteristics[1][0]).sel(metric = 'TPR').to_array().values.flatten().any() > 0.0001: 
            for forecasts_std_substr in forecasts_std:
                if model_characteristics[0] in forecasts_std_substr:
                    ax.errorbar(x + bar_shift * bar_width, 
                            metrics.sel(forecast = model_characteristics[0]).sel(balance_type = model_characteristics[1][0]).sel(metric = 'TPR').to_array().values.flatten(),
                            yerr = metrics.sel(forecast = model_characteristics[0] + '_std').sel(balance_type = model_characteristics[1][0]).sel(metric = 'TPR').to_array().values.flatten(),
                            fmt = 'o', color = 'black', capsize = 4, linewidth = 1, markersize = 2)
    
    ## Ticks
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    ## Plot legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor = (1.28, 1.021), loc = 'upper right', fontsize = 14)
    ## Plot title
    if dictionary['plot_fig_title'] is True:
        plt.title(tgn + ' prediction', fontsize = 16)
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + 'test/' + tgn + '/metrics_plot/'
    if dictionary['cv_type'] == 'nested': save_name = dir_name + 'TP_lt_plot_' + tgn + '_nested_CV.pdf'
    else: save_name = dir_name + 'TP_lt_plot_' + tgn + '.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches='tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    


# In[ ]:


def show_coeffs(_model_, model_name, _pred_names, _feature_names, _tgn_, _lead_time_, outer_split_num_ = None):
    
    """
    inputs
    ------
    _model_                     fct : trained model
    model_name                  str : 'MLR' or 'RC'
    _pred_names         list of str : names of all predictors
    _feature_names      list of str : names of all features 
    _tgn_                       str : name of target
    _lead_time_                 int : lead time of prediction
    " outer_split_num_          int : counter for outer splits "

    outputs
    -------
    None.  Prints the coefficients of the Multilinear regression and saves them to file
    
    """

    coeffs = _model_.coef_
    
    ## Print coefficients
    if dictionary['verbosity'] > 1: 
        intercept = _model_.intercept_
        if model_name == 'RC': intercept = intercept[0]
        print('Intercept: {:.4f}'.format(intercept))
        print('\nRegression coefficients: ')           
        
        if model_name == 'RC': coeffs = coeffs[0]
        feature_coeffs = [(feature, round(coef, 4)) for feature, coef in zip(_feature_names, coeffs)]
        sorted_feature_coeffs = sorted(feature_coeffs, key = lambda x: np.abs(x[1]), reverse = True)
        [print('Predictor: {:20} Coefficient: {}'.format(*pair)) for pair in sorted_feature_coeffs]
        
        # Sum of all coeffs
        sum_coeffs = 0  
        for i in range(len(_feature_names)):
            sum_coeffs = sum_coeffs + coeffs[i]
        print('Sum of ' + model_name + ' coeffs: ', round(sum_coeffs, 4))
            
        
    ## Plot coefficients
    coef = Series(coeffs, _feature_names).sort_values(key = np.abs)
    coef = np.flip(coef, axis = 0)
    coef.plot(kind = 'bar', figsize = (2 + len(coeffs) * 0.25, 5))
    plt.grid(axis = 'y', color = '#D3D3D3', linestyle = 'solid')
    plt.xticks(rotation = 90)
    plt.ylabel('Regression coefficients')
    plt.xlabel('Feature')
    if dictionary['plot_fig_title'] == True:
        plt.title('Regression Coefficients for ' + model_name + ' at lead time ' + str(_lead_time_) + ' week', fontsize = 14)
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + 'train/' + _tgn_ + '/features/' + model_name + '/'
    if dictionary['cv_type'] == 'nested': save_name = dir_name + model_name.lower() + '_coeffs_' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + model_name.lower() + '_coeffs_' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    
    # Regression coefficients table        
    if dictionary['cv_type'] == 'none' and dictionary['server'] == 'CFC': 
        save_name = dir_name + 'data_' + model_name.lower() + '_coeffs_' + _tgn_ + '.pdf'        
        df = pd.DataFrame(data = {'lead_time_' + str(_lead_time_): list(np.round(coeffs, 2)), 
                                  'feature_names': _feature_names})
        df = df.set_index('feature_names')
        if _lead_time_ > dictionary['lead_times'][0]:
            df_old = pd.read_csv(save_name, index_col = 0)
            df = df_old.join(df, how = 'outer', sort = False)
        df.to_csv(save_name, index = True)  
        if _lead_time_ == dictionary['lead_times'][-1] and dictionary['verbosity'] > 1: 
            # Reorder index for manuscript
            ordered_features_manuscript = [x + '_7D_lag' + str(i) for x in _pred_names for i in np.arange(dictionary['lead_times'][0], dictionary['lead_times'][-1] + dictionary['num_lags'])]
            df = df.reindex(ordered_features_manuscript)            
            # Show table
            pd.set_option('display.max_rows', None, 'display.max_columns', None)
            display(df)
            pd.reset_option('display.max_rows', 'display.max_columns')
            


# In[4]:


def show_rf_imp(_rf_, _pred_names, _feature_names, _tgn_, _lead_time_, outer_split_num_ = None):

    """
    inputs
    ------
    _rf_                     fct : trained RF model
    _pred_names      list of str : names of all predictors
    _feature_names   list of str : names of all features 
    _tgn_                    str : name of target
    _lead_time_              int : lead time of prediction
    " outer_split_num_       int : counter for outer splits "

    outputs
    -------
    None.  Prints the importances of the predictors for the Random Forest and saves them to file
    
    """

    # Print feature importances       
    ## Get feature importances, sort them, and print the result
    importances = list(_rf_.feature_importances_)
    feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(_feature_names, importances)]
    sorted_feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    if dictionary['verbosity'] > 1: [print('Predictor: {:20} Importance: {}'.format(*pair)) for pair in sorted_feature_importances];
    sorted_features, sorted_importances = zip(*sorted_feature_importances)

    # Sum of all importances
    sum_imps = 0  
    for i in range(len(_feature_names)):
        sum_imps = sum_imps + importances[i]
    if dictionary['verbosity'] > 1: print('Sum of the Random Forest\'s importances: ', round(sum_imps, 4))
    
    # Visualize the variables importances sorted in plot
    plt.figure(figsize = (2 + len(importances) * 0.25, 5))
    plt.bar(sorted_features, sorted_importances, orientation = 'vertical')
    #plt.bar(_feature_names, importances, orientation = 'vertical')
    plt.xticks(rotation = 90)
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    if dictionary['plot_fig_title'] == True:
        plt.title('Relative feature importance for RF at lead time ' + str(_lead_time_) + ' week', fontsize = '14')
    plt.grid(axis = 'y', color = '#D3D3D3', linestyle = 'solid')
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + 'train/' + _tgn_ + '/features/RF/'
    if dictionary['cv_type'] == 'nested': save_name = dir_name + 'rf_imp_' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + 'rf_imp_' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    
    # Importances table        
    if dictionary['cv_type'] == 'none' and dictionary['server'] == 'CFC': 
        save_name = dir_name + 'data_rf_imp_' + _tgn_ + '.pdf'        
        df = pd.DataFrame(data = {'lead_time_' + str(_lead_time_): list(np.round(importances, 2)), 
                                  'feature_names': _feature_names})
        df = df.set_index('feature_names')
        if _lead_time_ > dictionary['lead_times'][0]:
            df_old = pd.read_csv(save_name, index_col = 0)
            df = df_old.join(df, how = 'outer')
        df.to_csv(save_name, index = True)  
        if _lead_time_ == dictionary['lead_times'][-1] and dictionary['verbosity'] > 1: 
            # Reorder index for manuscript
            ordered_features_manuscript = [x + '_7D_lag' + str(i) for x in _pred_names for i in np.arange(dictionary['lead_times'][0], dictionary['lead_times'][-1] + dictionary['num_lags'])]
            df = df.reindex(ordered_features_manuscript)
            # Show table
            pd.set_option('display.max_rows', None, 'display.max_columns', None)
            display(df)
            pd.reset_option('display.max_rows', 'display.max_columns')


# In[ ]:


def tree_plotter(_X_train_, _y_train_, _tgn_, _lead_time_, max_depth, min_samples_split, min_samples_leaf, bootstrap, outer_split_num_ = None):

    """
    inputs
    ------
    _X_train_      xr.Dataset : predictors in train dataset
    _y_train_      xr.Dataset : target in train dataset
    _tgn_                 str : name of target
    _lead_time_           int : lead time of prediction
    max_depth             int : maximum depth of the Decission Tree
    min_samples_split     int : minimum samples need at each branch to split a node
    min_samples_leaf      int : minimum samples in final leaves
    bootstrap            bool : Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
    " outer_split_num_          int : counter for outer splits "

    outputs
    -------
    None.  Plots a simple Decision Tree trained on the same data then the Random Forest
    
    """
    
    # Initialize model
    if 'bin' in _tgn_:                 
        dt = DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, bootstrap = bootstrap, random_state = 7, criterion = 'gini')
    elif 'bin' not in _tgn_:
        dt = DecisionTreeRegressor(max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, bootstrap = bootstrap, random_state = 7, criterion = 'mse')
    
    # Train model
    dt.fit(_X_train_, _y_train_)
    
    # Plot tree
    if dictionary['verbosity'] > 1: 
        print('*****************************************************************************************\n',
              'Plot simple Decision Tree trained on the same data then the Rdm Forest') 
    if dictionary['verbosity'] > 2: 
        print('Rule at split node: if condition fulfilled go left, else go right\n',
              'Gini: measure of impurity at the node\n', 
              'Samples: total # samples at the node \n'
              'Value: [# samples without HW, # samples with HW]')
    _feature_names = _X_train_.columns
    plt.figure(figsize = (25,10))
    plot_tree(dt, fontsize = None, feature_names = _feature_names, precision = 2, max_depth = 2)
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + 'train/' + _tgn_ + '/tree_plot/'
    if dictionary['cv_type'] == 'nested': save_name = dir_name + 'tree_plot_' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + 'tree_plot_' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    if dictionary['verbosity'] > 1: print('*****************************************************************************************')
    


# In[ ]:


def plot_random_forest_overfitting(X_train_, y_train_, X_vali_, y_vali_, optimize_metric_, tgn_, _lead_time_, balance_, best_n_estimators_, best_max_depth_, best_min_samples_split_, best_min_samples_leaf_, best_bootstrap_, outer_split_num_ = None):
    
    """
    inputs
    ------
    X_train_             pd.Dataframe : matrix of attributes (time x lagged features) for training
    y_train_                pd.Series : array containing the target variable (time) for training
    X_vali_              pd.Dataframe : matrix of attributes (time x lagged features) for validation
    y_vali_                 pd.Series : array containing the target variable (time) for validation
    optimize_metric_              str : metric to optimize: Regression ('Corr', 'RMSE'), Classification ('G-mean', 'F1' 'precision')
    tgn_                          str : name of target variable
    _lead_time_                   int : lead time of prediction in weeks
    balance_                      str : 'none', 'undersampling' or 'oversampling'
    best_hyperparameters            - : -
    " outer_split_num_          int : counter for outer splits "


    outputs
    -------
    None: Plots the metric against the model complexity (in terms of min_samples_leaf) for both the train and vali set   
    
    """
    
    if dictionary['verbosity'] > 1: print('** Overfitting exploration **')
    # Initialize figure
    plt.figure(figsize = (6,4))
    fig, ax = plt.subplots()
    seed_ = 7
    ## Initialize score_max with its minumum
    if optimize_metric_ == 'RMSE': score_max_vali = 1.
    else: score_max_vali = 0.
        
    ## Iterate over all hyperparameters and choose the best set 
    first_guess_min_samples_leaf_ = best_min_samples_leaf_
    if dictionary['verbosity'] > 3: print('first_guess_min_samples_leaf_: ', first_guess_min_samples_leaf_)
    step_min_samples_leaf_ = int(first_guess_min_samples_leaf_/10)
    if step_min_samples_leaf_ < 1: step_min_samples_leaf_ = 1    
    if dictionary['verbosity'] > 3: print('step_min_samples_leaf_: ', step_min_samples_leaf_)
    bootstrap_ = best_bootstrap_
    n_estimators_ = best_n_estimators_
    if balance_ == 'undersampling': max_depth_ = 14
    else: max_depth_ = 23 # Too large, to control with min_samples_leaf only
    mean_sample_leaves_ = np.arange(first_guess_min_samples_leaf_ - 10*step_min_samples_leaf_, first_guess_min_samples_leaf_ + 15*step_min_samples_leaf_ + 1, step_min_samples_leaf_)
    for min_samples_leaf_ in mean_sample_leaves_: #25 
        if min_samples_leaf_ > 1:
            min_samples_split_ = min_samples_leaf_
            ### Initialize model
            if 'bin' in tgn_:                 
                rf_ = RandomForestClassifier(n_estimators = n_estimators_, max_depth = max_depth_, min_samples_split = min_samples_split_, min_samples_leaf = min_samples_leaf_, bootstrap = bootstrap_, random_state = seed_, criterion = 'gini')
            elif 'bin' not in tgn_:
                rf_ = RandomForestRegressor(n_estimators = n_estimators_, max_depth = max_depth_, min_samples_split = min_samples_split_, min_samples_leaf = min_samples_leaf_, bootstrap = bootstrap_, random_state = seed_, criterion = 'mse')
            ### Train model
            rf_.fit(X_train_, y_train_)
            ### Predict
            if 'bin' in tgn_: 
                pred_train_ = rf_.predict_proba(X_train_)[:,1]
                pred_vali_ = rf_.predict_proba(X_vali_)[:,1]
            else: 
                pred_train_ = rf_.predict(X_train_)
                pred_vali_ = rf_.predict(X_vali_)
            ### Compute score
            score_new_train = compute_score(optimize_metric_, y_train_, pred_train_)
            score_new_vali = compute_score(optimize_metric_, y_vali_, pred_vali_)           
            
            ### Plot points
            plt.scatter(min_samples_leaf_, score_new_train, marker = 'o', c = 'cornflowerblue', label = 'Train')
            plt.scatter(min_samples_leaf_, score_new_vali, marker = 'o', c = 'orange', label = 'Validation')
            ### Compare to previous scores & choose the best one
            if optimize_metric_ == 'RMSE':
                if (score_new_vali < score_max_vali): 
                    score_max_vali = score_new_vali
                    best_n_estimators = n_estimators_
                    best_max_depth = max_depth_
                    best_min_samples_split = min_samples_split_
                    best_min_samples_leaf = min_samples_leaf_
                    best_bootstrap = bootstrap_
            else:
                if (score_new_vali > score_max_vali): 
                    score_max_vali = score_new_vali
                    best_n_estimators = n_estimators_
                    best_max_depth = max_depth_
                    best_min_samples_split = min_samples_split_
                    best_min_samples_leaf = min_samples_leaf_
                    best_bootstrap = bootstrap_
                    
            if dictionary['verbosity'] > 3: print('((n_estimators, max_depth, min_samples_split, min_samples_leaf), ', optimize_metric_, ') : ((', n_estimators_, ', ', max_depth_, ', ', min_samples_split_, ', ', min_samples_leaf_, ', ', bootstrap_, '), {:.3f}'.format(score_new_vali), ')')        

    if dictionary['verbosity'] > 1: print('Best (n_estimators, max_depth, min_samples_split, min_samples_leaf) combination for the RF is: (', best_n_estimators, ', ', best_max_depth, ', ', best_min_samples_split, ', ', best_min_samples_leaf, ', ', best_bootstrap, '), with ', optimize_metric_, ' : {:.3f}'.format(score_max_vali), ')')
       
    ## Vertical line at 0 lag time (prediction)
    plt.axvline(best_min_samples_leaf, color = 'black', label = 'Best model', linewidth = 1, linestyle = '-')
    ## Custom legend
    legend_elements = [Line2D([0], [0], marker = 'o', color = 'cornflowerblue', label = 'Train', markersize = 5),
                   Line2D([0], [0], marker = 'o', color = 'orange', label = 'Validation', markersize = 5)]
    ax.legend(handles = legend_elements, loc = 'best')
    plt.gca().invert_xaxis()
    plt.xlabel('min_samples_leaf (model complexity)$^{-1}$', fontsize = 12)
    plt.ylabel(optimize_metric_, fontsize = 12)
    if dictionary['plot_fig_title'] == True:
        plt.title('Overfitting exploration for ' + tgn_ + ' prediction', fontsize = 14)
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + 'train/' + tgn_ + '/overfitting/' + balance_ + '/'
    if dictionary['cv_type'] == 'nested': save_name = dir_name + 'overfitting_plot_' + tgn_ + '_' + balance_ +'_lead_time_' + str(_lead_time_) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + 'overfitting_plot_' + tgn_ + '_' + balance_ +'_lead_time_' + str(_lead_time_) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')    
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()


# In[ ]:


def plot_scatter_diagrams(X, y, tgn_):
    
    """
    inputs
    ------
    X             pd.Dataframe : matrix of attributes (time x lagged features)
    y                pd.Series : array containing the target variable (time)
    tgn_                   str : name of target variable


    outputs
    -------
    None: Plots scatter diagrams showing the correlation between the predictors and the target  
    
    """
    
    predictor_names = list(X)
    for predictor in predictor_names:
        plt.scatter(X[predictor], y, color = 'cornflowerblue', s = 2, marker = 'o')
        if dictionary['plot_fig_title'] == True:
            plt.title(tgn_ +' as a function of ' + predictor, fontsize = 14)
        plt.xlabel(predictor, fontsize = 14)
        plt.ylabel(tgn_, fontsize = 14)
        plt.grid(True)
        if dictionary['verbosity'] > 0: plt.show()
        plt.close()


# In[7]:


def plot_var_expl(var_expl_, num_pc_choosen, _tgn_, _lead_time_, outer_split_num_ = None):    
    
    """
    inputs
    ------
    var_expl_            list : explained variance of sorted features after PCA
    num_pc_choosen        int : minimum number of principal components whose cumulative explained variance is at least the 
                                min_cum_exp_var threshold specified in the dictionary
    _tgn_                 str : name of target
    _lead_time_           int : lead time of prediction
    " outer_split_num_          int : counter for outer splits "


    outputs
    -------
    None: Plots the explained cumulative variance 
    
    """
    
    plt.figure(figsize = (2 + len(var_expl_) * 0.25, 5))
    plt.bar(['PC{}'.format(i) for i in range(1, len(var_expl_) + 1)], var_expl_, orientation = 'vertical', color = 'lightsteelblue', label = 'Non-selected PCs')
    plt.bar(['PC{}'.format(i) for i in range(1, num_pc_choosen + 1)], var_expl_[:num_pc_choosen], orientation = 'vertical', color = 'cornflowerblue', label = 'Selected PCs')
    plt.xticks(rotation = 90)
    plt.ylabel('Explained variance')
    plt.xlabel('Principal component')
    plt.legend(loc = 'best')
    if dictionary['plot_fig_title'] == True:
        plt.title('Explained variances for principal components after PCA', fontsize = '14')
    plt.grid(axis = 'y', color = '#D3D3D3', linestyle = 'solid')
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + 'train/' + _tgn_ + '/features/PCA/'
    if dictionary['cv_type'] == 'nested': save_name = dir_name + 'PCA_var_expl' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + 'PCA_var_expl' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()


# In[31]:


def plot_data_histogram(t2m_anom, hw_bin_2in7, include_2in7_index):
    
    """
      inputs
      -------
        t2m_anom                             xr.DataArray : 2m temperature anomalies in Central European box (time series)
        hw_bin_2in7                          xr.DataArray : binary summer heat wave index 2 heat wave days in one week
        include_2in7_index                           bool : if True, plots the 2in7 index in yellow on top


      outputs
      -------
        None. Plots histogram of data dsitribution of train_full set, that illustrates the different heatwave definitions. The blue events correspond to the histogram of 
        standarized (μ = 0, σ = 1) temperature anomalies. The vertical blue line marks the mean (μ = 0) of the distribution. The stippled orange (red) 
        line marks +1 (+1.5) standard deviations (σ) from the mean. The +1σ (+1.5σ) index is defined as 1 for the events found on the right of the orange 
        (red) line and as 0 for the events located to the left of it. The yellow events correspond to the histogram of 2in7 events. The vertical yellow 
        line marks the mean (μ) of the distribution.

    """ 
    
    # Time selection
    ## Select season
    (hw_bin_2in7, t2m_anom) = (hw_bin_2in7.loc[hw_bin_2in7['time.month'] <= dictionary['final_month']], 
                                                   t2m_anom.loc[t2m_anom['time.month'] <= dictionary['final_month']])
    (hw_bin_2in7, t2m_anom) = (hw_bin_2in7.loc[dictionary['initial_month'] <= hw_bin_2in7['time.month']], 
                                                   t2m_anom.loc[dictionary['initial_month'] <= t2m_anom['time.month']])
    ## Select train_full set only
    train_full_slice = slice(str(int(dictionary['splits'].sel(dataset = dictionary['dataset'], slice_type = 'train_full', edge = 'start'))), 
                             str(int(dictionary['splits'].sel(dataset = dictionary['dataset'], slice_type = 'train_full', edge = 'end'))))
    hw_bin_2in7 = hw_bin_2in7.sel(time = train_full_slice)
    t2m_anom = t2m_anom.sel(time = train_full_slice)
    # Standarize data again
    t2m_anom = (t2m_anom.groupby('time.dayofyear'))/(t2m_anom.groupby('time.dayofyear').std('time'))
    
    # Plot
    fig = plt.figure(figsize = (9, 5))
    ### The histogram of the data
    n, bins, patches = plt.hist(t2m_anom.values, 200, facecolor = 'cornflowerblue', alpha = 0.75, density = False
                                #, label = 'Std T$_{anom}$ distribution'
                               )
    plt.xlabel('Standarized temperature anomalies ($\sigma$)', fontsize = 16)
    plt.ylabel('Number of events', fontsize = 16)
    ### Best fit of data
    (mu, sigma) = norm.fit(t2m_anom)
    plt.axvline(np.round(mu, decimals = 0), color = 'cornflowerblue', label = 'Std T$_{anom}$ $\mu$=%.0f' %(abs(mu)))

    if dictionary['plot_fig_title'] == True: 
        plt.title(r'$\mathrm{Histogram\ of\ the\ standarized\ temperature\ anomalies:}\ \mu=%.0f,\ \sigma=%.0f$' %(abs(mu), sigma))
    plt.grid(True)
    ### Add 1SD and 1.5SD lines
    plt.axvline(1, color = 'darkorange', label = 'Std T$_{anom}$ at +1$\sigma$', linestyle = '--')
    plt.axvline(1.5, color = 'darkred', label = 'Std T$_{anom}$ at +1.5$\sigma$', linestyle = '--')

    if include_2in7_index == True:
        ### Add 2in7 histogram on top
        hw_bin_2in7_temp = t2m_anom[np.where(hw_bin_2in7 == 1)]
        n, bins, patches = plt.hist(hw_bin_2in7_temp, 200, facecolor = 'gold', alpha = 0.75, 
                                    #label = '2 in 7 index distribution', 
                                    density = False)
        ### Best fit of data
        (mu_2in7, sigma_2in7) = norm.fit(hw_bin_2in7_temp)
        ### Add mean line 2in7 lines 
        plt.axvline(mu_2in7, color = 'gold', label = '2in7 index $\mu$=%.1f' %(abs(mu_2in7)))
    plt.legend(fontsize = 16, loc = 'upper left')
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    ### Save plot
    dir_name = dictionary['path_plots_prediction'] + 'data_visualization/'
    if include_2in7_index == True:
        save_name = dir_name + 'temperature_anomalies_histogram_' + dictionary['timestep'] + '_' + dictionary['mean_type'] + '_mean_all_indices.pdf'
    else:
        save_name = dir_name + 'temperature_anomalies_histogram_' + dictionary['timestep'] + '_' + dictionary['mean_type'] + '_mean.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    


# In[8]:


def plot_lagged_correlation_heatmap(target_pos, y, p_matrix, long_predictor_names, N, tau_max):
    
        # Plot Heat map showing the causality strength of each driver for different time lags        
        ## Prepare plot
        fig = plt.figure(figsize = [tau_max, N/1.5])
        sns.set_theme(palette = 'vlag')
        ### Set axes for new colorbar
        cbaxes = fig.add_axes([1, 0.13, 0.05, 0.75])
        # Draw a heatmap with the numeric values in each cell
        tmin = -.8
        tmax = .8
        ax = fig.add_subplot(111)
        # Define x axis labels
        x_axis_labels = np.arange(1, tau_max + 1, step = 1)
        heat = sns.heatmap(y[1:, target_pos, 1:tau_max + 1], xticklabels = x_axis_labels, annot = True, linewidths = 0.2, fmt = '.2f', ax = ax, cbar_ax = cbaxes, cmap = 'vlag', vmin = tmin, vmax = tmax, norm = MidpointNormalize(midpoint = 0, vmin = tmin, vmax = tmax))
        ## Hatch non-significant pixel i.e. p_value > alpha_level
        hatch_non_significant_cells (ax, p_matrix, target_pos, tau_max)
        ## Add axis labels
        ax.yaxis.set_label_position('right')
        ax.yaxis.set_major_formatter(FuncFormatter(format_fn_y))
        plt.yticks(ticks = np.arange(0, N) + 0.5 * np.ones(N), labels = long_predictor_names, rotation = 0)
        if dictionary['timestep'] == '1D':
            plt.xlabel('Time lag in days', fontsize = 14)
        elif dictionary['timestep'] == '7D':
            plt.xlabel('Time lag in weeks', fontsize = 14)
        elif dictionary['timestep'] == '1M':
            plt.xlabel('Time lag in months', fontsize = 14)
        else:
            plt.xlabel('Time lag step (%s each)'%(dictionary['timestep']), fontsize = 14)
        ### Save plot
        if len(long_predictor_names) == 18: pred_set = 'all_'
        else: pred_set = ''
        dir_name = dictionary['path_plots_prediction'] + 'data_visualization/'
        save_name = dir_name + pred_set + 'lagged_correlations_' + dictionary['timestep'] + '_' + dictionary['mean_type'] + '_mean.pdf'
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        plt.savefig(save_name, bbox_inches = 'tight')
        


# In[ ]:




