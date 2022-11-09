""" HWAI plotting """

# Author: Elizabeth Weirich Benet 
# License: MIT License 2022

# Import packages
import xarray as xr
import pandas as pd
from pandas import Series
import numpy as np
import sklearn
from sklearn.calibration import calibration_curve
import matplotlib                             
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import gridspec
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
from collections import namedtuple
from shapely import geometry
import string
from scipy.stats import norm
import seaborn as sns
import os
import datetime
import itertools
## Metrics
from sklearn.metrics import (precision_recall_curve,
                             roc_curve)

# Import own functions
## Metrics
from metrics import (roc_auc_score, 
                     pr_auc_score)
## Utils
from utils import (filter_dates_like, 
                   final_day_of_month, 
                   MidpointNormalize, 
                   format_fn_y, 
                   hatch_non_significant_cells, 
                   save_to_file,
                   read_old_regr_ml_forecasts)

# Import constants
from const import dictionary


# **----------------------------------------------------------------------------------------------------------------------------------------------**

# Function definitions 

def show_multiple_time_series(dset):
    
    """
      inputs
      -------
        dset                      xr.Dataset : containing multiple time series


      outputs
      -------
        None. Plots the time series for each of the variables in dset one after another. 

    """ 
   
    var_names = list(dset)
    for var_name in var_names:
        plt.figure(figsize = (20,3))
        plt.plot(dset.time, dset[var_name])
        plt.axhline(0, color = 'navy', linewidth = 2, linestyle = '--')
        plt.title(var_name)
        if dictionary['verbosity'] > 0: plt.show()
        plt.close()



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
    


def show_first_x_eofs(eofs_, x):

    """
      inputs
      -------
        eofs_                      xr.Dataset : containing 'x' EOFs i.e. lat-lon maps
        x                                 int : number of EOFs you want to plot


      outputs
      -------
        None. Plots the x first EOFs.

    """ 
    
    for i in range(0,x):
        eof = eofs_.sel(mode = i)
        print('EOF number ', i + 1)
        eof.plot.contourf(vmin = -0.1, vmax = 0.1, levels = 9, cmap = 'coolwarm')
        plt.show()
        plt.close()
        
        

def show_hyperparameters(tgn):
    
    """
    inputs
    ------
    tgn                     str : name of target 

    outputs
    -------
    None.  Prints all hyperparameters (linear + random forest) for the no CV case as table and saves the table to file.
    
    """   
    
    if 'bin' in tgn: 
        rf = 'RFC'
        pred_type = 'classi'
    else: 
        rf = 'RFR'
        pred_type = 'regr'
        
    if dictionary['cv_type'] == 'none': 
        # Read hyperparameters and combine them into dataframe
        ## Initialize
        ds_lts = []
        ## Specify path
        dir_name = dictionary['path_hyperparam'] + 'data/'        
        ## Add hyperparameters for each lead time
        for lt in dictionary['lead_times']:
            file_name_rf = 'rf/best_rf_hyperparam_' + tgn + '_lead_time_' + str(lt) + '.nc'
            file_name_linear = 'linear/best_linear_hyperparam_' + tgn + '_lead_time_' + str(lt) + '.npy'
            ds_lt_rf = xr.open_dataset(dir_name + file_name_rf).sel(forecast = rf)
            ds_lt_linear = xr.Dataset({'value': (('best_hyperparameter', 'lead_time'), np.reshape(float(np.load(dir_name + file_name_linear)), (1,1)))},
                                            coords = {'best_hyperparameter': ['best_alpha'], 'lead_time': [lt]})
            ds_lt = ds_lt_rf.combine_first(ds_lt_linear) 
            ds_lts.append(ds_lt)
    
        ds = xr.concat(ds_lts, dim = 'lead_time')
        df = ds.to_dataframe().drop(['forecast'], axis = 1).transpose()
            
        # Save table to file
        save_to_file(df, dictionary['path_hyperparam']  + 'table/', 'table_best_hyperparam_' + tgn + '.csv', 'csv')
        


        
def plot_pred_time_series(y, predictions_rr, predictions_rfr, persistence, climatology, ecmwf, lead_time, tgn, outer_split_num_ = None):
    
    """
    inputs
    ------
    y                                pd.Series : time series of the true target
    predictions_rr                   np.array  : time series of the predicted target (same length and time as y) by the RR model
    predictions_rfr                  np.array  : time series of the predicted target (same length and time as y) by the RFR model
    persistence                     pd.Series  : time series of the target's persistence forecast (same length and time as y)
    climatology                      np.array  : time series of the target's climatology forecast (same length and time as y)
    ecmwf                         xr.DataArray : time series of the target's ECMWF forecast (same length and time as y)
    lead_time                              int : lead time of prediction in units of timestep
    tgn                                    str : name of target variable 
    " outer_split_num_                     int : counter for outer splits (only for nested CV case) "


    outputs
    -------
    None.  Plots time series for the selected season only for all years.
    
    """

    predictions_rr_df = pd.DataFrame(data = predictions_rr, index = y.index).sort_index()  
    predictions_rfr_df = pd.DataFrame(data = predictions_rfr, index = y.index).sort_index()  
    
    ## Define the ranges for the dates
    drange = [[datetime.date(i, dictionary['initial_month'], 1),datetime.date(i, dictionary['final_month'], 30)] for i in range(y.index[0].year, y.index[-1].year + 1)]
    ## Create as many subplots as there are date ranges
    fig, axes = plt.subplots(ncols = len(drange), sharey = True, figsize = (20,5))
    fig.subplots_adjust(bottom = 0.3, wspace = 0)

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
        
        # Ground truth
        ax.plot(y.sort_index(), label = 'Ground Truth', color = 'black', linewidth = 2)
        # ML predictions
        ax.plot(predictions_rr_df, '--', label = 'RR', color = 'crimson')
        ax.plot(predictions_rfr_df, '--', label = 'RFR', color = 'dodgerblue')
        # Persistence forecast: take target value from date - lead time
        ax.plot(persistence, '--', label = 'Persistence', color = 'lightgray')
        # Climatology forecast: plot the climatology forecast for the target variable
        ax.plot(climatology, '--', label = 'Climatology', color = 'navy')
        # ECMWF forecast: plot the ECMWF forecast for the target variable
        if ecmwf is not None: 
            ax.plot(ecmwf, '--', label = 'ECMWF', color = 'orange')       
        loc = matplotlib.dates.MonthLocator(np.arange(dictionary['initial_month'], dictionary['final_month'] + 1))
        fmt =  matplotlib.dates.DateFormatter("%Y-%b")
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)
        plt.setp(ax.get_xticklabels(), rotation = 90, fontsize = 16)
        if i != 0:
            ax.tick_params(axis = 'y', which = 'both', length = 0)
        if i == 0: ax.set_ylabel('Temperature anomalies (°C)', fontsize = 16)           
        ax.yaxis.set_tick_params(labelsize = 16)
        
    ax.legend(bbox_to_anchor = (1.1, 1.5), loc = 'upper right')
    if dictionary['plot_fig_title'] == True:
        fig.suptitle('Predicted and true ' + tgn + ' on test set at ' + str(lead_time) + ' weak(s) lead time', fontsize = 14)
    ## Save plot
    dir_name = dictionary['path_plots'] + 'test/regr/time_series/individual/'
    if dictionary['cv_type'] == 'nested': save_name = dir_name + 'pred_time_series_' + tgn + '_lead_time_' + str(lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + 'pred_time_series_' + tgn + '_lead_time_' + str(lead_time) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    
    
    
    
def plot_zoomed_in_pred_time_series(y, predictions_rr, predictions_rfr, persistence, climatology, ecmwf, start_date_test, lead_time, tgn, outer_split_num_ = None):

    """
    inputs
    ------
    y                                pd.Series : time series of the true target
    predictions_rr                    np.array : time series of the predicted target (same length and time as y) by the RR model
    predictions_rfr                   np.array : time series of the predicted target (same length and time as y) by the RFR model
    persistence                      pd.Series : time series of the target's persistence forecast (same length and time as y)
    climatology                       np.array : time series of the target's climatology forecast (same length and time as y)
    ecmwf                         xr.DataArray : time series of the target's ECMWF forecast
    start_date_test                        str : start date of test time period
    lead_time                              int : lead time of prediction in units of timestep
    tgn                                    str : name of target variable 
    " outer_split_num_                     int : counter for outer splits (only for nested CV case) "

    outputs
    -------
    None.  Plots the zoomed-in time series (only one year). 
    
    """

    predictions_rr_df = pd.DataFrame(data = predictions_rr, index = y.index).sort_index()  
    predictions_rfr_df = pd.DataFrame(data = predictions_rfr, index = y.index).sort_index()  
    climatology_df = pd.DataFrame(data = climatology, index = y.index).sort_index()  
    
    # Initialize figure
    plt.figure(figsize = (20,5))
    # Find zoom period
    ## Select 2nd year of test set: 
    zoom_year = str(int(start_date_test[:4]) + 1)
    zoom_period_start = zoom_year + '-' + str(dictionary['initial_month']).zfill(2) + '-01'
    zoom_period_end = zoom_year + '-' + str(dictionary['final_month']).zfill(2) + '-' + str(final_day_of_month(dictionary['final_month'])).zfill(2)
    
    if dictionary['verbosity'] > 2: 
        print('zoom_period_start: ', zoom_period_start)
        print('zoom_period_end: ', zoom_period_end)
    # Ground truth
    plt.plot(y.sort_index().loc[zoom_period_start:zoom_period_end], label = 'Ground Truth', color = 'black', linewidth = 2)
    # ML predictions
    plt.plot(predictions_rr_df.loc[zoom_period_start:zoom_period_end], '--', label = 'RR', color = 'crimson')
    plt.plot(predictions_rfr_df.loc[zoom_period_start:zoom_period_end], '--', label = 'RFR', color = 'dodgerblue')
    # Persistence forecast: take target value from date - lead time
    plt.plot(persistence.loc[zoom_period_start:zoom_period_end], '--', label = 'Persistence', color = 'lightgray')
    # Climatology forecast: plot the climatology forecast for the target variable
    plt.plot(climatology_df.loc[zoom_period_start:zoom_period_end], '--', label = 'Climatology', color = 'navy')
    # ECMWF forecast: plot the ECMWF forecast for the target variable
    if ecmwf is not None: 
        ecmwf = ecmwf[tgn + '_mean']
        plt.plot(ecmwf.loc[zoom_period_start:zoom_period_end], '--', label = 'ECMWF', color = 'orange')
    
    plt.xlabel('Time (year-month)', fontsize = 12)
    plt.ylabel('Temperature anomalies (°C)', fontsize = 12)
    plt.legend(loc = 'best')
    if dictionary['plot_fig_title'] == True:
        plt.title('Zoomed in: ' + str(zoom_year) + ' for the predicted and true ' + tgn + ' on test set at ' + str(lead_time) + ' weak(s) lead time', fontsize = 14)
    ## Save plot
    dir_name = dictionary['path_plots'] + 'test/regr/time_series/individual_zoomed_in/'  
    if dictionary['cv_type'] == 'nested': save_name = dir_name + 'zoomed_in_pred_time_series_' + tgn + '_lead_time_' + str(lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + 'zoomed_in_pred_time_series_' + tgn + '_lead_time_' + str(lead_time) + '_weeks.pdf'    
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    
    
    

def plot_pred_time_series_all_lead_times():
    
    """
    inputs
    ------
    y                               pd.Series : time series of the true target
    predictions_rr                  np.array  : time series of the predicted target (same length and time as y) by the RR model
    predictions_rfr                 np.array  : time series of the predicted target (same length and time as y) by the RFR model
    persistence                    pd.Series  : time series of the target's persistence forecast (same length and time as y)
    climatology                     np.array  : time series of the target's climatology forecast (same length and time as y)
    ecmwf                        xr.DataArray : time series of the target's ECMWF forecast (same length and time as y)
    lead_time                             int : lead time of prediction in units of timestep
    tgn                                   str : name of target variable 
    " outer_split_num_                    int : counter for outer splits (only for nested CV case) "


    outputs
    -------
    None.  Plots time series for the selected season only for all years and all lead times in a multiplot format. 
    
    """

    # Extract data LT-indep data
    path = dictionary['path_time_series'] + 'test/t2m/'
    ## GT
    gt = np.load(path + 't2m_GT_test.npy')
    index = pd.DatetimeIndex(np.load(path + 't2m_index_test.npy'))
    
    # Prepare plot
    ## Define the ranges for the dates
    drange = [[datetime.date(i, dictionary['initial_month'], 1),
               datetime.date(i, dictionary['final_month'], 30)] for i in range(index[0].year, index[-1].year + 1)]
    ## Create as many subplots as there are date ranges
    fig, axes = plt.subplots(nrows = len(dictionary['lead_times']), ncols = len(drange), sharey = True, figsize = (20, 5 * len(dictionary['lead_times'])))
    fig.subplots_adjust(bottom = 0.3, wspace = 0)
        
    for lt in dictionary['lead_times']:
        ## Read forecasts for specific lead time
        pred_rr_full, pred_rr_ensemble, pred_rfr_full, pred_rfr_ensemble = read_old_regr_ml_forecasts('t2m', lt)
        pred_rr = pred_rr_full['test']
        pred_rfr = pred_rfr_full['test']
        persist = np.load(path + 't2m_persistence_test_lead_time_' + str(lt) + '_weeks.npy') 
        clim = np.load(path + 't2m_climatology_test_lead_time_' + str(lt) + '_weeks.npy') 
        ecmwf = np.load(path + 't2m_ECMWF_test_lead_time_' + str(lt) + '_weeks.npy') 
        ecmwf_index = np.load(path + 't2m_ECMWF_index_test_lead_time_' + str(lt) + '_weeks.npy') 
    
        # Make dataframes
        ground_truth = pd.DataFrame(data = gt, index = index).sort_index() 
        predictions_rr = pd.DataFrame(data = pred_rr, index = index).sort_index()  
        predictions_rfr = pd.DataFrame(data = pred_rfr, index = index).sort_index()  
        persistence = pd.DataFrame(data = persist, index = index).sort_index()  
        climatology = pd.DataFrame(data = clim, index = index).sort_index()  
        ecmwf = pd.DataFrame(data = ecmwf, index = ecmwf_index).sort_index()   

        if ecmwf is not None:
            ## Select ensemble mean
            ecmwf = ecmwf[0]
            ymax = ecmwf.max()
            ymin = ecmwf.min()
        else:
            ymax = ground_truth.max()
            ymin = ground_truth.min()

        ## Loop over subplots and limit each to one date range
        for i, ax in enumerate(axes[lt - 1,:]):
            ax.set_xlim(drange[i][0],drange[i][1])

            # Ground truth
            ax.plot(ground_truth, label = 'Ground Truth', color = 'dimgray', linewidth = 2)
            # ML predictions
            ax.plot(predictions_rr, '--', label = 'RR', color = 'crimson')
            ax.plot(predictions_rfr, '--', label = 'RFR', color = 'dodgerblue')
            # Persistence forecast: take target value from date - lead time
            ax.plot(persistence, '--', label = 'Persistence', color = 'lightgray')
            # Climatology forecast: plot the climatology forecast for the target variable
            ax.plot(climatology, '--', label = 'Climatology', color = 'navy')
            # ECMWF forecast: plot the ECMWF forecast for the target variable
            if ecmwf is not None: 
                ax.plot(ecmwf, '--', label = 'ECMWF', color = 'orange')   
                
            # Format
            # x-axis ticks and labels 
            if lt == dictionary['lead_times'][-1]: 
                loc = matplotlib.dates.MonthLocator(np.arange(dictionary['initial_month'], dictionary['final_month'] + 1))
                fmt =  matplotlib.dates.DateFormatter("%Y-%b")
                ax.xaxis.set_major_locator(loc)
                ax.xaxis.set_major_formatter(fmt)
                plt.setp(ax.get_xticklabels(), rotation = 90, fontsize = 14)
            else: ax.xaxis.set_ticklabels([])
            # y-axis and labels 
            if i != 0: ax.tick_params(axis = 'y', which = 'both', length = 0)
            if i == 0:
                if lt == int(len(dictionary['lead_times'])/2): ax.set_ylabel('Temperature anomalies (°C)', fontsize = 16)  
                else: ax.set_ylabel(' ', fontsize = 20) 
            ax.yaxis.set_tick_params(labelsize = 16)
            ## ABC Labels  
            axes[lt - 1, 0].text(-0.15, 1.05, string.ascii_lowercase[lt - 1], transform = axes[lt - 1, 0].transAxes, size = 18, weight = 'bold')

        # Add legend 
        if lt == dictionary['lead_times'][-1]: ax.legend(bbox_to_anchor = (-8, .0001), loc = 'lower left', fontsize = 14)
        
    ## Title
    if dictionary['plot_fig_title']:
        fig.suptitle('Predicted and true 2m-air temperature on test set at 1--6 weaks lead time', fontsize = 14)
    ## Save plot
    dir_name = dictionary['path_plots'] + 'test/regr/time_series/all_lead_times/'
    save_name = dir_name + 'pred_time_series_t2m_all_lead_times.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()


    
def plot_cross_corr(y, predictions_rr, predictions_rfr, predictions_ecmwf, persistence, lead_time, tgn, outer_split_num_ = None):
    
    """
    inputs
    ------
    y                             pd.Series : time series of the true target
    predictions_rr                np.array  : time series of the predicted target (same
                                              length and time as y) by the RR model
    predictions_rfr               np.array  : time series of the predicted target (same
                                              length and time as y) by the RFR model
    predictions_ecmwf             np.array  : time series of the predicted target (same
                                              length and time as y) by the RFR model
    persistence                   np.array  : time series of the target's persistence (same
                                              length and time as y)
    lead_time                           int : lead time of prediction in units of timestep
    tgn                                 str : name of target variable 
    " outer_split_num_                  int : counter for outer splits (only for nested CV case) "

    outputs
    -------
    None.  Displays the lagged correlations plot
    
    """
    
    maxlagnum = 7 * dictionary['timestep_num']
    plt.figure(figsize = (8,5))
    
    # Correlations to Ground Truth & to persistence
    ## RR
    plt.xcorr(y, 
              predictions_rr, 
              maxlags = maxlagnum, 
              normed = True, 
              color = 'crimson', 
              label = 'RR-Ground Truth', 
              linewidth = 1, 
              usevlines = False, 
              linestyle = '-', 
              marker = 'o', 
              markersize = 5)
    plt.xcorr(persistence, 
              predictions_rr, 
              maxlags = maxlagnum, 
              normed = True, 
              color = 'crimson', 
              label = 'RR-Persistence', 
              linewidth = 1, 
              usevlines = False, 
              linestyle = '--', 
              marker = 'o', 
              markerfacecolor = 'none', 
              markersize = 5)
    ## RFR
    plt.xcorr(y, 
              predictions_rfr, 
              maxlags = maxlagnum, 
              normed = True, 
              color = 'dodgerblue', 
              label = 'RFR-Ground Truth', 
              linewidth = 1, 
              usevlines = False, 
              linestyle = '-', 
              marker = 'o', 
              markersize = 5)
    plt.xcorr(persistence, 
              predictions_rfr, 
              maxlags = maxlagnum, 
              normed = True, 
              color = 'dodgerblue', 
              label = 'RFR-Persistence', 
              linewidth = 1, 
              usevlines = False, 
              linestyle = '--', 
              marker = 'o', 
              markerfacecolor = 'none', 
              markersize = 5)
    ## ECMWF
    if predictions_ecmwf is not None: 
        predictions_ecmwf = predictions_ecmwf[tgn + '_mean']
        plt.xcorr(filter_dates_like(y, predictions_ecmwf), 
                  predictions_ecmwf, 
                  maxlags = maxlagnum, 
                  normed = True, 
                  color = 'orange', 
                  label = 'ECMWF-Ground Truth', 
                  linewidth = 1, 
                  usevlines = False, 
                  linestyle = '-', 
                  marker = 'o', 
                  markersize = 5)  
        plt.xcorr(filter_dates_like(persistence, predictions_ecmwf), 
                  predictions_ecmwf, 
                  maxlags = maxlagnum, 
                  normed = True, 
                  color = 'orange', 
                  label = 'ECMWF-Persistence', 
                  linewidth = 1, 
                  usevlines = False, 
                  linestyle = '--', 
                  marker = 'o', 
                  markerfacecolor = 'none',
                  markersize = 5)  
        
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
    plt.legend()
    ## Save plot
    dir_name = dictionary['path_plots'] + 'test/regr/cross_correlation/'    
    if dictionary['cv_type'] == 'nested': 
        save_name = dir_name + 'cross_corr_' + tgn + '_lead_time_' + str(lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: 
        save_name = dir_name + 'cross_corr_' + tgn + '_lead_time_' + str(lead_time) + '_weeks.pdf'    
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    
    

def plot_roc_curve(fpr, tpr, subset, tg_name, model_name_, _lead_time, best_th_ix = None, outer_split_num_ = None, inner_split_num_ = None):
    
    """
    inputs
    ------
    fpr                        ndarray : False Positive Rate for each probability threshold               
    tpr                        ndarray : True Positive Rate for each probability threshold  
    subset                         str : 'vali', 'train', 'train_full', 'test'
    tg_name                        str : target name
    model_name_                    str : name of the ML model 
    _lead_time                     int : lead time for prediction in units of timestep
    " best_th_ix                   int : index of best probability threshold "
    " outer_split_num_             int : counter for outer splits (only for nested CV case) "
    " inner_split_num_             int : counter for inner splits (only for nested CV case) "
    
    outputs
    -------
    None. Plots the ROC AUC curve with or without best threshold
    
    """           
    
    # Plot the ROC curve for the model
    plt.figure(figsize = (5,5))
    plt.plot([0,1], [0,1], linestyle = '--', label = 'No Skill')
    plt.plot(fpr, tpr, label = 'Prediction')
    if best_th_ix: 
        if isinstance(fpr, pd.Series) or isinstance(fpr, pd.DataFrame):
            plt.scatter(fpr.iloc[best_th_ix], tpr.iloc[best_th_ix], marker = 'o', color = 'black', label = 'Best')
        else:
            plt.scatter(fpr[best_th_ix], tpr[best_th_ix], marker='o', color='black', label='Best')
    ## Axis labels
    plt.xlabel('FPR', fontsize = 12)
    plt.ylabel('TPR', fontsize = 12)
    plt.legend(loc = 'lower right')
    plt.title('ROC curve', fontsize = 14)
    ## Save plot
    dir_name = dictionary['path_plots'] + subset + '/proba_classi/' + tg_name + '/ROC/' + model_name_ + '/'
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



def plot_pr_curve(precision, recall, subset, tg_name, model_name_, _lead_time, best_th_ix = None, outer_split_num_ = None, inner_split_num_ = None):

    """
    inputs
    ------
    precision                  ndarray : precision for each probability threshold               
    recall                     ndarray : recall for each probability threshold  
    subset                         str : 'vali', 'train', 'train_full', 'test'
    tg_name                        str : target name
    model_name_                    str : name of the ML model 
    _lead_time                     int : lead time for prediction in units of timestep
    " best_th_ix                   int : index of best probability threshold "
    " outer_split_num_             int : counter for outer splits (only for nested CV case) "
    " inner_split_num_             int : counter for inner splits (only for nested CV case) "
    
    outputs
    -------
    None. Plots the PR AUC curve with or without best threshold
    
    """      
           
    # Plot the PR curve for the model
    plt.figure(figsize = (5,5))
    plt.plot(recall, precision, label = 'Prediction')
    if best_th_ix: 
        if isinstance(recall, pd.Series) or isinstance(recall, pd.DataFrame):
            plt.scatter(recall.iloc[best_th_ix], precision.iloc[best_th_ix], marker='o', color='black', label='Best')
        else:
            plt.scatter(recall[best_th_ix], precision[best_th_ix], marker = 'o', color = 'black', label = 'Best')    
    ## Axis labels
    plt.xlabel('Recall', fontsize = 12)
    plt.ylabel('Precision', fontsize = 12)
    plt.legend()
    ## Save plot
    dir_name = dictionary['path_plots'] + subset + '/proba_classi/' + tg_name + '/PR/' + model_name_ + '/'
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
    


def plot_calibration_curve(y_true, y_prob, subset, tg_name, model_name_, _lead_time, outer_split_num_ = None, inner_split_num_ = None):

    """
    inputs
    ------
    y_true                   pd.Series : binary ground truth time series              
    y_prob                   pd.Series : probabilities of belonging to the class 1 (being a HW) time series  
    subset                         str : 'vali', 'train', 'train_full', 'test'
    tg_name                        str : target name
    model_name_                    str : name of the ML model 
    _lead_time                     int : lead time for prediction in units of timestep
    " outer_split_num_             int : counter for outer splits (only for nested CV case) "
    " inner_split_num_             int : counter for inner splits (only for nested CV case) "
    
    outputs
    -------
    None. Plots the calibration (or reliability) curve.
    
    """ 
    
    # Compute curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins = 10)

    # Plot curve 
    ## Initialize figure 
    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(111)
    ## Add reference line
    plt.plot([0, 1], [0, 1], linestyle = 'dashed', label = 'well-calibrated')
    ## Plot
    plt.plot(mean_predicted_value, fraction_of_positives)
    fig.suptitle('Calibration curve', fontsize = 14)
    ax.set_xlabel('Predicted probability frequency', fontsize = 14)
    ax.set_ylabel('Fraction of observed positives', fontsize = 14)
    plt.legend(loc = 'lower right')
    ## Save plot
    dir_name = dictionary['path_plots'] + subset + '/proba_classi/' + tg_name + '/reliability_curve/' + model_name_ + '/'
    if dictionary['cv_type'] == 'nested': 
        save_name = dir_name + 'reliability_curve_' + model_name_ + '_' + tg_name + '_lead_time_' + str(_lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '_inner_split_' + str(inner_split_num_) + '.pdf'
    else: save_name = dir_name + 'reliability_curve_' + model_name_ + '_' + tg_name + '_lead_time_' + str(_lead_time) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name)
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    


def plot_proba_histogram(pred_proba_, subset, tgn_, _model_name_, _lead_time_, best_threshold = None, outer_split_num_ = None, inner_split_num_ = None):

    """
    inputs
    ------
    pred_proba_              pd.Series : probabilities of belonging to the class 1 (being a HW) time series
    subset                         str : 'vali', 'train', 'train_full', 'test'
    tgn_                           str : target name
    _model_name_                   str : name of the ML model 
    _lead_time_                    int : lead time for prediction in units of timestep
    " best_threshold             float : best probability threshold "  
    " outer_split_num_             int : counter for outer splits (only for nested CV case) "
    " inner_split_num_             int : counter for inner splits (only for nested CV case) "
    
    outputs
    -------
    None. Plots the histogram of heat wave probabilities and marks the best threshold
    
    """    
    
    # Plot histogram
    fig = plt.figure(figsize = (6, 4))
    plt.xlim([0, 1])
    plt.hist(pred_proba_, bins = 100, facecolor = 'orange', alpha = 0.75)
    plt.xlabel('Probability of having a heat wave', fontsize = 12)
    plt.ylabel('Number of events', fontsize = 12)
    ## Vertical line at best_threshold
    if best_threshold: 
        plt.axvline(best_threshold, color = 'darkred', label = 'Best threshold', linewidth = 1, linestyle = '-')
        plt.legend(loc = 'best')
    if dictionary['plot_fig_title']:
        plt.title('Histogram of heat wave probabilities')
    plt.grid(True)
    ## Save plot
    dir_name = dictionary['path_plots'] + subset + '/proba_classi/' + tgn_ + '/probability_histogram/' + _model_name_ + '/'
    if dictionary['cv_type'] == 'nested': 
        save_name = dir_name + 'hist_' + _model_name_ + '_' + tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks_outer_split_' + str(outer_split_num_) + '_inner_split_' + str(inner_split_num_) + '.pdf'
    else: save_name = dir_name + 'hist_' + _model_name_ + '_' + tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    if subset != 'vali': plt.savefig(save_name)
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()

    

def classification_plots(y, pred_proba, subset_, tgn_, model_name, lt_, best_th_ = None, _outer_split_num_ = None):
    
    """
    inputs
    ------
    y                        pd.Series : binary ground truth time series  
    pred_proba               pd.Series : probabilities of belonging to the class 1 (being a HW) time series
    subset_                        str : time period (e.g., 'test')
    tgn_                           str : target name
    model_name                     str : name of the ML model ('RC' or 'RFC')
    lt_                            int : lead time for prediction in units of timestep
    " best_th_                   float : best probability threshold "  
    " _outer_split_num_            int : counter for outer splits (only for nested CV case) "
    
    outputs
    -------
    None. Plots several classification plots.
    
    """  
    
    ### Calibration curve
    plot_calibration_curve(y, pred_proba, subset_, tgn_, model_name, lt_, outer_split_num_ = _outer_split_num_)        
    ### Probability histogram
    plot_proba_histogram(pred_proba, subset_, tgn_, model_name, lt_, best_threshold = best_th_, outer_split_num_ = _outer_split_num_)  
    ### ROC 
    #### ROC AUC score
    if dictionary['verbosity'] > 2: print('ROC AUC score = %.3f' % (roc_auc_score(y, pred_proba)))
    ### ROC curve 
    fpr, tpr, thresholds = roc_curve(y, pred_proba)
    if best_th_: best_th_ix_ = (np.abs(thresholds - best_th_)).argmin()        
    else: best_th_ix_ = None
    plot_roc_curve(fpr, tpr, subset_, tgn_, model_name, lt_, best_th_ix = best_th_ix_, outer_split_num_ = _outer_split_num_)          
    ### PR 
    #### PR AUC score
    if dictionary['verbosity'] > 2: print('PR AUC score = %.3f' % (pr_auc_score(y, pred_proba)))
    #### PR curve 
    precision, recall, thresholds = precision_recall_curve(y, pred_proba)
    if best_th_: best_th_ix_ = (np.abs(thresholds - best_th_)).argmin()        
    else: best_th_ix_ = None
    plot_pr_curve(precision, recall, subset_, tgn_, model_name, lt_, best_th_ix = best_th_ix_, outer_split_num_ = _outer_split_num_) 
    


def plot_class_distr(index, tgn_):

    """
    inputs
    ------
    index                 xr.DataArray : binary heat wave index time series         
    tgn_                           str : target name
    
    outputs
    -------
    None. Plots the class distribution of heat wave index in time 
    
    """    
    
    fig = plt.figure(figsize = (15, 4))
    #plt.hist(index, bins = 100, facecolor = 'orange', alpha = 0.75)
    index.plot(linewidth = 0.3, color = 'black')
    plt.xlabel('# Events', fontsize = 12)
    plt.ylabel('Time', fontsize = 12)
    if dictionary['plot_fig_title']:
        plt.title('Probability class distribution for '+ tgn_)
    plt.grid(True)
    ## Save plot
    dir_name = dictionary['path_plots'] + 'data_visualization/'
    save_name = dir_name + 'class_distr_' + tgn_ + '.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name)   
    
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    


def plot_tp(metrics, ax, metrics_std = None):
    
    """
    inputs
    ------
    metrics                  xr.Dataset : dataset of metrics data with dimensions (forecast x metric)
    ax                  matplotlib axis : location to place the TPR bar plot next to the other binary classification metrics
    "metrics_std             xr.Dataset : dataset of the standard deviation of the ensemble of metrics with dimensions (forecast x metric) (only for nested CV)"


    outputs
    -------
    None.  Plots a bar chart with TPR and FPR to evaluate (and compare) the performance of the models. 
           x-axis: lead time, y-axis: TPR (coloured bars) and FPR (stippled bars)
    
        TPR: True Positive Rate (events that were predicted and detected)
        FPR: False Positive Rate (events that were predicted but not detected)
    
    """
        
    # Prepare plot    
    ## Bar width 
    bar_width = 0.12
    ## x axis
    xlabels = list(metrics.lead_time.values)
    ### Set label locations
    x = np.arange(len(xlabels)) 
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation = 0)
    
    ## Define models, colors...
    reference_models = ['Ground Truth', 'Climatology', 'Persistence']    
    ml_models = ['RC', 'RFC']
    forecasts_std = ['RC_std', 'RFC_std']
    colors = ['dimgray', 'navy', 'lightgray', 'darkred', 'teal']    
    
    ## Include ECMWF
    if 'ECMWF' in metrics.forecast:
        colors.insert(len(reference_models), 'orange')
        reference_models.append('ECMWF')
        forecasts_std.append('ECMWF_std')
    
    ## Define positions of bars
    bar_shifts = np.linspace(1 - len(reference_models), len(ml_models), len(reference_models) + len(ml_models), dtype = int) 
       
    ## Plot
    all_model_characteristics = reference_models + ml_models    
    for model_name, color, bar_shift in zip(all_model_characteristics, colors, bar_shifts):
        # Bar plot
        ax.bar(x + bar_shift * bar_width, 
               metrics.sel(forecast = model_name).sel(metric = 'TPR').to_array().values.flatten(),
               label = model_name,
               color = color, 
               width = bar_width)        
    for model_name, color, bar_shift, i in zip(all_model_characteristics, colors, bar_shifts, range(len(colors))):
        # Hatch
        ax.bar(x + bar_shift * bar_width, 
               metrics.sel(forecast = model_name).sel(metric = 'FPR').to_array().values.flatten(),
               label = 'FPR' if i == 0 else '', 
               edgecolor = 'black', 
               hatch = '//', 
               width = bar_width, 
               facecolor = 'none') 
        # Add error bars representing uncertainty
        if dictionary['cv_type'] == 'nested':   
            if metrics_std.sel(forecast = model_name).sel(metric = 'TPR').to_array().values.flatten().any() > 0.0001:
                ax.errorbar(x + bar_shift * bar_width, 
                            metrics.sel(forecast = model_name).sel(metric = 'TPR').to_array().values.flatten(),
                            yerr = metrics_std.sel(forecast = model_name).sel(metric = 'TPR').to_array().values.flatten(),
                            fmt = 'o', color = 'black', capsize = 4, linewidth = 1, markersize = 2)   
        elif dictionary['cv_type'] == 'none': 
            for forecasts_std_substr in forecasts_std:
                if model_name in forecasts_std_substr:
                    ax.errorbar(x + bar_shift * bar_width, 
                            metrics.sel(forecast = model_name).sel(metric = 'TPR').to_array().values.flatten(),
                            yerr = metrics.sel(forecast = model_name + '_std').sel(metric = 'TPR').to_array().values.flatten(),
                            fmt = 'o', color = 'black', capsize = 4, linewidth = 1, markersize = 2)

    ## Ticks
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)




def plot_metrics(metrics_full, prediction_type):
    
    """
    inputs
    ------
    metrics_full            dict of xr.Dataset : dataset of metrics data with dimensions (forecast x metric x lead_time "x statistics (only for nested CV)")
    prediction_type                        str : 'regr', 'classi' or 'proba_classi'


    outputs
    -------
    None.  Plots M graphs (one for each metric) to evaluate (and compare) the performance of the models. 
           x-axis: lead time, y-axis: the metric
    
                regr:    RMSE: Root Mean Squared Error
                         Corr: Correlation
        
        proba_classi: ROC AUC: Receiver Operating Characteristic Area Under Curve
                           BS: Brier score 
        
              classi:     EDI: Extremal Dependence Index
                      TPR-FPR: True Positive Rate and False Positive Rate (bar plot)
        
    
    """
                
    # Select prediction type
    metrics_full = metrics_full[prediction_type]
    
    # Separate mean and std
    if dictionary['cv_type'] == 'nested': 
        metrics = metrics_full.sel(statistics = 'mean')
        metrics_std = metrics_full.sel(statistics = 'std')
    else: metrics = metrics_full
    
    # Prepare metrics
    if prediction_type == 'regr': 
        metrics = metrics.sel(target_name = dictionary['target_names_regr'][0])
        if dictionary['cv_type'] == 'nested': metrics_std = metrics_std.sel(target_name = dictionary['target_names_regr'][0])
    
    # Select metrics to plot
    if prediction_type == 'regr': metrics_names = ['RMSE', 'Correlation']
    elif prediction_type == 'proba_classi': metrics_names = ['BS', 'ROC AUC']
    elif prediction_type == 'classi': metrics_names = ['EDI', 'TPR']
    
    # Define models, forecasts, and colors
    reference_models = ['Ground Truth', 'Climatology', 'Persistence']
    if prediction_type == 'regr': 
        forecasts_std = ['RR_std', 'RFR_std']
        colors = ['dimgray', 'navy', 'lightgray', 'crimson', 'dodgerblue']
    elif prediction_type != 'regr': 
        forecasts_std = ['RC_std', 'RFC_std']
        colors = ['dimgray', 'navy', 'lightgray', 'darkred', 'teal']
        
    # Add ECMWF
    if 'ECMWF' in metrics.forecast:
            colors.insert(len(reference_models), 'orange')
            reference_models.append('ECMWF')
            forecasts_std.append('ECMWF_std')
    
    # Prepare plot     
    if prediction_type == 'regr': rows = len(dictionary['target_names_regr'])
    else: rows = len(dictionary['target_names_classi'])           
    columns = len(metrics_names)
    if prediction_type == 'classi': 
        fig = plt.figure(figsize = (6 * len(metrics_names) + 2, 4 * rows))
        gs = gridspec.GridSpec(rows, columns, width_ratios = [1, 1.5]) 
    else: fig = plt.figure(figsize = (6 * len(metrics_names), 4 * rows))
    
    # REGRESSION
    if prediction_type == 'regr': 
        ## Loop over all metrics    
        for metric, index in zip(metrics_names, range(1, columns * rows + 1)):
            ax = fig.add_subplot(rows, columns, index)  
            # Plot lines and uncertainty            
            if prediction_type == 'regr': 
                forecasts = reference_models + ['RR', 'RFR']
                forecasts_labels = forecasts
            elif prediction_type != 'regr': forecasts = reference_models + ['RC', 'RFC']       
            for (forecast, forecast_label, color) in zip(forecasts, forecasts_labels, colors):
                if dictionary['cv_type'] == 'nested': 
                    plt.errorbar(metrics.lead_time, 
                                 metrics.sel(metric = metric, forecast = forecast).value, 
                                 yerr = metrics_std.sel(metric = metric, forecast = forecast).value,                            
                                 fmt = '-o', 
                                 label = forecast_label, 
                                 color = color, ecolor = color, 
                                 capsize = 4, linewidth = 2, markersize = 4)
                elif dictionary['cv_type'] == 'none':
                    plt.plot(metrics.lead_time, 
                                 metrics.sel(metric = metric, forecast = forecast).value, 
                                 '-o', 
                                 label = forecast_label, 
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
            # Formatting
            plt.xlabel('Prediction lead time in weeks', fontsize = 14)
            plt.xticks(ticks =  metrics.lead_time, fontsize = 14)
            if metric == 'RMSE': plt.ylabel('RMSE (°C)', fontsize = 14)
            else: plt.ylabel(metric, fontsize = 14)
            plt.grid(axis = 'x', color = '#D3D3D3', linestyle = 'solid')
            plt.grid(axis = 'y', color = '#D3D3D3', linestyle = 'solid')
            ## ABC Labels  
            ax.text(-0.15, 1.05, string.ascii_lowercase[index -1], transform = ax.transAxes, size = 18, weight = 'bold')
            ## y-axis limits and ticks
            if metric == 'RMSE': ymin, ymax, step = (0., 3., 0.5)
            elif metric == 'Correlation': ymin, ymax, step = (0.,1., 0.2)
            ax.set_ylim(ymin - step/2, ymax + step/2)
            if dictionary['cv_type'] == 'nested':
                if metric == 'Correlation': ax.set_ylim(ymin - step, ymax + step/2)
                if metric == 'RMSE': ax.set_ylim(ymin - step/2, ymax + step)
            yticks = np.arange(ymin, ymax + step, step)
            ax.set_yticks(yticks)
            plt.yticks(fontsize = 14)

    # CLASSIFICATION
    index = 1
    if prediction_type != 'regr':
        ## Loop over all target names
        for tgn in dictionary['target_names_classi']:
            ## Loop over all metrics    
            for metric in metrics_names:
                if prediction_type == 'classi': ax = plt.subplot(gs[index - 1])
                else: ax = fig.add_subplot(rows, columns, index)  
                if metric != 'TPR':          
                    forecasts = reference_models + ['RC', 'RFC'] 
                    for (forecast, color) in zip(forecasts, colors):
                        if dictionary['cv_type'] == 'nested': 
                            plt.errorbar(metrics.lead_time, 
                                         metrics.sel(metric = metric, forecast = forecast, target_name = tgn).value, 
                                         yerr = metrics_std.sel(metric = metric, forecast = forecast, target_name = tgn).value,                            
                                         fmt = '-o', 
                                         label = forecast, 
                                         color = color, ecolor = color, 
                                         capsize = 4, linewidth = 2, markersize = 4)
                        elif dictionary['cv_type'] == 'none':
                            plt.plot(metrics.lead_time, 
                                     metrics.sel(metric = metric, forecast = forecast, target_name = tgn).value, 
                                     '-o', 
                                     label = forecast, 
                                     color = color, 
                                     linewidth = 2, markersize = 4)  
                            if forecast + '_std' in forecasts_std:
                                plt.errorbar(metrics.lead_time, 
                                             metrics.sel(metric = metric, forecast = forecast, target_name = tgn).value, 
                                             yerr = metrics.sel(metric = metric, forecast = forecast + '_std', target_name = tgn).value,                            
                                             fmt = '-o', 
                                             label = None, 
                                             color = color, ecolor = color, 
                                             capsize = 4, linewidth = 2, markersize = 4)                      


                # Add TPR-FPR plot for binary classification
                if prediction_type == 'classi' and metric == 'TPR':
                    if dictionary['cv_type'] == 'nested': plot_tp(metrics.sel(target_name = tgn), ax, metrics_std = metrics_std.sel(target_name = tgn))
                    else: plot_tp(metrics.sel(target_name = tgn), ax)
          
                # Formatting
                if index in range(rows * columns - columns + 1, rows * columns + 1): plt.xlabel('Prediction lead time in weeks', fontsize = 14)
                if metric != 'TPR': plt.xticks(ticks =  metrics.lead_time, fontsize = 14)
                if metric == 'RMSE': plt.ylabel('RMSE (°C)', fontsize = 14)
                else: plt.ylabel(metric, fontsize = 14)
                plt.yticks(fontsize = 14)
                if metric != 'TPR':
                    plt.grid(axis = 'x', color = '#D3D3D3', linestyle = 'solid')
                    plt.grid(axis = 'y', color = '#D3D3D3', linestyle = 'solid')
                    ## y-axis limits and ticks                
                    if metric == 'ROC AUC': ymin, ymax, step = (0.4, 1., 0.1)
                    elif metric == 'BS': ymin, ymax, step = (0.,0.3, 0.05)
                    elif metric == 'EDI': ymin, ymax, step = (-1.,1., 0.25)
                    ax.set_ylim(ymin - step/2, ymax + step/2)
                    if dictionary['cv_type'] == 'nested':
                        if metric == 'ROC AUC': ax.set_ylim(ymin - step, ymax + step/2)
                        if metric == 'BS': ax.set_ylim(ymin - step/2, ymax + step)
                    yticks = np.arange(ymin, ymax + step, step)
                    ax.set_yticks(yticks)
                plt.yticks(fontsize = 14)
                ## ABC Labels  
                ax.text(-0.15, 1.05, string.ascii_lowercase[index - 1], transform = ax.transAxes, size = 18, weight = 'bold')
                ## Update index
                index = index + 1  
        
    # Legend
    if rows == 1: y_pos = 1.028
    elif rows == 2: y_pos = 2.23
    if prediction_type == 'classi': x_pos = 1.38
    else: x_pos = 1.54
    plt.legend(bbox_to_anchor = (x_pos, y_pos), loc = 'upper right', fontsize = 14)
            
   # Save plot
    dir_name = dictionary['path_plots'] + 'test/' + prediction_type + '/metrics_plot/'
    if dictionary['cv_type'] == 'nested': save_name = 'metrics_lt_plot_' + prediction_type + '_nested_CV.pdf'
    else: save_name = 'metrics_lt_plot_' + prediction_type + '.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(dir_name + save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()



def show_coeffs(_model_, model_name, _pred_names, _feature_names, _tgn_, _lead_time_, outer_split_num_ = None):
    
    """
    inputs
    ------
    _model_                                fct : trained model
    model_name                             str : 'RR' or 'RC'
    _pred_names                    list of str : names of all predictors
    _feature_names                 list of str : names of all features (lagged predictors) 
    _tgn_                                  str : name of target
    _lead_time_                            int : lead time of prediction in units of timestep
    " outer_split_num_                     int : counter for outer splits (only for nested CV case) "

    outputs
    -------
    None. Prints the regression coefficients of the Ridge Regressor/Classifier, plots them in descending order as a bar plot and saves them to file (.csv format).
    
    """

    coeffs = _model_.coef_
    
    # Print coefficients
    if dictionary['verbosity'] > 1: 
        intercept = _model_.intercept_
        if model_name == 'RC': intercept = intercept[0]
        print('Intercept: {:.4f}'.format(intercept))
        print('\nRegression coefficients: ')           
        if model_name == 'RC': coeffs = coeffs[0]
        feature_coeffs = [(feature, round(coef, 4)) for feature, coef in zip(_feature_names, coeffs)]
        sorted_feature_coeffs = sorted(feature_coeffs, key = lambda x: np.abs(x[1]), reverse = True)
        [print('Predictor: {:20} Coefficient: {}'.format(*pair)) for pair in sorted_feature_coeffs]
    
    if dictionary['verbosity'] > 2:         
        ## Sum of all coeffs
        sum_coeffs = 0  
        for i in range(len(_feature_names)):
            sum_coeffs = sum_coeffs + coeffs[i]
        print('Sum of ' + model_name + ' coeffs: ', round(sum_coeffs, 4))
            
        
    # Plot coefficients
    coef = Series(coeffs, _feature_names).sort_values(key = np.abs)
    coef = np.flip(coef, axis = 0)
    coef.plot(kind = 'bar', figsize = (2 + len(coeffs) * 0.25, 5))
    plt.grid(axis = 'y', color = '#D3D3D3', linestyle = 'solid')
    plt.xticks(rotation = 90)
    plt.ylabel('Regression coefficients')
    plt.xlabel('Feature')
    if dictionary['plot_fig_title']:
        plt.title('Regression Coefficients for ' + model_name + ' at lead time ' + str(_lead_time_) + ' week', fontsize = 14)
    ## Save plot
    if 'bin' in _tgn_: pred_type = 'proba_classi'
    else: pred_type = 'regr'
    dir_name = dictionary['path_plots'] + 'train_full/' + pred_type + '/' + _tgn_ + '/features/' + model_name + '/'
    if dictionary['cv_type'] == 'nested': save_name = dir_name + model_name + '_regr_coeffs_' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + model_name + '_regr_coeffs_' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    
    # Regression coefficients table        
    if dictionary['cv_type'] == 'none': 
        dir_name = dictionary['path_features']
        file_name = model_name + '_regr_coeffs_' + _tgn_ + '.csv'        
        df = pd.DataFrame(data = {'lead_time_' + str(_lead_time_): list(np.round(coeffs, 2)), 
                                  'feature_names': _feature_names})
        df = df.set_index('feature_names')
        ordered_feature_names = [x + '_' + dictionary['timestep'] + '_lag' + str(lag) for x in _pred_names for lag in range(_lead_time_, _lead_time_ + dictionary['num_lags'])]
        df = df.reindex(ordered_feature_names, axis = 0)
        if _lead_time_ > dictionary['lead_times'][0]:
            df_old = pd.read_csv(dir_name + file_name, index_col = 0)
            df_new = df_old.join(df, how = 'outer')
        else: df_new = df
        ordered_feature_names = [x + '_' + dictionary['timestep'] + '_lag' + str(lag) for x in _pred_names for lag in range(dictionary['lead_times'][0], _lead_time_ + dictionary['num_lags'])]
        df_new = df_new.reindex(ordered_feature_names, axis = 0)
        save_to_file(df_new, dir_name, file_name, 'csv')
        



def show_rf_imp(_rf_, _pred_names, _feature_names, _tgn_, _lead_time_, outer_split_num_ = None):

    """
    inputs
    ------            
    _rf_                                 fct : trained RF model
    _pred_names                  list of str : names of all predictors
    _feature_names               list of str : names of all features (lagged predictors)
    _tgn_                                str : name of target
    _lead_time_                          int : lead time of prediction in units of timestep
    " outer_split_num_                   int : counter for outer splits (only for nested CV case) "

    outputs
    -------
    None. Prints the feature importances of the Random Forest, plots them in descending order as a bar plot and saves them to file (.csv format).
    
    """

    # Print feature importances       
    ## Get feature importances, sort them, and print the result
    importances = list(_rf_.feature_importances_)
    feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(_feature_names, importances)]
    sorted_feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    if dictionary['verbosity'] > 1: [print('Predictor: {:20} Importance: {}'.format(*pair)) for pair in sorted_feature_importances];
    sorted_features, sorted_importances = zip(*sorted_feature_importances)

    if dictionary['verbosity'] > 2: 
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
    if dictionary['plot_fig_title']:
        plt.title('Relative feature importance for RF at lead time ' + str(_lead_time_) + ' week', fontsize = '14')
    plt.grid(axis = 'y', color = '#D3D3D3', linestyle = 'solid')
    ## Save plot
    if 'bin' in _tgn_: 
        pred_type = 'proba_classi'
        model_name = 'RFC'
    else: 
        pred_type = 'regr'
        model_name = 'RFR'
    dir_name = dictionary['path_plots'] + 'train_full/' + pred_type + '/' + _tgn_ + '/features/' + model_name + '/'
    if dictionary['cv_type'] == 'nested': 
        save_name = dir_name + model_name + '_feature_imp_' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: 
        save_name = dir_name + model_name + '_feature_imp_' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    
    # Importances table    
    if dictionary['cv_type'] == 'none': 
        dir_name = dictionary['path_features']
        file_name = model_name + '_feature_imp_' + _tgn_ + '.csv'     
        df = pd.DataFrame(data = {'lead_time_' + str(_lead_time_): list(np.round(importances, 2)), 
                                  'feature_names': _feature_names})
        df = df.set_index('feature_names')
        ordered_feature_names = [x + '_' + dictionary['timestep'] + '_lag' + str(lag) for x in _pred_names for lag in range(_lead_time_, _lead_time_ + dictionary['num_lags'])]
        df = df.reindex(ordered_feature_names, axis = 0)
        if _lead_time_ > dictionary['lead_times'][0]:
            df_old = pd.read_csv(dir_name + file_name, index_col = 0)
            df_new = df_old.join(df, how = 'outer')
        else: df_new = df
        ordered_feature_names = [x + '_' + dictionary['timestep'] + '_lag' + str(lag) for x in _pred_names for lag in range(dictionary['lead_times'][0], _lead_time_ + dictionary['num_lags'])]
        df_new = df_new.reindex(ordered_feature_names, axis = 0)
        save_to_file(df_new, dir_name, file_name, 'csv')
        

            
            
def plot_data_histogram(t2m_anom):
    
    """
      inputs
      -------
        t2m_anom                             xr.DataArray : 2m-temperature anomalies in Central European box (time series)


      outputs
      -------
        None. Plots histogram of data dsitribution (full time period), that illustrates the different heatwave definitions. The blue events correspond to 
        the histogram of standarized (μ = 0, σ = 1) temperature anomalies. The vertical blue line marks the mean (μ = 0) of the distribution. The stippled 
        orange (red) line marks +1 (+1.5) standard deviations (σ) from the mean. The +1σ (+1.5σ) index is defined as 1 for the events found on the right 
        of the orange (red) line and as 0 for the events located to the left of it.

    """ 
    
    # Time selection
    ## Select season
    t2m_anom = t2m_anom.loc[t2m_anom['time.month'] <= dictionary['final_month']]
    t2m_anom = t2m_anom.loc[dictionary['initial_month'] <= t2m_anom['time.month']]
    ## Select train_full set only
    train_full_slice = slice(str(int(dictionary['splits'].sel(slice_type = 'train_full', edge = 'start'))), 
                             str(int(dictionary['splits'].sel(slice_type = 'train_full', edge = 'end'))))
    t2m_anom = t2m_anom.sel(time = train_full_slice)
    
    # Standarize data again
    t2m_anom = (t2m_anom.groupby('time.dayofyear'))/(t2m_anom.groupby('time.dayofyear').std('time'))
    
    # Plot
    fig = plt.figure(figsize = (9, 5))
    ### The histogram of the data
    n, bins, patches = plt.hist(t2m_anom.values, 200, facecolor = 'cornflowerblue', alpha = 0.75, density = False)
    plt.xlabel('Standardized temperature anomalies ($\sigma$)', fontsize = 16)
    plt.ylabel('Number of events', fontsize = 16)
    ### Best fit of data
    (mu, sigma) = norm.fit(t2m_anom)
    plt.axvline(np.round(mu, decimals = 0), color = 'cornflowerblue', label = 'Std T$_{anom}$ $\mu$=%.0f' %(abs(mu)))
    ## Title
    if dictionary['plot_fig_title']: 
        plt.title(r'$\mathrm{Histogram\ of\ the\ standarized\ temperature\ anomalies:}\ \mu=%.0f,\ \sigma=%.0f$' %(abs(mu), sigma))
    ## Add grid
    plt.grid(True)
    ### Add 1SD and 1.5SD lines
    plt.axvline(1, color = 'orange', label = 'Std T$_{anom}$ at +1$\sigma$', linestyle = '--')
    plt.axvline(1.5, color = 'darkred', label = 'Std T$_{anom}$ at +1.5$\sigma$', linestyle = '--')
    ## Formatting
    plt.legend(fontsize = 16, loc = 'upper left')
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    ### Save plot
    dir_name = dictionary['path_plots'] + 'data_visualization/'
    save_name = dir_name + 'temperature_anomalies_histogram_' + dictionary['timestep'] + '.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    



def plot_lagged_correlation_heatmap(target_pos, y, p_matrix, long_predictor_names, N, tau_max):
    
    """
      inputs
      -------
        target_pos                             int : position of target among the targets
        y                                   matrix : lagged linear correlations between each of the predictors and the target with dimensions 
                                                     (tau_max x N)
        p_matrix                            matrix : p-values for each lagged linear correlation with dimensions like y
        long_predictor_names       list of strings : print long names of predictors to be shown in plot
        N                                      int : number of predictors
        tau_max                                int : number of time lags shown in plot in units of timestep (e.g., tau_max = 6 and timestep = '7D' 
                                                     shows correlations between predictor and target time series with a shift of up to 6 weeks)


      outputs
      -------
        None. Plots heatmap of linear lagged correlations between the target and each of the predictors with hatched non-significant cells at 
              alpha_level significance level.
              x-axis: lags in range [1, tau_max], y-axis: N predictors

    """ 
    
    # Plot
    ## Prepare plot
    fig = plt.figure(figsize = [tau_max, N/1.5])
    sns.set_theme(palette = 'vlag')
    ### Set axes for new colorbar
    cbaxes = fig.add_axes([1, 0.13, 0.05, 0.75])
    # Draw a heatmap with the numeric values in each cell
    tmin = -.8
    tmax = .8
    ax = fig.add_subplot(111)
    ## Define x axis labels
    x_axis_labels = np.arange(1, tau_max + 1, step = 1)
    heat = sns.heatmap(y[1:, target_pos, 1:tau_max + 1], 
                       xticklabels = x_axis_labels, 
                       annot = True, 
                       linewidths = 0.2, 
                       fmt = '.2f', 
                       ax = ax, 
                       cbar_ax = cbaxes, 
                       cmap = 'vlag', 
                       vmin = tmin, 
                       vmax = tmax, 
                       norm = MidpointNormalize(midpoint = 0, vmin = tmin, vmax = tmax))
    ## Hatch non-significant pixel i.e. p_value > alpha_level
    hatch_non_significant_cells(ax, p_matrix, target_pos, tau_max)
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
    dir_name = dictionary['path_plots'] + 'data_visualization/'
    save_name = dir_name + pred_set + 'lagged_correlations_' + dictionary['timestep'] + '.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')        
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    
    
# Add lat-lon box to axis
def add_sub_region_box(ax_, sub_region_, color_, lw, ls = None):
    
    """
      inputs
      -------
        ax_                         matplotlib axis : original axis
        sub_region_                          Region : lat-lon coordinates
        color_                                  str : line color
        lw                                      int : line width in pt
        ls                                      int : line style (e.g., '-')
    


      outputs
      -------
        ax_                        matplotlib axis : updated axis containing the new subregion box

    """ 
    
    if ls == None: ls = '-'
    geom = geometry.box(minx = sub_region_.lonmin, maxx = sub_region_.lonmax, miny = sub_region_.latmin, maxy = sub_region_.latmax)
    ax_.add_geometries([geom], crs = ccrs.PlateCarree(), alpha = 1, edgecolor = color_, facecolor = 'None', linewidth = lw, linestyle = ls, label = str(sub_region_.region_name))
    return ax_
    
    
def plot_latlon_boxes():
    
    """
      inputs
      -------
        None.


      outputs
      -------
        None. Plots map with latitude-longitude boxes defined in const.py and used to define the predictors.

    """ 
    
    # Choose map projection and define origin coordinate frame
    geo = ccrs.PlateCarree(central_longitude = -90.) 
    robin = ccrs.Robinson(central_longitude = 0.) 
    # Prepare figure
    plt.figure(figsize = (12,5))
    ax = plt.subplot(1, 1, 1, projection = geo)
    ax.set_extent([-100, 40, 10, 60])

    # Define colors
    box_colors = ['orange', 'orange', 'dimgray', 'c',  'darkblue']
    lines = ['-', '-','dotted', '-', '-']

    # Add gridlines
    gl = ax.gridlines(draw_labels = True)
    gl.xlabel_style = {'size': 14, 'color': 'k','rotation':0}
    gl.ylabel_style = {'size': 14, 'color': 'k','rotation':0}

    # Add coastlines
    ax.coastlines(alpha = 0.5)

    # Draw boxes
    for box, box_color, line in zip(dictionary['boxes'], box_colors, lines): 
        if str(box.box.values) != 'ce_obs':
            Region = namedtuple('Region', field_names = ['region_name','lonmin','lonmax','latmin','latmax'])
            sub_region =  Region(
                                 region_name = box.box.values,
                                 lonmin = box[0][1],
                                 lonmax = box[1][1],
                                 latmin = box[1][0],
                                 latmax = box[0][0]
                                )
            ax = add_sub_region_box(ax, sub_region, box_color, 4, ls = line)    

    # Add legend
    box_names = list(dictionary['boxes'].box.values)
    box_names.remove('ce_obs')
    long_box_names = {box_name: dictionary['long_box_names'][box_name] for box_name in box_names}.values()
    custom_lines = [Line2D([0], [0], color = box_colors[1], lw = 4),
                    Line2D([0], [0], color = box_colors[2], lw = 4, linestyle = 'dotted'),
                    Line2D([0], [0], color = box_colors[3], lw = 4),
                    Line2D([0], [0], color = box_colors[4], lw = 4)]
    ax.legend(custom_lines, long_box_names, bbox_to_anchor = (0., 0.), loc = 'lower left', fontsize = 14)

    # Add title
    if dictionary['plot_fig_title']: plt.title('Location of boxes', fontsize = 14)

    # Save plot
    dir_name = dictionary['path_plots'] + 'data_visualization/'
    save_name = dir_name + 'boxes.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
        

