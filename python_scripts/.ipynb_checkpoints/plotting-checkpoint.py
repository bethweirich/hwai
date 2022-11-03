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
from metrics import roc_auc_score, pr_auc_score
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
        plt.axhline(0, color = 'navy', linewidth = 2, linestyle = '--')
        plt.title(var_name)
        if dictionary['verbosity'] > 0: plt.show()
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
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    print('SEA Eof')
    sea_eof_.plot.contourf(vmin = -0.12, vmax = 0.12, levels = 9, cmap = 'coolwarm')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    print('SNAO Eof daily composite')
    z500_snao_.plot.contourf(vmin = -60, vmax = 60, levels = 7, cmap = 'coolwarm')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    print('SEA Eof daily composite')
    z500_sea_.plot.contourf(vmin = -180, vmax = 180, levels = 7, cmap = 'coolwarm')
    if dictionary['verbosity'] > 0: plt.show()
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
        if dictionary['verbosity'] > 0: plt.show()
        plt.close()
        


# In[49]:


def show_hyperparameters(tgn):
    
    """
    inputs
    ------
    tgn                     str : name of target 

    outputs
    -------
    None.  Prints all hyperparameters for the no cross-validation case and saves them to file
    
    """   
    
    if 'bin' in tgn: rf = 'RFC_' + dictionary['balance_types'][0]
    else: rf = 'RFR'
        
    if dictionary['cv_type'] == 'none': 
        # Read hyperparameters and combine them into dataframe
        ## Initialize
        ds_lts = []
        ## Specify path
        dir_name = dictionary['path_hyperparam']        
        ## Add hyperparameters for each lead time
        for lt in dictionary['lead_times']:
            file_name_rf = 'best_rf_hyperparam_none_' + tgn + '_lead_time_' + str(lt) + '.nc'
            file_name_linear = 'best_linear_hyperparam_none_' + tgn + '_lead_time_' + str(lt) + '.npy'
            ds_lt_rf = xr.open_dataset(dir_name + file_name_rf).sel(forecast = rf)
            ds_lt_linear = xr.Dataset({'value': (('best_hyperparameter', 'lead_time'), np.reshape(float(np.load(dir_name + file_name_linear)), (1,1)))},
                                            coords = {'best_hyperparameter': ['best_alpha'], 'lead_time': [lt]})
            ds_lt = ds_lt_rf.combine_first(ds_lt_linear) 
            ds_lts.append(ds_lt)
    
        ds = xr.concat(ds_lts, dim = 'lead_time')
        df = ds.to_dataframe().drop(['forecast'], axis = 1).transpose()
            
        if dictionary['verbosity'] > 1:
            # Show table
            pd.set_option('display.max_rows', None, 'display.max_columns', None)
            display(df)
            pd.reset_option('display.max_rows', 'display.max_columns')
            
        # Save table to file
        save_to_file(df, dir_name + 'tables/', 'table_best_rf_hyperparam_' + tgn + '.csv', 'csv')
        


# In[ ]:


def plot_pred_time_series(y, predictions_rr, predictions_rfr, persistence, climatology, ecmwf, lead_time, tgn, outer_split_num_ = None):
    
    """
    inputs
    ------
    y                    pd.Series : time series of the true target
    predictions_rr        np.array : time series of the predicted target (same length and time as y) by the RR model
    predictions_rfr       np.array : time series of the predicted target (same length and time as y) by the RFR model
    persistence          pd.Series : time series of the target's persistence forecast (same length and time as y)
    climatology           np.array : time series of the target's climatology forecast (same length and time as y)
    ecmwf             xr.DataArray : time series of the target's ECMWF forecast (same length and time as y)
    lead_time                  int : Lead time of prediction
    tgn                        str : name of target variable 
    " outer_split_num_         int : counter for outer splits "


    outputs
    -------
    None.  Displays time series cutting only the summer months plot
    
    """

    predictions_rr_df = pd.DataFrame(data = predictions_rr, index = y.index).sort_index()  
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
        ax.plot(y.sort_index(), label = 'Ground Truth', color = 'black', linewidth = 2)
        # ML predictions
        if dictionary['reg_regr']: ax.plot(predictions_rr_df, '--', label = 'RR', color = 'crimson')
        else: ax.plot(predictions_rr_df, '--', label = 'RR', color = 'crimson')
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
        #if i == 0: ax.set_ylabel('Temperature anomalies (°C)', fontsize = 16)           
        ax.yaxis.set_tick_params(labelsize = 16)
        
    ax.legend(bbox_to_anchor = (1.1, 1.5), loc = 'upper right')
    if dictionary['plot_fig_title'] == True:
        fig.suptitle('Predicted and true ' + tgn + ' on test set at ' + str(lead_time) + ' weak(s) lead time', fontsize = 14)
    ## Save plot
    dir_name = dictionary['path_plots'] + 'test/' + tgn + '/pred_time_series/'
    if dictionary['cv_type'] == 'nested': save_name = dir_name + 'pred_time_series_' + tgn + '_lead_time_' + str(lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + 'pred_time_series_' + tgn + '_lead_time_' + str(lead_time) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()


def plot_pred_time_series_all_lead_times():
    
    """
    inputs
    ------
    y                    pd.Series : time series of the true target
    predictions_rr       np.array  : time series of the predicted target (same length and time as y) by the RR model
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

    # Extract data LT-indep data
    path = dictionary['path_time_series'] + 't2m/'
    ## GT
    gt = np.load(path + 't2m_GT_test.npy')
    index = pd.DatetimeIndex(np.load(path + 't2m_index_test_lead_time_1_weeks.npy'))
    
    # Prepare plot
    ## Define the ranges for the dates
    drange = [[datetime.date(i, dictionary['initial_month'], 1),datetime.date(i, dictionary['final_month'], 30)] for i in range(index[0].year, index[-1].year + 1)]
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
            if dictionary['reg_regr']: ax.plot(predictions_rr, '--', label = 'RR', color = 'crimson')
            else: ax.plot(predictions_rr, '--', label = 'RR', color = 'crimson')
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
    if dictionary['plot_fig_title'] == True:
        fig.suptitle('Predicted and true 2m-air temperature on test set at 1--6 weaks lead time', fontsize = 14)
    ## Save plot
    dir_name = dictionary['path_plots'] + 'test/regr/time_series/'
    save_name = dir_name + 'pred_time_series_t2m_all_lead_times.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()

# In[8]:


def plot_roc_curve(fpr, tpr, subset, tg_name, model_name_, _lead_time, best_th_ix = None, outer_split_num_ = None, inner_split_num_ = None):
    
    """
    inputs
    ------
    fpr                        ndarray : False Positive Rate for each probability threshold               
    tpr                        ndarray : True Positive Rate for each probability threshold  
    subset                         str : 'vali', 'train', 'test'
    tg_name                        str : target name
    model_name_                    str : name of the ML model 
    _lead_time                     int : lead time for prediction
    " best_th_ix                   int : index of best probability threshold "
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
    if best_th_ix: 
        if isinstance(fpr, pd.Series) or isinstance(fpr, pd.DataFrame):
            plt.scatter(fpr.iloc[best_th_ix], tpr.iloc[best_th_ix], marker='o', color='black', label='Best')
        else:
            plt.scatter(fpr[best_th_ix], tpr[best_th_ix], marker='o', color='black', label='Best')
    ## Axis labels
    plt.xlabel('FPR', fontsize = 12)
    plt.ylabel('TPR', fontsize = 12)
    plt.legend(loc = 'lower right')
    plt.title('ROC curve', fontsize = 14)
    ## Save plot
    dir_name = dictionary['path_plots'] + subset + '/' + tg_name + '/ROC/' + model_name_ + '/'
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


def plot_pr_curve(precision, recall, subset, tg_name, model_name_, _lead_time, best_th_ix = None, outer_split_num_ = None, inner_split_num_ = None):

    """
    inputs
    ------
    precision                  ndarray : False Positive Rate for each probability threshold               
    recall                     ndarray : True Positive Rate for each probability threshold  
    subset                         str : 'vali', 'train', 'test'
    tg_name                        str : target name
    model_name_                    str : name of the ML model 
    _lead_time                     int : lead time for prediction
    " best_th_ix                   int : index of best probability threshold "
    " outer_split_num_             int : counter for outer splits "
    " inner_split_num_             int : counter for inner splits "
    
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
    dir_name = dictionary['path_plots'] + subset + '/' + tg_name + '/PR/' + model_name_ + '/'
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


def plot_calibration_curve(y_true, y_prob, subset, tg_name, model_name_, _lead_time, outer_split_num_ = None, inner_split_num_ = None):

    """
    inputs
    ------
    y_true                   pd.Series : False Positive Rate for each probability threshold               
    y_prob                     ndarray : True Positive Rate for each probability threshold  
    subset                         str : 'vali', 'train', 'test'
    tg_name                        str : target name
    model_name_                    str : name of the ML model 
    _lead_time                     int : lead time for prediction
    " outer_split_num_             int : counter for outer splits "
    " inner_split_num_             int : counter for inner splits "
    
    outputs
    -------
    None. Plots the calibration curve
    
    """ 
    
    # Compute curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins = 10,)
    #                         strategy = 'quantile')

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
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    


# In[ ]:


def plot_proba_histogram(pred_proba_, tgn_, _model_name_, _lead_time_, best_threshold = None, outer_split_num_ = None, inner_split_num_ = None):

    """
    inputs
    ------
    pred_proba_              pd.Series : probabilities of belonging to the class 1 (being a HW)
    tgn_                           str : target name
    _model_name_                   str : name of the ML model 
    _lead_time_                    int : lead time for prediction
    " best_threshold             float : best probability threshold "  
    " outer_split_num_             int : counter for outer splits "
    " inner_split_num_             int : counter for inner splits "
    
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
    if dictionary['plot_fig_title'] == True:
        plt.title('Histogram of heat wave probabilities')
    plt.grid(True)
    ## Save plot
    dir_name = dictionary['path_plots'] + 'vali/' + tgn_ + '/hist/' + _model_name_ + '/'
    if dictionary['cv_type'] == 'nested': 
        save_name = dir_name + 'hist_' + _model_name_ + '_' + tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks_outer_split_' + str(outer_split_num_) + '_inner_split_' + str(inner_split_num_) + '.pdf'
    else: save_name = dir_name + 'hist_' + _model_name_ + '_' + tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name)
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    


# In[ ]:


def classification_plots(y, pred_proba, subset_, tgn_, model_name, lt_, best_th_ = None, _outer_split_num_ = None):
    
    """
    inputs
    ------
    y                        pd.Series : ground truth for heatwave index (binary)
    pred_proba               pd.Series : probabilities of belonging to the class 1 (being a HW)
    subset_                        str : time period (e.g., 'test')
    tgn_                           str : target name
    model_name                     str : name of the ML model ('RC', 'RFC')
    lt_                            int : lead time for prediction
    " best_th_                   float : best probability threshold "  
    " _outer_split_num_            int : counter for outer splits "
    
    outputs
    -------
    None. Plots several classification plots.
    
    """  
    
    ### Calibration curve
    plot_calibration_curve(y, pred_proba, subset_, tgn_, model_name, lt_, outer_split_num_ = _outer_split_num_)        
    ### Probability histogram
    plot_proba_histogram(pred_proba, tgn_, model_name, lt_, best_threshold = best_th_, outer_split_num_ = _outer_split_num_)  
    ### ROC 
    #### ROC AUC score
    if dictionary['verbosity'] > 1: print('ROC AUC score = %.3f' % (roc_auc_score(y, pred_proba)))
    ### ROC curve 
    fpr, tpr, thresholds = roc_curve(y, pred_proba)
    if best_th_: best_th_ix_ = (np.abs(thresholds - best_th_)).argmin()        
    else: best_th_ix_ = None
    plot_roc_curve(fpr, tpr, subset_, tgn_, model_name, lt_, best_th_ix = best_th_ix_, outer_split_num_ = _outer_split_num_)          
    ### PR 
    #### PR AUC score
    if dictionary['verbosity'] > 1: print('PR AUC score = %.3f' % (pr_auc_score(y, pred_proba)))
    #### PR curve 
    precision, recall, thresholds = precision_recall_curve(y, pred_proba)
    if best_th_: best_th_ix_ = (np.abs(thresholds - best_th_)).argmin()        
    else: best_th_ix_ = None
    plot_pr_curve(precision, recall, subset_, tgn_, model_name, lt_, best_th_ix = best_th_ix_, outer_split_num_ = _outer_split_num_) 
    


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
    #plt.hist(index, bins = 100, facecolor = 'orange', alpha = 0.75)
    index.plot(linewidth = 0.3, color = 'black')
    plt.xlabel('# Events', fontsize = 12)
    plt.ylabel('Time', fontsize = 12)
    if dictionary['plot_fig_title'] == True:
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
    


# In[5]:


def plot_tp(metrics, ax, metrics_std = None):
    
    """
    inputs
    ------
    metrics                  xr.Dataset : dataset of metrics data with dimensions (forecast x metric)
    ax                  matplotlib axis : location to plcae the TP plot next to the other classification metrics
    "metrics_std             xr.Dataset : dataset of the standard deviation of the ensemble of metrics data with dimensions (forecast x metric) (only for nested CV)"


    outputs
    -------
    None.  Plots a bar chart with TPR and FPR to evaluate (and compare) the performance 
    of the models. x-axis: lead time, y-axis: TPR and FPR
    
        TPR: True Positive Rate
        FPR: False Positive Rate (events that were predicted but not detected) = FP/(TN+FN)
    
    """
        
    # Prepare plot    
    ## Bar width 
    if len(dictionary['balance_types']) > 1: bar_width = 0.1
    else: bar_width = 0.12
    ## x axis
    xlabels = list(metrics.lead_time.values)
    ### The label locations
    x = np.arange(len(xlabels)) 
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation = 0)
    
    # Define models, colors...
    reference_models = ['Ground Truth', 'Climatology', 'Persistence']    
    if sorted(dictionary['balance_types']) == sorted(['oversampling', 'undersampling']): ml_models = list(itertools.product(['RC', 'RFC'], zip(['oversampling', 'undersampling'], ['+', '-'])))
    else: ml_models = ['RC', 'RFC']
    forecasts_std = ['RC_std', 'RFC_std']
    if sorted(dictionary['balance_types']) == sorted(['oversampling', 'undersampling']): colors = ['dimgray', 'navy', 'lightgray', 'darkred', 'indianred', 'teal', 'turquoise']
    else: colors = ['dimgray', 'navy', 'lightgray', 'darkred', 'teal']    
    
    if 'ECMWF' in metrics.forecast:
        colors.insert(len(reference_models), 'orange')
        reference_models.append('ECMWF')
        forecasts_std.append('ECMWF_std')
    if sorted(dictionary['balance_types']) == sorted(['oversampling', 'undersampling']):
        reference_models = list(itertools.product(reference_models, zip(['undersampling'], [''])))
        bar_shifts = np.linspace(-len(reference_models), len(ml_models), len(reference_models) + len(ml_models) + 1, dtype = int) 
    else: bar_shifts = np.linspace(1 - len(reference_models), len(ml_models), len(reference_models) + len(ml_models), dtype = int) 
       
    ## Plot
    all_model_characteristics = reference_models + ml_models   
    if sorted(dictionary['balance_types']) == sorted(['oversampling', 'undersampling']):
        for model_characteristics, color, bar_shift in zip(all_model_characteristics, colors, bar_shifts):
            # Bar plot
            ax.bar(x + bar_shift * bar_width, 
                   metrics.sel(forecast = model_characteristics[0]).sel(balance = model_characteristics[1][0]).sel(metric = 'TPR').to_array().values.flatten(),
                   label = model_characteristics[0] + '$^{' + model_characteristics[1][1] + '}$',
                   color = color, 
                   width = bar_width)
        for model_characteristics, color, bar_shift, i in zip(all_model_characteristics, colors, bar_shifts, range(len(colors))):
            # Hatch
            ax.bar(x + bar_shift * bar_width, 
                   metrics.sel(forecast = model_characteristics[0]).sel(balance = model_characteristics[1][0]).sel(metric = 'FPR').to_array().values.flatten(),
                   label = 'FPR' if i == 0 else '', 
                   edgecolor = 'black', 
                   hatch = '//', 
                   width = bar_width, 
                   facecolor = 'none') 
            # Add error bars representing uncertainty
            if dictionary['cv_type'] == 'nested':             
                if metrics_std.sel(forecast = model_characteristics[0]).sel(balance = model_characteristics[1][0]).sel(metric = 'TPR').to_array().values.flatten().any() > 0.0001:
                    ax.errorbar(x + bar_shift * bar_width, 
                                metrics.sel(forecast = model_characteristics[0]).sel(balance = model_characteristics[1][0]).sel(metric = 'TPR').to_array().values.flatten(),
                                yerr = metrics_std.sel(forecast = model_characteristics[0]).sel(balance = model_characteristics[1][0]).sel(metric = 'TPR').to_array().values.flatten(),
                                fmt = 'o', color = 'black', capsize = 4, linewidth = 1, markersize = 2)   
            elif dictionary['cv_type'] == 'none': 
                for forecasts_std_substr in forecasts_std:
                    if model_characteristics[0] in forecasts_std_substr:
                        ax.errorbar(x + bar_shift * bar_width, 
                                metrics.sel(forecast = model_characteristics[0]).sel(balance = model_characteristics[1][0]).sel(metric = 'TPR').to_array().values.flatten(),
                                yerr = metrics.sel(forecast = model_characteristics[0] + '_std').sel(balance = model_characteristics[1][0]).sel(metric = 'TPR').to_array().values.flatten(),
                                fmt = 'o', color = 'black', capsize = 4, linewidth = 1, markersize = 2)
                 
    else: 
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


# In[6]:


def plot_metrics(metrics_full, prediction_type):
    
    """
    inputs
    ------
    metrics_full            dict of xr.Dataset : dataset of metrics data with dimensions (forecast x metric x lead_time "x statistics")
    prediction_type                        str : regr, classi or proba_classi


    outputs
    -------
    None.  Plots n graphs (one for each metric) to evaluate (and compare) the performance 
    of the models. x-axis: lead time, y-axis: the metric
    
                regr:    RMSE: Root Mean Squared Error
                         Corr: Correlation
        
        proba_classi: ROC AUC: Receiver Operating Characteristic Area Under Curve
                      BS: Brier score 
        
              classi:     EDI: Extremal Dependence Index
                          TPR-FPR bar plot
        
    
    """
                
    # Select prediction type
    metrics_full = metrics_full[prediction_type]
    
    # Separate mean and std
    if dictionary['cv_type'] == 'nested': 
        metrics = metrics_full.sel(statistics = 'mean')
        metrics_std = metrics_full.sel(statistics = 'std')
    else: metrics = metrics_full
    
    # Prepare metrics
    if prediction_type != 'regr' and len(dictionary['balance_types']) == 1:  
        metrics = metrics.sel(balance = dictionary['balance_types'][0])
        if dictionary['cv_type'] == 'nested': metrics_std = metrics_std.sel(balance = dictionary['balance_types'][0])
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
        if sorted(dictionary['balance_types']) == sorted(['oversampling', 'undersampling']):
            ml_models = list(itertools.product(['RC', 'RFC'], zip(dictionary['balance_types'],['+', '-'], ['-o', '--o'])))               
            colors = ['dimgray', 'navy', 'lightgray', 'darkred', 'indianred', 'teal', 'turquoise']
        else:              
            colors = ['dimgray', 'navy', 'lightgray', 'darkred', 'teal']
    # Add ECMWF
    if 'ECMWF' in metrics.forecast:
            colors.insert(len(reference_models), 'orange')
            reference_models.append('ECMWF')
            forecasts_std.append('ECMWF_std')
    
    ## Prepare plot     
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
            elif prediction_type != 'regr' and len(dictionary['balance_types']) == 1: forecasts = reference_models + ['RC', 'RFC']       
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
                    if len(dictionary['balance_types']) == 1:             
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
                    elif sorted(dictionary['balance_types']) == sorted(['oversampling', 'undersampling']):  
                        reference_models = list(itertools.product(reference_models, zip(['undersampling'],[''], ['-o'])))
                        all_model_characteristics = reference_models + ml_models
                        for model_characteristics, color in zip(all_model_characteristics, colors): 
                            if dictionary['cv_type'] == 'nested': 
                                plt.errorbar(metrics.lead_time, 
                                             metrics.sel(metric = metric, forecast = model_characteristics[0], balance = model_characteristics[1][0], target_name = tgn).value, 
                                             yerr = metrics_std.sel(metric = metric, forecast = model_characteristics[0], balance = dictionary['balance_types'][0], target_name = tgn).value, 
                                             fmt = model_characteristics[1][2], 
                                             label = model_characteristics[0] + '$^{' + model_characteristics[1][1] + '}$', 
                                             color = color, 
                                             capsize = 4, linewidth = 2, markersize = 4)
                            elif dictionary['cv_type'] == 'none':
                                plt.plot(metrics.lead_time, 
                                             metrics.sel(metric = metric, forecast = model_characteristics[0], balance = model_characteristics[1][0], target_name = tgn).value, 
                                             model_characteristics[1][2], 
                                             label = model_characteristics[0] + '$^{' + model_characteristics[1][1] + '}$', 
                                             color = color, 
                                             linewidth = 2, markersize = 4)
                                if model_characteristics[0] + '_std' in forecasts_std:
                                    plt.errorbar(metrics.lead_time, 
                                                 metrics.sel(metric = metric, forecast = model_characteristics[0], balance = model_characteristics[1][0], target_name = tgn).value, 
                                                 yerr = metrics.sel(metric = metric, forecast = model_characteristics[0] + '_std', balance = model_characteristics[1][0], target_name = tgn).value,                            
                                                 fmt = model_characteristics[1][2], 
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
    if dictionary['cv_type'] == 'nested': save_name = 'metrics_lt_plot_' + prediction_type + '_nested_CV'
    else: save_name = 'metrics_lt_plot_' + prediction_type
    if prediction_type != 'regr' and dictionary['balance_types'] == ['oversampling']: save_name = save_name + '_only_oversampled.pdf'
    elif prediction_type != 'regr' and dictionary['balance_types'] == ['none']: save_name = save_name + '_non_bal.pdf'
    else: save_name = save_name + '.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(dir_name + save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()


# In[ ]:


def show_coeffs(_model_, model_name, _pred_names, _feature_names, _tgn_, _lead_time_, outer_split_num_ = None):
    
    """
    inputs
    ------
    _model_                     fct : trained model
    model_name                  str : 'RR' or 'RC'
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
    dir_name = dictionary['path_plots'] + 'train/' + _tgn_ + '/features/' + model_name + '/'
    if dictionary['cv_type'] == 'nested': save_name = dir_name + model_name.lower() + '_coeffs_' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + model_name.lower() + '_coeffs_' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    
    # Regression coefficients table        
    if dictionary['cv_type'] == 'none': 
        file_name = 'data_' + model_name.lower() + '_coeffs_' + _tgn_ + '.csv'        
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
        if _lead_time_ == dictionary['lead_times'][-1] and dictionary['verbosity'] > 1:            
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
    dir_name = dictionary['path_plots'] + 'train/' + _tgn_ + '/features/RF/'
    if dictionary['cv_type'] == 'nested': save_name = dir_name + 'rf_imp_' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks_outer_split_' + str(outer_split_num_) + '.pdf'
    else: save_name = dir_name + 'rf_imp_' + _tgn_ + '_lead_time_' + str(_lead_time_) + '_weeks.pdf'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    
    # Importances table        
    if dictionary['cv_type'] == 'none': 
        file_name = 'data_rf_imp_' + _tgn_ + '.csv'     
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
        if _lead_time_ == dictionary['lead_times'][-1] and dictionary['verbosity'] > 1: 
            # Show table
            pd.set_option('display.max_rows', None, 'display.max_columns', None)
            display(df)
            pd.reset_option('display.max_rows', 'display.max_columns')


# In[1]:


def plot_data_histogram(t2m_anom):
    
    """
      inputs
      -------
        t2m_anom                             xr.DataArray : 2m temperature anomalies in Central European box (time series)


      outputs
      -------
        None. Plots histogram of data dsitribution of train_full set, that illustrates the different heatwave definitions. The blue events correspond to the histogram of 
        standarized (μ = 0, σ = 1) temperature anomalies. The vertical blue line marks the mean (μ = 0) of the distribution. The stippled orange (red) 
        line marks +1 (+1.5) standard deviations (σ) from the mean. The +1σ (+1.5σ) index is defined as 1 for the events found on the right of the orange 
        (red) line and as 0 for the events located to the left of it.

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

    if dictionary['plot_fig_title'] == True: 
        plt.title(r'$\mathrm{Histogram\ of\ the\ standarized\ temperature\ anomalies:}\ \mu=%.0f,\ \sigma=%.0f$' %(abs(mu), sigma))
    plt.grid(True)
    ### Add 1SD and 1.5SD lines
    plt.axvline(1, color = 'orange', label = 'Std T$_{anom}$ at +1$\sigma$', linestyle = '--')
    plt.axvline(1.5, color = 'darkred', label = 'Std T$_{anom}$ at +1.5$\sigma$', linestyle = '--')

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
        

