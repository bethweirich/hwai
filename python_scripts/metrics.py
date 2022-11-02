#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Import packages
import xarray as xr
import pandas as pd
import matplotlib                             
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import os


## Metrics
from sklearn.metrics import (confusion_matrix,
                             roc_auc_score,
                             f1_score,
                             recall_score,
                             precision_score,
                             mean_squared_error,
                             precision_recall_curve,
                             roc_curve,
                             auc)
from scipy import stats
from scipy.stats import pearsonr


# In[20]:


# Import own functions
## Utils
from ipynb.fs.defs.utils import filter_dates_like, save_to_file


# In[21]:


# Import dictionary
from ipynb.fs.full.const import dictionary


# In[22]:


def pr_auc_score(_y, _y_pred):
    
    np.seterr(divide = 'ignore', invalid = 'ignore')
    if dictionary['verbosity'] > 2: print('Computing PR AUC score')
    _precision, _recall, _thresholds = precision_recall_curve(y_true = _y, probas_pred = _y_pred)
    # Use AUC function to calculate the area under the curve of precision recall curve
    _auc_precision_recall = auc(_recall, _precision)
    if np.isnan(_auc_precision_recall):
        _auc_precision_recall = 0. 
    np.seterr(divide = 'warn', invalid = 'warn')
    
    return _auc_precision_recall 


# In[23]:


def gmean_score(_y, _y_pred):
    
    # For all thresholds
    # calculate roc curves
    _fpr, _tpr, _thresholds = roc_curve(_y, _y_pred)
    # calculate the g-mean for each threshold
    _gmean = np.sqrt(_tpr * (1 - _fpr))
    
    return _gmean


# In[24]:


def gmean_score_bin(_y, _y_pred):
    
    # For a given threshold
    #Confusion matrix
    tp = confusion_matrix(_y, _y_pred)[1,1]
    fn = confusion_matrix(_y, _y_pred)[1,0]
    fp = confusion_matrix(_y, _y_pred)[0,1]
    tn = confusion_matrix(_y, _y_pred)[0,0]
    
    #TPR
    tpr = tp / (tp + fn)
    
    #FPR
    fpr = fp / (tn + fp)
    
    # G-Mean
    gmean = np.sqrt(tpr * (1 - fpr))
    
    return gmean


# In[25]:


def compute_score(score_name, a, b):
    
    """
    inputs
    ------
    score_name                  str : name of score to be computed
    a                      np.array : true values 
    b                      np.array : predicted values (values, probabilities or binary)

    outputs
    -------
    score                     float : score of a,b
    
    """
    
    if dictionary['verbosity'] > 4: 
        print('score_name: ', score_name)
        print('a: ', a)
        print('b: ', b)
    if score_name == 'ROC AUC': score = roc_auc_score(a, b) # b is proba  
    elif score_name == 'PPV': score = precision_score(a, b, zero_division = 0.) # b is binary
    elif score_name == 'TPR': score = recall_score(a, b) # b is binary
    elif score_name == 'F1': score = f1_score(a, b) # b is binary
    elif score_name == 'PR AUC': score = pr_auc_score(a, b) # b is proba  
    elif score_name == 'G-Mean': score = gmean_score(a,b) # b is binary
            
    ############################################################################################################################        
    elif score_name == 'Corr': score = stats.pearsonr(a, b)[0] # b is value
    elif score_name == 'RMSE': score = mean_squared_error(a, b) # b is value
    elif score_name == 'RMSE_Corr': score = stats.pearsonr(a, b)[0] - mean_squared_error(a, b)/5 # b is value
        
    return score


# In[26]:


def tpr_from_dset(metrics_data, forecast_name, balance):

    """
    inputs
    ------
    metrics_data       xr.Dataset : dataset of metrics data with dimensions (forecast x metric x lead_time (x balance_type))                     
    forecast_name             str : name of forecast
    tgn_name                  str : name of target

    outputs
    -------
    tpr                       int : True Positive Rate (TPR), Recall or Hit Rate (HR) = TP/(TP+FN)
                                    It’s the probability that an actual positive will test positive.
    
    """
    
    tp = metrics_data.sel(metric = 'TP', forecast = forecast_name, balance_type = balance).value
    fn = metrics_data.sel(metric = 'FN', forecast = forecast_name, balance_type = balance).value
    tpr = tp / (tp + fn)
    
    return tpr


# In[27]:


def fpr_from_dset(metrics_data, forecast_name, balance):

    """
    inputs
    ------
    metrics_data       xr.Dataset : dataset of metrics data with dimensions (forecast x metric x lead_time (x balance_type))                     
    forecast_name             str : name of forecast
    tgn_name                  str : name of target

    outputs
    -------
    fpr                       int : False Positive Rate (FPR) or False Alarm Rate (FPR) = FP/(TN+FP)
                                    It’s the probability that a false alarm will be raised i.e. that a positive result will be 
                                    given when the true value is negative.
    
    """
    
    fp = (metrics_data.sel(metric = 'FP', forecast = forecast_name, balance_type = balance)).value
    tn = (metrics_data.sel(metric = 'TN', forecast = forecast_name, balance_type = balance)).value
    fpr = fp / (tn + fp)
    # Correct for trivial forecast (only positives predicted)
    fpr = fpr.fillna(1)
    
    return fpr


# In[28]:


def tpr(confusion_mat):

    """
    inputs
    ------
    confusion_mat        np.array : 2 x 2 dimensional matrix with entries [[TN, FP], [FN, TP]] 

    outputs
    -------
    tpr                       int : True Positive Rate (TPR), Recall or Hit Rate (HR) = TP/(TP+FN)
                                    It’s the probability that an actual positive will test positive.
    
    """
    
    tp = confusion_mat[1,1]
    fn = confusion_mat[1,0]
    tpr = tp / (tp + fn)
    
    return tpr


# In[29]:


def fpr(confusion_mat):

    """
    inputs
    ------
    confusion_mat        np.array : 2 x 2 dimensional matrix with entries [[TN, FP], [FN, TP]]                

    outputs
    -------
    fpr                       int : False Positive Rate (FPR) or False Alarm Rate (FPR) = FP/(TN+FP)
                                    It’s the probability that a false alarm will be raised i.e. that a positive result will be 
                                    given when the true value is negative.
    
    """
     
    fp = confusion_mat[0,1]
    tn = confusion_mat[0,0]
    fpr = fp / (tn + fp)
    
    return fpr


# In[30]:


def choose_threshold_for_best_score(pred_proba, score, score_name, y):
    
    """
    inputs
    ------
    pred_proba               pd.Series : probabilities for being a heatwave (classification of the test set)
    score                    function  : from sklearn.metrics to compute the score 
    score_name                     str : name of choosen score
    y                        pd.Series : time series of true target
    
    outputs
    -------
    best_threshold_              float : threshold for cutting in classification probabilities to optimize the given score
    
    """

    ## Choose threshold to optimize a given score
    ### Initialize score_max with its minumum
    score_max = 0.
    for threshold in np.round(np.arange(0., 1., 0.001), decimals = 3):
        pred = np.zeros(len(pred_proba))
        for i in np.where(pred_proba >= threshold):
            pred[i] = 1   
        if score_name == 'PPV': score_new = score(y, pred, zero_division = 0.)
        else: score_new = score(y, pred)
        if score_new >= score_max: 
            score_max = score_new
            best_threshold_ = threshold
            
    if dictionary['verbosity'] > 1: print('Best threshold: ', best_threshold_, ' with ', score_name, ' score: ', score_max)
   
    return best_threshold_, score_max


# In[1]:


def build_metrics_regr(y, predictions_mlr, predictions_rfr, predictions_rfr_ensemble, 
                       persistence, climatology, target_name, lead_time, subset, 
                       predictions_mlr_ensemble = None, ecmwf = None, outer_split_num_ = None): 

    """
    inputs
    ------
    y                                pd.Series : time series of the true target
    predictions_mlr                   np.array : time series of the predicted target (same
                                                 length and time as y) by the MLR model
    predictions_rfr                   np.array : time series of the predicted target (same
                                                 length and time as y) by the RFR model
    predictions_rfr_ensemble              dict : set of time series of the predicted target (same
                                                 length and time as y) by each estimator (tree) of the RFR model
    persistence                       np.array : time series of the target's persistence (same
                                                 length and time as y)
    climatology                       np.array : time series of the target's climatology (same
                                                 length and time as y)
    target_name                            str : name of target variable 
    lead_time                              int : number of weeks for prediction lead time
    subset                                 str : 'train' or 'test'
    
    " predictions_mlr_ensemble            dict : set of time series of the predicted target (same
                                                 length and time as y) by the MLR models trained on each leave-one-out subset "    
    " ecmwf                          np.array  : time series of the predicted target with ECMWF. This argument is optional. "
    " outer_split_num_                     int : counter for outer splits. This argument is optional. "

    outputs
    -------
    metrics_dset         xr.Dataset : 2 regression metrics for every forecast
    
    Displays table with 2 different metrics (Corr, RMSE) to evaluate (and compare) the performance 
    of the regression model
        Corr: Correlation
        RMSE: Root Mean Squared Error
    
    """
    
    # Metrics table
    fig, ax = plt.subplots()
    ## Hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ## Define column names
    metrics_names = ['RMSE', 'Correlation']
    ## Define row names
    forecasts_names = ['Ground Truth', 'Persistence', 'Climatology', 'MLR', 'RFR']
    ## Compute metrics
    metrics_text = [[mean_squared_error(y, y), stats.pearsonr(y, y)[0]],
                    [mean_squared_error(y, persistence), stats.pearsonr(y, persistence)[0]],
                    [mean_squared_error(y, climatology), 0.],
                    [mean_squared_error(y, predictions_mlr), stats.pearsonr(y, predictions_mlr)[0]],
                    [mean_squared_error(y, predictions_rfr), stats.pearsonr(y, predictions_rfr)[0]]]

    
    ### Correct entry for climatology corr manually
    if (target_name != 't2m'): metrics_text[2][1] = stats.pearsonr(y, climatology)[0]
    
    if dictionary['cv_type'] == 'none':
        ### Add std for ML models
        forecasts_names = forecasts_names + ['MLR_std', 'RFR_std']    
        for predictions_ensemble in [predictions_mlr_ensemble, predictions_rfr_ensemble]:
            metrics_data = []
            for i in np.arange(len(predictions_ensemble)):
                pred = pd.Series(predictions_ensemble[i], copy = False, index = y.index)
                metrics_metric = []
                for metric in metrics_names: 
                    if metric == 'RMSE':
                        metrics_metric.append(mean_squared_error(filter_dates_like(y, pred), pred))
                    if metric == 'Correlation':
                        metrics_metric.append(stats.pearsonr(filter_dates_like(y, pred), pred.astype(float))[0])
                metrics_data.append(metrics_metric)
            metrics = pd.DataFrame(metrics_data, columns = metrics_names, index = np.arange(len(predictions_ensemble)))
            #### Standard deviation
            metrics_std = metrics.std()
            #### Add to metrics
            metrics_text = metrics_text + [[metrics_std.RMSE, metrics_std.Correlation]]

        if ecmwf is not None: 
            ### Add ECMWF forecast
            forecasts_names = forecasts_names + ['ECMWF', 'ECMWF_std']
            ecmwf_metrics_data = []
            for var in list(ecmwf):
                ecmwf_metrics_metric = []
                for metric in metrics_names: 
                    if metric == 'RMSE':
                        ecmwf_metrics_metric.append(mean_squared_error(filter_dates_like(y, ecmwf[var]), ecmwf[var]))
                    if metric == 'Correlation':
                        ecmwf_metrics_metric.append(stats.pearsonr(filter_dates_like(y, ecmwf[var]), ecmwf[var].astype(float))[0])
                ecmwf_metrics_data.append(ecmwf_metrics_metric)
            ecmwf_metrics = pd.DataFrame(ecmwf_metrics_data, columns = metrics_names, index = list(ecmwf))
            #### Mean
            ecmwf_metrics_mean = ecmwf_metrics.loc[target_name + '_mean']
            # Standard deviation
            ecmwf_metrics_std = ecmwf_metrics.drop(target_name + '_mean').std()
            #### Add to metrics
            metrics_text = metrics_text + [[ecmwf_metrics_mean.RMSE, ecmwf_metrics_mean.Correlation],
                                           [ecmwf_metrics_std.RMSE, ecmwf_metrics_std.Correlation]]

    ### Round metrics
    metrics_text = np.round(metrics_text, decimals = 3) 
    ### Dataset
    metrics_dset = xr.Dataset({'value': (('forecast', 'metric'), metrics_text)}, coords = {'forecast': forecasts_names, 'metric': metrics_names})
    
    # Plot table
    metrics_table = ax.table(cellText = metrics_text,
                      rowLabels = forecasts_names,
                      colLabels = metrics_names,
                      loc = 'center')
    metrics_table.set_fontsize(12)
    metrics_table.scale(0.7, 2) 
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + subset + '/' + target_name + '/metrics_table/'    
    if outer_split_num_ != None:
        save_name = dir_name + 'metrics_table_' + subset + '_' + target_name + '_lead_time_' + str(lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '.png'
    else: save_name = dir_name + 'metrics_table_' + subset + '_' + target_name + '_lead_time_' + str(lead_time) + '_weeks.png'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches='tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    
    ## Return metrics as dataset
    return metrics_dset


# In[32]:


def build_metrics_classi(y, predictions_rc, predictions_rfc, predictions_rfc_ensemble, 
                         persistence, climatology, target_name, lead_time, subset, balance_type, 
                         predictions_rc_ensemble = None, ecmwf = None, outer_split_num_ = None): 

    """
    inputs
    ------
    y                                    pd.Series : time series of the true target
    predictions_rc                        np.array : time series of the predicted target (same
                                                     length and time as y) with the RC model
    predictions_rfc                       np.array : time series of the predicted target (same
                                                     length and time as y) with the RFC model
    predictions_rfc_ensemble                  dict : set of time series of the predicted target (same
                                                     length and time as y) by each estimator (tree) of the RFC model
    persistence                           np.array : time series of the target's persistence (same
                                                     length and time as y)
    climatology                           np.array : time series of the target's climatology (same
                                                     length and time as y)
    target_name                                str : name of target variable 
    lead_time                                  int : number of weeks for prediction lead time
    subset                                     str : 'train' or 'test'
    balance_type                               str : 'undersampling' or 'oversampling'
    
    " predictions_rc_ensemble                 dict : set of time series of the predicted target (same
                                                     length and time as y) by the RC models trained on each leave-one-out subset "    
    " ecmwf                              np.array  : time series of the predicted target with ECMWF. This argument is optional. "
    " outer_split_num_                         int : counter for outer splits. This argument is optional. "
    

    outputs
    -------
    metrics_dset          xr.Dataset : 5 classification metrics for every forecast
    
    Displays table with 5 different metrics (G-Mean, Conf:(TN, FN, TP, FP)) 
    to evaluate (and compare) the performance of the classification model
        G-Mean : geometric mean of TPR and FPR
        TPR : True Positive Rate
        FPR : False Positive Rate
        Conf: Confusion Matrix [[TN, FP],
                                [FN, TP]]
    
    """
    
    # Add a table at the bottom of the axes
    fig, ax = plt.subplots()
    ## Hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ## Define column names
    metrics_names = ['G-Mean', 'TPR', 'FPR', 'TN', 'FN', 'TP', 'FP']
    len_metrics = len(metrics_names) - 4
    ## Define row names
    forecasts_names = ['Ground Truth', 'Persistence', 'Climatology', 'RC', 'RFC'] 
    
    ## Compute metrics
    if dictionary['verbosity'] > 2: print('Compute metrics')
    float_metrics = [[gmean_score_bin(y, y), tpr(confusion_matrix(y, y)), fpr(confusion_matrix(y, y))],
                    [gmean_score_bin(y, persistence), tpr(confusion_matrix(y, persistence)), fpr(confusion_matrix(y, persistence))],
                    [gmean_score_bin(y, climatology), tpr(confusion_matrix(y, climatology)), fpr(confusion_matrix(y, climatology))],
                    [gmean_score_bin(y, predictions_rc), tpr(confusion_matrix(y, predictions_rc)), fpr(confusion_matrix(y, predictions_rc))],
                    [gmean_score_bin(y, predictions_rfc), tpr(confusion_matrix(y, predictions_rfc)), fpr(confusion_matrix(y, predictions_rfc))]]
        
    int_metrics = [[confusion_matrix(y, y)[0,0], confusion_matrix(y, y)[1,0], confusion_matrix(y, y)[1,1], confusion_matrix(y, y)[0,1]],
                  [confusion_matrix(y, persistence)[0,0], confusion_matrix(y, persistence)[1,0], confusion_matrix(y, persistence)[1,1], confusion_matrix(y, persistence)[0,1]],
                  [confusion_matrix(y, climatology)[0,0], confusion_matrix(y, climatology)[1,0], confusion_matrix(y, climatology)[1,1], confusion_matrix(y, climatology)[0,1]],
                  [confusion_matrix(y, predictions_rc)[0,0], confusion_matrix(y, predictions_rc)[1,0], confusion_matrix(y, predictions_rc)[1,1], confusion_matrix(y, predictions_rc)[0,1]],
                  [confusion_matrix(y, predictions_rfc)[0,0], confusion_matrix(y, predictions_rfc)[1,0], confusion_matrix(y, predictions_rfc)[1,1], confusion_matrix(y, predictions_rfc)[0,1]]]

    ## No CV
    if dictionary['cv_type'] == 'none':
        ## Add ML models std
        forecasts_names = forecasts_names + ['RC_std', 'RFC_std']    
        for predictions_ensemble in [predictions_rc_ensemble, predictions_rfc_ensemble]:
            metrics_data = []
            for k in np.arange(len(predictions_ensemble)):
                pred =  pd.Series(predictions_ensemble[k], copy = False, index = y.index)
                metrics_metric = []
                for metric, i, j in zip(metrics_names, [0,0,0,0,1,1,0], [0,0,0,0,0,1,1]): 
                    if metric == 'G-Mean':
                        metrics_metric.append(gmean_score_bin(filter_dates_like(y, pred), pred))
                    if metric == 'TPR':
                        metrics_metric.append(tpr(confusion_matrix(filter_dates_like(y, pred), pred)))
                    if metric == 'FPR':
                        metrics_metric.append(fpr(confusion_matrix(filter_dates_like(y, pred), pred)))
                    if metric in ['TN', 'FN', 'TP', 'FP']:
                        metrics_metric.append(confusion_matrix(filter_dates_like(y, pred), pred)[i,j])
                metrics_data.append(metrics_metric)
            metrics = pd.DataFrame(metrics_data, columns = metrics_names, index = np.arange(len(predictions_ensemble)))
            #### Standard deviation
            metrics_std = np.round(metrics.std(), decimals = 3)
            #### Add to metrics
            float_metrics = float_metrics + [[metrics_std['G-Mean'], metrics_std['TPR'], metrics_std['FPR']]]
            int_metrics = int_metrics + [[metrics_std['TN'], metrics_std['FN'], metrics_std['TP'], metrics_std['FP']]]
        
        if ecmwf is not None: 
            ### Add ECMWF
            forecasts_names = forecasts_names + ['ECMWF', 'ECMWF_std']
            ecmwf_metrics_data = []
            for var in list(ecmwf):
                ecmwf_metrics_metric = []
                for metric, i, j in zip(metrics_names, [0,0,0,0,1,1,0], [0,0,0,0,0,1,1]): 
                    if metric == 'G-Mean':
                        ecmwf_metrics_metric.append(gmean_score_bin(filter_dates_like(y, ecmwf[var]), ecmwf[var]))
                    if metric == 'TPR':
                        ecmwf_metrics_metric.append(tpr(confusion_matrix(filter_dates_like(y, ecmwf[var]), ecmwf[var])))
                    if metric == 'FPR':
                        ecmwf_metrics_metric.append(fpr(confusion_matrix(filter_dates_like(y, ecmwf[var]), ecmwf[var])))
                    if metric in ['TN', 'FN', 'TP', 'FP']:
                        ecmwf_metrics_metric.append(confusion_matrix(filter_dates_like(y, ecmwf[var]), ecmwf[var])[i,j])
                ecmwf_metrics_data.append(ecmwf_metrics_metric)
            ecmwf_metrics = pd.DataFrame(ecmwf_metrics_data, columns = metrics_names, index = list(ecmwf))
            #### Mean        
            ecmwf_metrics_mean = ecmwf_metrics.loc[target_name + '_mean']
            # Standard deviation
            ecmwf_metrics_std = np.round(ecmwf_metrics.drop(target_name + '_mean').std(), decimals = 3)
            #### Add to metrics
            float_metrics = float_metrics + [[ecmwf_metrics_mean['G-Mean'], ecmwf_metrics_mean['TPR'], ecmwf_metrics_mean['FPR']],
                                            [ecmwf_metrics_std['G-Mean'], ecmwf_metrics_std['TPR'], ecmwf_metrics_std['FPR']]]

            int_metrics = int_metrics + [[ecmwf_metrics_mean['TN'], ecmwf_metrics_mean['FN'], ecmwf_metrics_mean['TP'], ecmwf_metrics_mean['FP']],
                                        [ecmwf_metrics_std['TN'], ecmwf_metrics_std['FN'], ecmwf_metrics_std['TP'], ecmwf_metrics_std['FP']]]
            
    ### Combine the float and int metrics        
    metrics_text = np.ndarray((len(forecasts_names), len(metrics_names)), dtype = object)
    metrics_text[:,:len_metrics] = float_metrics
    metrics_text[:,len_metrics:] = int_metrics
    ### Round metrics
    metrics_text[:,:len_metrics] = np.round((metrics_text[:,:len_metrics]).astype(np.double), decimals = 3) 
    metrics_text[:,len_metrics:] = metrics_text[:,len_metrics:].astype(int)
    ### Dataset
    metrics_dset = xr.Dataset({'value': (('forecast', 'metric'), metrics_text)}, coords = {'forecast': forecasts_names, 'metric': metrics_names})
    ### Expand dimension
    metrics_dset = metrics_dset.assign_coords({'balance': balance_type})
    ## Print table
    metrics_table = ax.table(cellText = metrics_text,
                      rowLabels = forecasts_names,
                      colLabels = metrics_names,
                      loc = 'center')
    metrics_table.set_fontsize(12)
    metrics_table.scale(2, 4.5) 
    ## Save plot
    dir_name = dictionary['path_plots_prediction'] + subset + '/' + target_name + '/metrics_table/' + balance_type + '/'     
    if outer_split_num_ != None:
        save_name = dir_name + 'metrics_table_' + subset + '_' + target_name + '_' + balance_type + '_lead_time_' + str(lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '.png'
    else: save_name = dir_name + 'metrics_table_' + subset + '_' + target_name + '_' + balance_type + '_lead_time_' + str(lead_time) + '_weeks.png'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches='tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    
    ## Return metrics as dataset
    return metrics_dset


# In[33]:


def build_metrics_proba_classi(y, predictions_proba_rc, predictions_proba_rfc, predictions_proba_rfc_ensemble, 
                               persistence, climatology, target_name, lead_time, subset, balance_type, 
                               predictions_proba_rc_ensemble = None, ecmwf = None, outer_split_num_ = None): 

    """
    inputs
    ------
    y                                    pd.Series : time series of the true target
    predictions_proba_rc                  np.array : time series of the predicted target with the RC model
    predictions_proba_rfc                 np.array : time series of the predicted target with the RFC model
    predictions_proba_rfc_ensemble            dict : set of time series of the predicted target (same
                                                     length and time as y) by each estimator (tree) of the RFC model
    persistence                          np.array  : time series of the target's persistence 
    climatology                          np.array  : time series of the target's climatology 
    target_name                                str : name of target variable 
    lead_time                                  int : number of weeks for prediction lead time
    subset                                     str : 'train' or 'test'
    balance_type                               str : 'undersampling' or 'oversampling'

    " predictions_proba_rc_ensemble           dict : set of time series of the predicted target (same
                                                     length and time as y) by the RC models trained on each leave-one-out subset "
    " ecmwf                              np.array  : time series of the predicted target with ECMWF. This argument is optional. "
    " outer_split_num_                         int : counter for outer splits. This argument is optional. "
                                                     

    outputs
    -------
    metrics_dset                        xr.Dataset : 1 classification probability metric for every forecast

    
    Displays table with 1 metric (ROC AUC) to evaluate (and compare) the performance 
    of the classification model
        ROC AUC: Receiver Operating Characteristic Area Under Curve
    
    """
    
    # Add a table at the bottom of the axes
    fig, ax = plt.subplots()
    ## Hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ## Define column names
    metrics_names = ['ROC AUC']
    len_metrics = len(metrics_names)
    ## Define row names
    forecasts_names = ['Ground Truth', 'Persistence', 'Climatology', 'RC', 'RFC']
    ## Compute metrics
    if dictionary['verbosity'] > 2: print('Compute metrics')
    metrics_text = [[roc_auc_score(y, y)],
                   [roc_auc_score(y, persistence)],
                   [roc_auc_score(y, climatology)],
                   [roc_auc_score(y, predictions_proba_rc)],
                   [roc_auc_score(y, predictions_proba_rfc)]]
    ## No CV
    if dictionary['cv_type'] == 'none':
        ## Add ML models std
        forecasts_names = forecasts_names + ['RC_std', 'RFC_std']    
        for predictions_proba_ensemble in [predictions_proba_rc_ensemble, predictions_proba_rfc_ensemble]:
            metrics_data = []
            for k in np.arange(len(predictions_proba_ensemble)):
                pred =  pd.Series(predictions_proba_ensemble[k], copy = False, index = y.index)
                metrics_metric = []
                for metric in metrics_names: 
                    if metric == 'ROC AUC':
                        metrics_metric.append(roc_auc_score(filter_dates_like(y, pred), pred))
                metrics_data.append(metrics_metric)
            metrics = pd.DataFrame(metrics_data, columns = metrics_names, index = np.arange(len(predictions_proba_ensemble)))
            # Standard deviation
            metrics_std = metrics.std()
            # Add to metrics
            metrics_text = metrics_text + [[metrics_std['ROC AUC']]]

        if ecmwf is not None: 
            ### Add ECMWF
            forecasts_names = forecasts_names + ['ECMWF', 'ECMWF_std']
            ecmwf_metrics_data = []
            for var in list(ecmwf):
                ecmwf_metrics_metric = []
                for metric in metrics_names: 
                    if metric == 'ROC AUC':
                        ecmwf_metrics_metric.append(roc_auc_score(filter_dates_like(y, ecmwf[var]), ecmwf[var]))
                ecmwf_metrics_data.append(ecmwf_metrics_metric)
            ecmwf_metrics = pd.DataFrame(ecmwf_metrics_data, columns = metrics_names, index = list(ecmwf))
            # Mean
            ecmwf_metrics_mean = ecmwf_metrics.loc[target_name + '_mean']
            # Standard deviation
            ecmwf_metrics_std = ecmwf_metrics.drop(target_name + '_mean').std()
            # Add to metrics
            metrics_text = metrics_text + [[ecmwf_metrics_mean['ROC AUC']],
                                           [ecmwf_metrics_std['ROC AUC']]]
                                                                    
    ### Round metrics
    metrics_text = np.round(metrics_text, decimals = 3) 
    ### Dataset
    metrics_dset = xr.Dataset({'value': (('forecast', 'metric'), metrics_text)}, coords = {'forecast': forecasts_names, 'metric': metrics_names})
    ### Expand dimension
    metrics_dset = metrics_dset.assign_coords({'balance': balance_type})
    ## Print table
    metrics_table = ax.table(cellText = metrics_text,
                      rowLabels = forecasts_names,
                      colLabels = metrics_names,
                      loc = 'center')
    metrics_table.set_fontsize(12)
    metrics_table.scale(0.7, 2) 
    ## Save plot    
    dir_name = dictionary['path_plots_prediction'] + subset + '/' + target_name + '/proba_metrics_table/' + balance_type + '/' 
    if outer_split_num_ != None:
        save_name = dir_name + 'proba_metrics_table_' + subset + '_' + target_name + '_' + balance_type + '_lead_time_' + str(lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '.png'
    else: save_name = dir_name + 'proba_metrics_table_' + subset + '_' + target_name + '_' + balance_type + '_lead_time_' + str(lead_time) + '_weeks.png'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches='tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    
    ## Return metrics as dataset
    return metrics_dset


# In[34]:


def save_metrics(metrics, prediction_type, subset, target_name, lead_time, balance_type, outer_split_num_ = None):
    
    """
    inputs
    ------
    metrics            xr.Dataset : several metrics for each forecast
    prediction_type           str : 'regr', 'classi' or 'proba_classi'
    subset                    str : 'train' or 'test'
    target_name               str : name of target variable 
    lead_time                 int : Lead time of prediction
    balance_type              str : name of balance type 'undersampling', 'oversampling' or 'none'
    " outer_split_num_          int : counter for outer splits "

    outputs
    -------
    None. Saves the metrics and the corresponding parameters to a file
    
    """
    
    path = dictionary['path_metrics'] + subset + '/'
    if outer_split_num_ != None:
        save_name = subset + '_metrics_' + prediction_type + '_' + balance_type + '_' + target_name + '_lead_time_' + str(lead_time) + '_outer_split_' + str(outer_split_num_) + '.nc'
    else: save_name = subset + '_metrics_' + prediction_type + '_' + balance_type + '_' + target_name + '_lead_time_' + str(lead_time) + '.nc'

    save_to_file(metrics, path, save_name, 'nc')
    


# In[35]:


def construct_metrics_dset(tgn, pred_type, subset, balance_type):
    
    """
    inputs
    ------
    tgn                              str : name of target variable 
    pred_type                        str : 'regr', 'classi' or 'proba_classi'
    subset                           str : 'train' or 'test'
    balance_type                     str : name of balance type 'undersampling', 'oversampling' or 'none'


    outputs
    -------
    metrics                    xr.Dataset: dataset of metrics data with dimensions (forecast x metric x lead_time) (in the case of nested CV it is the avg over all outer splits)
    
    """
    
    ## Read metrics data to combined dataset   
    path = dictionary['path_metrics'] + subset + '/'
    metrics = []
    for lead_time in dictionary['lead_times']:
        metrics_lt = []
        # Nested CV
        if dictionary['cv_type'] == 'nested':
            for outer_split in np.arange(1, dictionary['num_outer_folds'] + 1):    
                file_name = path + subset + '_metrics_' + pred_type + '_' + balance_type + '_' + tgn + '_lead_time_' + str(lead_time) + '_outer_split_' + str(outer_split) + '.nc'
                metrics_lt.append(xr.open_dataset(file_name).assign_coords(outer_split = outer_split).expand_dims('outer_split'))   
            metrics_dset = xr.concat(metrics_lt, dim = 'outer_split')
        # Not nested CV
        if dictionary['cv_type'] == 'none':
            file_name = path + subset + '_metrics_' + pred_type + '_' + balance_type + '_' + tgn + '_lead_time_' + str(lead_time) + '.nc'
            metrics_dset = xr.open_dataset(file_name)
        metrics.append(metrics_dset.assign_coords(lead_time = lead_time).expand_dims('lead_time'))     
    metrics = xr.concat(metrics, dim = 'lead_time') 

    # Nested CV
    if dictionary['cv_type'] == 'nested':
        ## Standard deviation and average of metrics across splits
        metrics = metrics.expand_dims(dim = 'statistics')
        metrics = xr.concat([metrics.mean(dim = 'outer_split'), 
                             metrics.std(dim = 'outer_split')], 
                             'statistics')
        metrics = metrics.assign_coords({'statistics': ['mean', 'std']}) 

    return metrics
    


# In[ ]:




