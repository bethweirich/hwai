""" HWAI metrics """

# Author: Elizabeth Weirich Benet 
# License: MIT License 2022

# Import packages
import xarray as xr
import pandas as pd
import numpy as np
import os
import itertools 
import matplotlib.pyplot as plt
## Metrics
from sklearn.metrics import (auc,
                             confusion_matrix,
                             roc_auc_score,
                             mean_squared_error,
                             roc_curve,
                             precision_recall_curve,
                             brier_score_loss)
from scipy import stats
from scipy.stats import pearsonr

# Import own functions
## Utils
from utils import (filter_dates_like, 
                   save_to_file)

# Import constants
from const import dictionary


# **----------------------------------------------------------------------------------------------------------------------------------------------**

# Function definitions 

def pr_auc_score(y, y_pred):
    
    """
    inputs
    ------
      y                    array or series : ground truth time series
      y_pred               array or series : forecast time series 


    outputs
    -------
      _auc_precision_recall          float : precision-recall area under curve for y_pred forecast
    
    """
    
    np.seterr(divide = 'ignore', invalid = 'ignore')
    _precision, _recall, _thresholds = precision_recall_curve(y_true = y, probas_pred = y_pred)
    # Use AUC function to calculate the area under the curve of precision recall curve
    _auc_precision_recall = auc(_recall, _precision)
    if np.isnan(_auc_precision_recall):
        _auc_precision_recall = 0. 
    np.seterr(divide = 'warn', invalid = 'warn')
    
    return _auc_precision_recall 


def frequency_bias(y, y_pred):
    
    """
    inputs
    ------
      y                    array or series : ground truth time series
      y_pred               array or series : binary classification forecast time series 


    outputs
    -------
      b                              float : frequency bias for y_pred forecast
    
    """
    
    # For a given threshold
    #Confusion matrix
    tp = confusion_matrix(y, y_pred)[1,1]
    fn = confusion_matrix(y, y_pred)[1,0]
    fp = confusion_matrix(y, y_pred)[0,1]
    
    # Frequency bias
    b = (tp + fp)/(tp + fn)
    
    return b


def frequency_bias_all_th(y, y_pred):
    
    """
    inputs
    ------
      y                    array or series : ground truth time series
      y_pred               array or series : probabilistic classifiaction forecast time series 


    outputs
    -------
      b                               list : frequency bias for each threshold for y_pred forecast
      thresholds                      list : probability thresholds used for binarization of the probabilistic classification forecast
    
    """
    
    # For all thresholds
    # calculate base rate
    s = np.sum(y) / len(y)    
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    # calculate frequency bias for each threshold 
    b = tpr + (fpr * (1 - s))/s
    
    return b, thresholds


def extremal_dependence_index(y, y_pred):
    
    """
    inputs
    ------
      y                    array or series : ground truth time series
      y_pred               array or series : binary classification forecast time series 


    outputs
    -------
      edi                            float : extremal dependence index (EDI) for y_pred forecast
    
    """
    
    # Confusion matrix for a given threshold
    conf_mat = confusion_matrix(y, y_pred).astype(float)
    
    # Laplace correction (add small number if zero)
    for i, j in itertools.product(range(2), range(2)):
        if conf_mat[i][j] == 0: conf_mat[i,j] = 1e-12
        
    # Extract values
    tp = conf_mat[1,1]
    fn = conf_mat[1,0]
    fp = conf_mat[0,1]
    tn = conf_mat[0,0]
        
    # TPR
    tpr = tp / (tp + fn)
    
    # FPR
    fpr = fp / (tn + fp)
    
    # EDI
    edi = (np.log(fpr) - np.log(tpr))/(np.log(fpr) + np.log(tpr))
    
    return edi

def threat_score_all_th(y, y_pred_proba, thresholds):
    
    """
    inputs
    ------
      y                    array or series : ground truth time series
      y_pred               array or series : probabilistic classifiaction forecast time series 
      thresholds                      list : probability thresholds used for binarization of the probabilistic classification forecast


    outputs
    -------
      ts                               list : threat score (TS) for each threshold for y_pred forecast
    
    """
    
    # Initialize
    ts = []
    
    for th in thresholds:    
        ## Binarize prediction with threshold
        y_pred = np.zeros(len(y_pred_proba))
        for i in np.where(y_pred_proba >= th):
            y_pred[i] = 1   
            
        # Confusion matrix for a given threshold
        tp = confusion_matrix(y, y_pred)[1,1]
        fn = confusion_matrix(y, y_pred)[1,0]
        fp = confusion_matrix(y, y_pred)[0,1]

        # TS
        ts_th = tp/(tp + fp + fn)
        ts.append(ts_th)
    
    return ts


def compute_score(score_name, a, b):
    
    """
    inputs
    ------
      score_name                  str : name of score to be computed
      a               array or series : true values 
      b               array or series : predicted values (values, probabilities or binary)

    outputs
    -------
      score                   float : score of a,b
    
    """
    
    if dictionary['verbosity'] > 4: print('score_name: ', score_name)
    if score_name == 'ROC AUC': score = roc_auc_score(a, b) # b is proba  
    elif score_name == 'BS': score = brier_score_loss(a,b) # b is proba         
    elif score_name == 'Corr': score = stats.pearsonr(a, b)[0] # b is value
    elif score_name == 'RMSE': score = mean_squared_error(a, b, squared = False) # b is value
        
    return score

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


def build_metrics_regr(y, predictions_rr, predictions_rfr, 
                       persistence, climatology, target_name, lead_time, subset, 
                       predictions_rr_ensemble = None, predictions_rfr_ensemble = None, 
                       ecmwf = None, outer_split_num_ = None): 

    """
    inputs
    ------
    y                                         pd.Series : time series of the true target
    predictions_rr                             np.array : time series of the predicted target by the RR model
    predictions_rfr                            np.array : time series of the predicted target by the RFR model
    persistence                                np.array : time series of the target's persistence 
    climatology                                np.array : time series of the target's climatology 
    target_name                                     str : name of target variable 
    lead_time                                       int : lead time of prediction in units of timestep
    subset                                          str : 'train_full' or 'test'
    
    " predictions_rr_ensemble          dict of np.array : set of time series of the predicted target by the RR models trained on each 
                                                          bootstrap subset (only for no CV case) "   
    " predictions_rfr_ensemble         dict of np.array : set of time series of the predicted target by the RFR models trained on each 
                                                          bootstrap subset (only for no CV case) "
    " ecmwf                                   np.array  : time series of the predicted target by ECMWF. This argument is optional. "
    " outer_split_num_                              int : counter for outer splits (only for nested CV case) "

    outputs
    -------
    metrics_dset                             xr.Dataset : 2 regression metrics for every forecast (Corr, RMSE)
    
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
    forecasts_names = ['Ground Truth', 'Persistence', 'Climatology', 'RR', 'RFR']
    ## Compute metrics
    metrics_text = [[mean_squared_error(y, y, squared = False), stats.pearsonr(y, y)[0]],
                    [mean_squared_error(y, persistence, squared = False), stats.pearsonr(y, persistence)[0]],
                    [mean_squared_error(y, climatology, squared = False), 0.],
                    [mean_squared_error(y, predictions_rr, squared = False), stats.pearsonr(y, predictions_rr)[0]],
                    [mean_squared_error(y, predictions_rfr, squared = False), stats.pearsonr(y, predictions_rfr)[0]]]

    
    ### Correct entry for climatology corr manually
    if (target_name != 't2m'): metrics_text[2][1] = stats.pearsonr(y, climatology)[0]
    
    if dictionary['cv_type'] == 'none':
        ### Add std for ML models w.r.t the actual prediction
        forecasts_names = forecasts_names + ['RR_std', 'RFR_std']    
        for pred, predictions_ensemble in zip([predictions_rr, predictions_rfr], [predictions_rr_ensemble, predictions_rfr_ensemble]):
            metrics_data = []
            for i in np.arange(len(predictions_ensemble)):
                pred_ens_mem = pd.Series(predictions_ensemble[i], copy = False, index = y.index)
                metrics_metric = []
                for metric in metrics_names: 
                    if metric == 'RMSE':
                        metrics_metric.append(mean_squared_error(y, pred_ens_mem, squared = False))
                    if metric == 'Correlation':
                        metrics_metric.append(stats.pearsonr(y, pred_ens_mem.astype(float))[0])
                metrics_data.append(metrics_metric)
            metrics = pd.DataFrame(metrics_data, columns = metrics_names, index = np.arange(len(predictions_ensemble)))
            #### Standard deviation
            metrics_std = metrics.std(ddof = 1)
            #### Add to metrics
            metrics_text = metrics_text + [[metrics_std.RMSE, metrics_std.Correlation]]

        if ecmwf is not None: 
            ### Add ECMWF forecast and uncertainty via the std w.r.t the ensemble mean
            forecasts_names = forecasts_names + ['ECMWF', 'ECMWF_std']
            ecmwf_metrics_data = []
            for var in list(ecmwf):
                ecmwf_metrics_metric = []
                for metric in metrics_names: 
                    if metric == 'RMSE':
                        ecmwf_metrics_metric.append(mean_squared_error(filter_dates_like(y, ecmwf[var]), ecmwf[var], squared = False))
                    if metric == 'Correlation':
                        ecmwf_metrics_metric.append(stats.pearsonr(filter_dates_like(y, ecmwf[var]), ecmwf[var].astype(float))[0])
                ecmwf_metrics_data.append(ecmwf_metrics_metric)
            ecmwf_metrics = pd.DataFrame(ecmwf_metrics_data, columns = metrics_names, index = list(ecmwf))
            #### Mean
            ecmwf_metrics_mean = ecmwf_metrics.loc[target_name + '_mean']
            # Standard deviation
            ecmwf_metrics_std = ecmwf_metrics.drop(target_name + '_mean').std(ddof = 1)
            #### Add to metrics
            metrics_text = metrics_text + [[ecmwf_metrics_mean.RMSE, ecmwf_metrics_mean.Correlation],
                                           [ecmwf_metrics_std.RMSE, ecmwf_metrics_std.Correlation]]

    ## Prepare dataset
    ### Dataset
    metrics_dset = xr.Dataset({'value': (('forecast', 'metric'), metrics_text)}, coords = {'forecast': forecasts_names, 'metric': metrics_names})
    
    # Plot table
    ## Round metrics
    metrics_text_rounded = np.round(metrics_text, decimals = 2) 
    metrics_table = ax.table(cellText = metrics_text_rounded,
                      rowLabels = forecasts_names,
                      colLabels = metrics_names,
                      loc = 'center')
    metrics_table.set_fontsize(12)
    metrics_table.scale(0.7, 2) 
    ## Save plot
    dir_name = dictionary['path_plots'] + subset + '/regr/' + target_name + '/metrics_table/'    
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



def build_metrics_classi(y, predictions_rc, predictions_rfc, 
                         persistence, climatology, target_name, lead_time, subset, 
                         predictions_rc_ensemble = None, predictions_rfc_ensemble = None, 
                         ecmwf = None, outer_split_num_ = None): 

    """
    inputs
    ------
    y                                    pd.Series : time series of the true target
    predictions_rc                        np.array : time series of the predicted target by the RC model
    predictions_rfc                       np.array : time series of the predicted target by the RFC model
    persistence                           np.array : time series of the target's persistence 
    climatology                           np.array : time series of the target's climatology 
    target_name                                str : name of target variable 
    lead_time                                  int : lead time of prediction in units of timestep
    subset                                     str : 'train_full' or 'test'
    
    " predictions_rc_ensemble                 dict : set of time series of the predicted target by the RC models trained on each 
                                                     bootstrap subset (only for no CV case) "    
    " predictions_rfc_ensemble                dict : set of time series of the predicted target by the RFC models trained on each 
                                                     bootstrap subset (only for no CV case) "                                         
    " ecmwf                               np.array : time series of the predicted target by ECMWF. This argument is optional. "
    " outer_split_num_                         int : counter for outer splits (only for nested CV case) "
    

    outputs
    -------
    metrics_dset                        xr.Dataset : 8 binary classification metrics for every forecast
    
    Displays table with 8 different metrics (B, EDI, TPR, FPR, Conf:(TN, FN, TP, FP)) to evaluate (and compare) the performance of the classification 
    model
        B : frequency bias
        EDI : Extremal Dependence Index
        TPR : True Positive Rate
        FPR : False Positive Rate
        Conf: Confusion Matrix [[TN, FP],
                                [FN, TP]]
    
    """

    # Initialize plot: add a table at the bottom of the axes
    fig, ax = plt.subplots()
    ## Hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ## Define column names
    metrics_names = ['B', 'EDI', 'TPR', 'FPR', 'TN', 'FN', 'TP', 'FP']
    ## Define separation between continuous and discrete-valued metrics (extract 4 confusion matrix entries)
    len_metrics = len(metrics_names) - 4
    ## Define row names
    forecasts_names = ['Ground Truth', 'Persistence', 'Climatology', 'RC', 'RFC'] 
    
    ## Compute metrics
    if dictionary['verbosity'] > 2: print('Compute metrics')
    float_metrics = [[frequency_bias(y, y), extremal_dependence_index(y, y), tpr(confusion_matrix(y, y)), fpr(confusion_matrix(y, y))],
                    [frequency_bias(y, persistence), extremal_dependence_index(y, persistence), tpr(confusion_matrix(y, persistence)), fpr(confusion_matrix(y, persistence))],
                    [frequency_bias(y, climatology), extremal_dependence_index(y, climatology), tpr(confusion_matrix(y, climatology)), fpr(confusion_matrix(y, climatology))],
                    [frequency_bias(y, predictions_rc), extremal_dependence_index(y, predictions_rc), tpr(confusion_matrix(y, predictions_rc)), fpr(confusion_matrix(y, predictions_rc))],
                    [frequency_bias(y, predictions_rfc), extremal_dependence_index(y, predictions_rfc), tpr(confusion_matrix(y, predictions_rfc)), fpr(confusion_matrix(y, predictions_rfc))]]
        
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
                for metric, i, j in zip(metrics_names, [0,0,0,0,0,1,1,0], [0,0,0,0,0,0,1,1]): 
                    if metric == 'B':
                        metrics_metric.append(frequency_bias(filter_dates_like(y, pred), pred))
                    if metric == 'EDI':
                        metrics_metric.append(extremal_dependence_index(filter_dates_like(y, pred), pred))
                    if metric == 'TPR':
                        metrics_metric.append(tpr(confusion_matrix(filter_dates_like(y, pred), pred)))
                    if metric == 'FPR':
                        metrics_metric.append(fpr(confusion_matrix(filter_dates_like(y, pred), pred)))
                    if metric in ['TN', 'FN', 'TP', 'FP']:
                        metrics_metric.append(confusion_matrix(filter_dates_like(y, pred), pred)[i,j])
                metrics_data.append(metrics_metric)
            metrics = pd.DataFrame(metrics_data, columns = metrics_names, index = np.arange(len(predictions_ensemble)))
            #### Standard deviation
            metrics_std = metrics.std(ddof = 1)
            #### Add to metrics
            float_metrics = float_metrics + [[metrics_std['B'], metrics_std['EDI'], metrics_std['TPR'], metrics_std['FPR']]]
            int_metrics = int_metrics + [[metrics_std['TN'], metrics_std['FN'], metrics_std['TP'], metrics_std['FP']]]

        if ecmwf is not None: 
            ### Add ECMWF
            forecasts_names = forecasts_names + ['ECMWF', 'ECMWF_std']
            ecmwf_metrics_data = []
            for var in list(ecmwf):
                ecmwf_metrics_metric = []
                for metric, i, j in zip(metrics_names, [0,0,0,0,0,1,1,0], [0,0,0,0,0,0,1,1]): 
                    if metric == 'B':
                        ecmwf_metrics_metric.append(frequency_bias(filter_dates_like(y, ecmwf[var]), ecmwf[var]))
                    if metric == 'EDI':
                        ecmwf_metrics_metric.append(extremal_dependence_index(filter_dates_like(y, ecmwf[var]), ecmwf[var]))
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
            ecmwf_metrics_std = ecmwf_metrics.drop(target_name + '_mean').std(ddof = 1)
            #### Add to metrics
            float_metrics = float_metrics + [[ecmwf_metrics_mean['B'], ecmwf_metrics_mean['EDI'], ecmwf_metrics_mean['TPR'], ecmwf_metrics_mean['FPR']],
                                            [ecmwf_metrics_std['B'], ecmwf_metrics_std['EDI'], ecmwf_metrics_std['TPR'], ecmwf_metrics_std['FPR']]]
            int_metrics = int_metrics + [[ecmwf_metrics_mean['TN'], ecmwf_metrics_mean['FN'], ecmwf_metrics_mean['TP'], ecmwf_metrics_mean['FP']],
                                        [ecmwf_metrics_std['TN'], ecmwf_metrics_std['FN'], ecmwf_metrics_std['TP'], ecmwf_metrics_std['FP']]]
            

    ## Prepare dataset
    ### Combine the float and int metrics        
    metrics_text = np.ndarray((len(forecasts_names), len(metrics_names)), dtype = object)
    metrics_text[:,:len_metrics] = float_metrics
    metrics_text[:,len_metrics:] = int_metrics
    ### Define type
    metrics_text[:,:len_metrics] = metrics_text[:,:len_metrics].astype(np.double) 
    metrics_text[:,len_metrics:] = metrics_text[:,len_metrics:].astype(int)
    ### Dataset
    metrics_dset = xr.Dataset({'value': (('forecast', 'metric'), metrics_text)}, coords = {'forecast': forecasts_names, 'metric': metrics_names})
    
    
    ## Plot table
    ### Round float metrics and keep int
    metrics_text_rounded = np.ndarray((len(forecasts_names), len(metrics_names)), dtype = object)
    metrics_text_rounded[:,:len_metrics] = np.round((metrics_text[:,:len_metrics]).astype(np.double), decimals = 2)
    metrics_text_rounded[:,len_metrics:] = metrics_text[:,len_metrics:].astype(int)
    metrics_table = ax.table(cellText = metrics_text_rounded,
                      rowLabels = forecasts_names,
                      colLabels = metrics_names,
                      loc = 'center')     
    metrics_table.set_fontsize(12)
    metrics_table.scale(2, 3.5) 
    ## Save plot
    dir_name = dictionary['path_plots'] + subset + '/classi/' + target_name + '/metrics_table/'   
    if outer_split_num_ != None:
        save_name = dir_name + 'metrics_table_' + subset + '_' + target_name + '_lead_time_' + str(lead_time) + '_weeks_outer_split_' + str(outer_split_num_) + '.png'
    else: save_name = dir_name + 'metrics_table_' + subset + '_' + target_name + '_lead_time_' + str(lead_time) + '_weeks.png'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(save_name, bbox_inches = 'tight')
    if dictionary['verbosity'] > 0: plt.show()
    plt.close()
    
    ## Return metrics as dataset
    return metrics_dset



def build_metrics_proba_classi(y, predictions_proba_rc, predictions_proba_rfc,  
                               persistence, climatology, target_name, lead_time, subset, 
                               predictions_proba_rc_ensemble = None, predictions_proba_rfc_ensemble = None,
                               ecmwf = None, outer_split_num_ = None): 

    """
    inputs
    ------
    y                                    pd.Series : time series of the true target
    predictions_proba_rc                  np.array : time series of the predicted target by the RC model
    predictions_proba_rfc                 np.array : time series of the predicted target by the RFC model
    persistence                           np.array : time series of the target's persistence 
    climatology                           np.array : time series of the target's climatology 
    target_name                                str : name of target variable 
    lead_time                                  int : lead time of prediction in units of timestep
    subset                                     str : 'train_full' or 'test'

    " predictions_proba_rc_ensemble           dict : set of time series of the predicted target by the RC models trained on each 
                                                     bootstrap subset (only for no CV case) "
    " predictions_proba_rfc_ensemble          dict : set of time series of the predicted target by the RFC models trained on each 
                                                     bootstrap subset (only for no CV case) "
    " ecmwf                              np.array  : time series of the predicted target by ECMWF. This argument is optional. "
    " outer_split_num_                         int : counter for outer splits (only for nested CV case) "
                                                     

    outputs
    -------
    metrics_dset                        xr.Dataset : 2 probabilistic classification metrics for every forecast (ROC AUC and BS)

    
    Displays table with 2 metrics (ROC AUC and BS) to evaluate (and compare) the performance 
    of the classification model
        ROC AUC: Receiver Operating Characteristic Area Under Curve
        BS: Brier score 
    
    """
    
    # Initialize plot: add a table at the bottom of the axes
    fig, ax = plt.subplots()
    ## Hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ## Define column names
    metrics_names = ['ROC AUC', 'BS']
    len_metrics = len(metrics_names)
    ## Define row names
    forecasts_names = ['Ground Truth', 'Persistence', 'Climatology', 'RC', 'RFC']
    ## Compute metrics
    if dictionary['verbosity'] > 2: print('Compute metrics')
    metrics_text = [[roc_auc_score(y, y), brier_score_loss(y, y)],
                   [roc_auc_score(y, persistence), brier_score_loss(y, persistence)],
                   [roc_auc_score(y, climatology), brier_score_loss(y, climatology)],
                   [roc_auc_score(y, predictions_proba_rc), brier_score_loss(y, predictions_proba_rc)],
                   [roc_auc_score(y, predictions_proba_rfc), brier_score_loss(y, predictions_proba_rfc)]]
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
                    if metric == 'BS':
                        metrics_metric.append(brier_score_loss(filter_dates_like(y, pred), pred))
                metrics_data.append(metrics_metric)
            metrics = pd.DataFrame(metrics_data, columns = metrics_names, index = np.arange(len(predictions_proba_ensemble)))
            # Standard deviation
            metrics_std = metrics.std(ddof = 1)
            # Add to metrics
            metrics_text = metrics_text + [[metrics_std['ROC AUC'], metrics_std['BS']]]

        if ecmwf is not None: 
            ### Add ECMWF
            forecasts_names = forecasts_names + ['ECMWF', 'ECMWF_std']
            ecmwf_metrics_data = []
            for var in list(ecmwf):
                ecmwf_metrics_metric = []
                for metric in metrics_names: 
                    if metric == 'ROC AUC':
                        ecmwf_metrics_metric.append(roc_auc_score(filter_dates_like(y, ecmwf[var]), ecmwf[var]))
                    if metric == 'BS':
                        ecmwf_metrics_metric.append(brier_score_loss(filter_dates_like(y, ecmwf[var]), ecmwf[var]))
                ecmwf_metrics_data.append(ecmwf_metrics_metric)
            ecmwf_metrics = pd.DataFrame(ecmwf_metrics_data, columns = metrics_names, index = list(ecmwf))
            # Mean
            ecmwf_metrics_mean = ecmwf_metrics.loc[target_name + '_mean']
            # Standard deviation
            ecmwf_metrics_std = ecmwf_metrics.drop(target_name + '_mean').std(ddof = 1)
            # Add to metrics
            metrics_text = metrics_text + [[ecmwf_metrics_mean['ROC AUC'], ecmwf_metrics_mean['BS']],
                                           [ecmwf_metrics_std['ROC AUC'], ecmwf_metrics_std['BS']]]
            
                                                                    
    ## Prepare dataset
    ### Dataset
    metrics_dset = xr.Dataset({'value': (('forecast', 'metric'), metrics_text)}, coords = {'forecast': forecasts_names, 'metric': metrics_names})
    
    
    ## Plot table
    ### Round metrics
    metrics_text_rounded = np.round(metrics_text, decimals = 2) 
    metrics_table = ax.table(cellText = metrics_text_rounded,
                      rowLabels = forecasts_names,
                      colLabels = metrics_names,
                      loc = 'center')
    metrics_table.set_fontsize(12)
    metrics_table.scale(0.7, 2) 
    ## Save plot    
    dir_name = dictionary['path_plots'] + subset + '/proba_classi/' + target_name + '/metrics_table/'  
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



def save_metrics(metrics, prediction_type, subset, target_name, lead_time, outer_split_num_ = None):
    
    """
    inputs
    ------
    metrics                         xr.Dataset : several metrics for each forecast
    prediction_type                        str : 'regr', 'classi' or 'proba_classi'
    subset                                 str : 'train_full' or 'test'
    target_name                            str : name of target variable 
    lead_time                              int : lead time of prediction in units of timestep
    " outer_split_num_                     int : counter for outer splits (only for nested CV case) "

    outputs
    -------
    None. Saves the metrics and the corresponding parameters to a file
    
    """
    
    path = dictionary['path_metrics'] + subset + '/'
    if outer_split_num_ != None:
        save_name = subset + '_metrics_' + prediction_type + '_' + target_name + '_lead_time_' + str(lead_time) + '_outer_split_' + str(outer_split_num_) + '.nc'
    else: save_name = subset + '_metrics_' + prediction_type + '_' + target_name + '_lead_time_' + str(lead_time) + '.nc'

    save_to_file(metrics, path, save_name, 'nc')
    

    
def construct_metrics_dset(tgn, pred_type, subset):
    
    """
    inputs
    ------
    tgn                              str : name of target variable 
    pred_type                        str : 'regr', 'classi' or 'proba_classi'
    subset                           str : 'train_full' or 'test'


    outputs
    -------
    metrics                    xr.Dataset: dataset of metrics data with dimensions (forecast x metric x lead_time) 
                                           (in the case of nested CV it is the avg over all outer splits)
    
    """
    
    # Read metrics data to combined dataset   
    path = dictionary['path_metrics'] + subset + '/'
    metrics = []
    for lead_time in dictionary['lead_times']:
        metrics_lt = []
        ## Nested CV
        if dictionary['cv_type'] == 'nested':
            for outer_split in np.arange(1, dictionary['num_outer_folds'] + 1):    
                file_name = path + subset + '_metrics_' + pred_type + '_' + tgn + '_lead_time_' + str(lead_time) + '_outer_split_' + str(outer_split) + '.nc'
                metrics_lt.append(xr.open_dataset(file_name).assign_coords(outer_split = outer_split).expand_dims('outer_split'))   
            metrics_dset = xr.concat(metrics_lt, dim = 'outer_split')
        ## Not nested CV
        if dictionary['cv_type'] == 'none':
            file_name = path + subset + '_metrics_' + pred_type + '_' + tgn + '_lead_time_' + str(lead_time) + '.nc'
            metrics_dset = xr.open_dataset(file_name)
        metrics.append(metrics_dset.assign_coords(lead_time = lead_time).expand_dims('lead_time'))     
    metrics = xr.concat(metrics, dim = 'lead_time') 

    # Nested CV
    if dictionary['cv_type'] == 'nested':
        ## Standard deviation and average of metrics across splits
        metrics = metrics.expand_dims(dim = 'statistics')
        metrics = xr.concat([metrics.mean(dim = 'outer_split'), 
                             metrics.std(dim = 'outer_split', ddof = 1)], 
                             'statistics')
        metrics = metrics.assign_coords({'statistics': ['mean', 'std']}) 

    return metrics
