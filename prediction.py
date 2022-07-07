#!/usr/bin/env python
# coding: utf-8

# # Prediction

# In[1]:


# Import packages
import matplotlib                             
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import xarray as xr
from sklearn.utils.extmath import softmax
import random
import itertools
import time

## Models
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, ElasticNet, RidgeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold

# Metrics
from sklearn.metrics import (precision_recall_curve,
                             roc_curve)


# In[3]:


# Import own functions
## Utils
from ipynb.fs.defs.utils import save_to_file, print_model_name, save_time_series, traverse, interpolate, print_all, compute_ones_percentage

## Plotting
from ipynb.fs.defs.plotting import (tree_plotter, 
                                    plot_random_forest_overfitting, 
                                    show_coeffs, 
                                    show_rf_imp, 
                                    plot_pred_time_series,
                                    plot_zoomed_in_pred_time_series,
                                    plot_cross_corr,
                                    plot_scatter_diagrams,
                                    plot_roc_curve,
                                    plot_pr_curve,
                                    plot_proba_histogram)

## Metrics
from ipynb.fs.defs.metrics import (compute_score,
                                   roc_auc_score, 
                                   pr_auc_score, 
                                   build_metrics_regr,
                                   build_metrics_classi,
                                   build_metrics_proba_classi,
                                   save_metrics)

## Reference forecasts
from ipynb.fs.defs.reference_forecasts import compute_reference_forecasts

## Preprocessing 2
from ipynb.fs.defs.preprocessing_part2 import lagged_features, split_sets, split_sets_nested_cv, PCA, rf_feature_selection, select_season, data_balance, loo_ensemble_rdm


# In[3]:


# Import dictionary
from ipynb.fs.full.const import dictionary


# In[4]:


class RidgeClassifierwithProba(RidgeClassifier):
    def predict_proba(self, X):
        d = self.decision_function(X)
        d_2d = np.c_[-d, d]
        return softmax(d_2d)


# In[7]:


def read_old_regr_ml_forecasts(tgn, _lead_time_, outer_split_num_ = None):
    
    """
    inputs
    ------
    tgn                            str : name of target
    _lead_time_                    int : lead time of prediction in weeks
    " outer_split_num_             int : counter for outer splits. This argument is optional. "

    
    outputs
    -------
    mlr_test
    rfr_test (and ensemble)
    r_train_full
    rfr_train_full (and ensemble)
    
    """   
    
    # Specify directory
    path = dictionary['path_time_series'] + tgn + '/'
    # Make combinations of models and subsets
    if dictionary['cv_type'] == 'none':
        models = ['MLR', 'MLR_ensemble', 'RFR', 'RFR_ensemble']
    elif dictionary['cv_type'] == 'nested':
        models = ['MLR', 'RFR', 'RFR_ensemble']
    subsets = ['test', 'train_full']
    combinations = list(itertools.product(subsets, models))
    # Run over all combinations and load numpy files
    d = {}
    for i in range(len(combinations)):
        subset = combinations[i][0]
        model = combinations[i][1]
        file = path + tgn + '_' + model + '_' + subset + '_lead_time_' + str(_lead_time_) 
        if dictionary['cv_type'] == 'nested': file = file + '_weeks_outer_split_' + str(outer_split_num_) + '.npy'
        elif dictionary['cv_type'] == 'none': file = file + '_weeks.npy'
        forecast = np.load(file)
        model_lc = str.lower(model)
        d[model_lc + '_' +  subset] = forecast
        
    if dictionary['cv_type'] == 'none':
        return d['mlr_test'], d['mlr_ensemble_test'], d['rfr_test'], d['rfr_ensemble_test'], d['mlr_train_full'], d['mlr_ensemble_train_full'], d['rfr_train_full'], d['rfr_ensemble_train_full']    
    elif dictionary['cv_type'] == 'nested':
        return d['mlr_test'], d['rfr_test'], d['rfr_ensemble_test'], d['mlr_train_full'], d['rfr_train_full'], d['rfr_ensemble_train_full']


# In[59]:


def read_old_classi_ml_forecasts(tgn, _lead_time_, balance_sign_, outer_split_num_ = None):
    
    """
    inputs
    ------
    tgn                            str : name of target
    _lead_time_                    int : lead time of prediction in weeks
    balance_sign_                  str : '', '-' or '+'
    " outer_split_num_             int : counter for outer splits. This argument is optional. "

    
    outputs
    -------
    proba_rc_test
    proba_rfc_test
    rc_test
    rfc_test
    proba_rc_train_full
    proba_rfc_train_full
    rc_train_full
    rfc_train_full
    
    """    

    # Specify directory
    path = dictionary['path_time_series'] + tgn + '/'
    # Make combinations of models and subsets
    if dictionary['cv_type'] == 'none':
        models = ['RC', 'RC_ensemble', 'RFC', 'RFC_ensemble']
        models_sign = ['RC' + balance_sign_, 'RC' + balance_sign_ + '_ensemble', 'RFC' + balance_sign_, 'RFC' + balance_sign_ + '_ensemble']
    elif dictionary['cv_type'] == 'nested':
        models = ['RC', 'RFC', 'RFC_ensemble']
        models_sign = ['RC' + balance_sign_, 'RFC' + balance_sign_, 'RFC' + balance_sign_ + '_ensemble']
    pred_type = ['_proba', '']
    subsets = ['test', 'train_full']
    combinations = list(itertools.product(subsets, pred_type, zip(models, models_sign)))
    # Run over all combinations and load numpy files
    d = {}
    for i in range(len(combinations)):
        subset = combinations[i][0]
        pred_type  = combinations[i][1]
        model = combinations[i][2][0]
        model_sign = combinations[i][2][1]
        file = path + tgn + '_' + model_sign + '_' + subset + pred_type + '_lead_time_' + str(_lead_time_) 
        if dictionary['cv_type'] == 'nested': file = file + '_weeks_outer_split_' + str(outer_split_num_) + '.npy'
        elif dictionary['cv_type'] == 'none': file = file + '_weeks.npy'
        forecast = np.load(file)
        model_lc = str.lower(model)
        d[model_lc + '_' +  subset + pred_type] = forecast
        
    if dictionary['cv_type'] == 'none':
        return (d['rc_test_proba'], d['rc_ensemble_test_proba'], d['rfc_test_proba'], d['rfc_ensemble_test_proba'], 
                d['rc_test'], d['rc_ensemble_test'], d['rfc_test'], d['rfc_ensemble_test'], 
                d['rc_train_full_proba'], d['rc_ensemble_train_full_proba'], d['rfc_train_full_proba'], d['rfc_ensemble_train_full_proba'], 
                d['rc_train_full'], d['rc_ensemble_train_full'], d['rfc_train_full'], d['rfc_ensemble_train_full'])
    elif dictionary['cv_type'] == 'nested':
        return (d['rc_test_proba'], d['rfc_test_proba'], d['rfc_ensemble_test_proba'],
                d['rc_test'], d['rfc_test'], d['rfc_ensemble_test'],
                d['rc_train_full_proba'], d['rfc_train_full_proba'], d['rfc_ensemble_train_full_proba'],
                d['rc_train_full'], d['rfc_train_full'], d['rfc_ensemble_train_full'])


# In[8]:


def predict_classi(model, X, th):
    
    """
    inputs
    ------
    model             function : trained ML model
    X             pd.Dataframe : matrix of attributes (time x lagged features)
    th                   float : threshold to binarize probability output      


    outputs
    -------
    pred             pd.Series : binary classification prediction
    pred_proba       pd.Series : probabilities of belonging to the class 1 (being a HW)
    
    """
    
    ## Predict
    pred_proba = model.predict_proba(X)[:,1]
    
    ## Binarize prediction with best threshold
    pred = np.zeros(len(pred_proba))
    for i in np.where(pred_proba >= th):
        pred[i] = 1     
    
    return pred, pred_proba


# In[10]:


def choose_best_proba_threshold_classi(model, model_name, X_tr, y_tr, X_v, y_v, tgn_, _lead_time_, outer_split_num__ = None):
    
    """
    inputs
    ------
    model                    function : trained ML model_
    model_name                    str : 'RC' or 'RFC'
    X_tr        dict of pd.Dataframes : matrix of attributes (time x lagged features) in train period
    y_tr            dict of pd.Series : array containing the target variable (time) in train period
    X_v         dict of pd.Dataframes : matrix of attributes (time x lagged features) in validation period
    y_v             dict of pd.Series : array containing the target variable (time) in validation period
    tgn_                          str : target name
    _lead_time_                   int : lead time for prediction
    " outer_split_num__           int : counter for outer splits "

    outputs
    -------
    best_threshold      float : probability threshold that optimizes classification metric
    
    """

    if dictionary['cv_type'] == 'none':
        outer_split_num__ = None
        
    gmean = {}
    tpr = {}
    fpr = {}
    for key in X_tr:  
        print('>> For ', key)
        # Fit again with train period for further hyperparameter optimization step
        model.fit(X_tr[key], y_tr[key])
        # Predict heat wave probability
        pred_proba = model.predict_proba(X_v[key])[:,1]
        # Calculate ROC AUC score
        if dictionary['verbosity'] > 1: print('ROC AUC score = %.3f' % (roc_auc_score(y_v[key], pred_proba))) 
        # Compute threshold to optimize G-mean or ROC AUC score
        ## Calculate ROC curve
        fpr_is, tpr_is, thresholds = roc_curve(y_v[key], pred_proba)                
        ## Make dataframe with index thresholds
        roc_df = pd.DataFrame({'TPR': tpr_is, 'FPR': fpr_is}, index = thresholds)
        ## Reverse index
        roc_df = roc_df.reindex(index = roc_df.index[::-1])
        ## Interpolate thresholds to universal linspace
        new_thresholds = np.linspace(0., 1., 299)
        roc_df_interp = interpolate(roc_df, new_thresholds)
        tpr[key] = roc_df_interp.TPR
        fpr[key] = roc_df_interp.FPR
        ## Calculate G-Mean for each threshold
        gmean[key] = np.sqrt(tpr[key] * (1 - fpr[key]))
           
    # Mean over inner split
    tpr_mean = np.mean(list(tpr.values()), axis = 0)  
    fpr_mean = np.mean(list(fpr.values()), axis = 0)
    gmean = np.mean(list(gmean.values()), axis = 0) 
    # Locate the index of the minimum cost for the average of the splits
    best_th_ix = np.argmax(gmean)  
    best_threshold = new_thresholds[best_th_ix]
    if dictionary['verbosity'] > 1: print('Best Threshold=%f' %(best_threshold))   
    # Plot ROC AUC curve and mark best threshold
    if dictionary['verbosity'] > 1: plot_roc_curve(fpr_mean, tpr_mean, best_th_ix, 'vali', tgn_, 'RC', _lead_time_, outer_split_num_ = outer_split_num__) 


    return best_threshold


# In[9]:


def optimize_random_forest_hyperparameters(X_train_, y_train_, X_train_full_, y_train_full_, X_vali_, y_vali_, optimize_metric_, tgn_, _lead_time_, balance_type_, outer_split_num___ = None):
    
    """
    inputs
    ------
    
    X_train_                   pd.Dataframe : matrix of attributes (time x lagged features) for training (to select the hyperparameters)
    y_train_                      pd.Series : array containing the target variable (time) for training (to select the hyperparameters)
    X_train_full_              pd.Dataframe : matrix of attributes (time x lagged features) for final training (after the hyperparameters are fixed)
    y_train_full_                 pd.Series : array containing the target variable (time) for final training (after the hyperparameters are fixed)   
    X_vali_                    pd.Dataframe : matrix of attributes (time x lagged features) for validating
    y_vali_                       pd.Series : array containing the target variable (time) for validating
    optimize_metric_                    str : metric to optimize: Regression ('Corr', 'RMSE'), Classification ('ROC AUC', 'PR AUC')
    tgn_                                str : name of target variable
    _lead_time_                         int : lead time of prediction in weeks
    balance_type_                 str : name of balance type 'undersampling', 'oversampling' or 'none'
    " outer_split_num___                int : counter for outer splits "
    

    outputs
    -------
    best_rf                        function : random forest model initialized with best hyperparameters and trained on train_full
    best_hyperparameters_dset_   xr.Dataset : hyperparameter set that optimizes optimize_metric_ when doing CV on train_full
    
    """
    
    start = time.time()
    
    if dictionary['verbosity'] > 1: print('** ', dictionary['hp_search_type'], ' RF hyperparameter optimization **')
    
    if dictionary['cv_type'] == 'none':
        outer_split_num___ = None
        
    
    # Initialize
    ## Random state
    seed = 7
    ## score_max with its minumum
    if optimize_metric_ == 'RMSE': score_max = 1000.
    else: score_max = 0.
    
    # Hyperparameter grid
    ## Number of trees in random forest
    n_estimators = [50, 100, 200, 400, 600]
    ## Minimum number of samples required at each leaf node
    first_guess_min_samples_leaf = int(len(X_train_full_)/100)
    step_min_samples_leaf = int(first_guess_min_samples_leaf/5)
    if step_min_samples_leaf < 1: step_min_samples_leaf = 1
    min_samples_leaf = np.arange(first_guess_min_samples_leaf - 5*step_min_samples_leaf, first_guess_min_samples_leaf + 10*step_min_samples_leaf, step_min_samples_leaf)
    ### Remove entries smaller than 1
    min_samples_leaf = min_samples_leaf[min_samples_leaf >= 1]
    ## Maximum number of levels in tree
    max_depth = np.arange(5, 15, 1)
    ## Create the hyperparameter grid
    hp_grid = {'n_estimators': n_estimators,           
               'min_samples_leaf': min_samples_leaf.tolist(),           
               'max_depth': max_depth.tolist()}
    
    # Exhaustive search on full grid
    if dictionary['hp_search_type'] == 'exhaustive': search_hp_grid = [*traverse(hp_grid)]
    # Search on random subset of grid
    elif dictionary['hp_search_type'] == 'rdm': search_hp_grid = random.sample([*traverse(hp_grid)], dictionary['num_hp_set_candidates'])
    if dictionary['verbosity'] > 1: print('Grid search through ', len(search_hp_grid), ' hyperparameter combinations')
    
    # Initialize 
    mean_score = []
    if 'bin' in tgn_:
        mean_bin_classi_score = []

    # Loop over all selected hyperparameter combis 
    for i, hp_set in zip(np.arange(len(search_hp_grid)), search_hp_grid):   
        ## Initialize model
        if 'bin' in tgn_:                 
            rf_ = RandomForestClassifier(n_estimators = hp_set['n_estimators'], 
                                         max_depth = hp_set['max_depth'], 
                                         min_samples_split = 2 * hp_set['min_samples_leaf'], 
                                         min_samples_leaf = hp_set['min_samples_leaf'], 
                                         bootstrap = True, 
                                         random_state = seed, 
                                         class_weight = 'balanced',
                                         criterion = 'gini', 
                                         n_jobs = dictionary['n_cores'])
        elif 'bin' not in tgn_:
            rf_ = RandomForestRegressor(n_estimators = hp_set['n_estimators'], 
                                        max_depth = hp_set['max_depth'], 
                                        min_samples_split = 2 * hp_set['min_samples_leaf'], 
                                        min_samples_leaf = hp_set['min_samples_leaf'], 
                                        bootstrap = True, 
                                        random_state = seed, 
                                        criterion = 'mse', 
                                        n_jobs = dictionary['n_cores'])
        scores = {}
        if 'bin' in tgn_:
            bin_classi_scores = []
            new_thresholds = np.linspace(0., 1., 299)
            inner_splits = list(X_train_.keys())
            dims = ('inner_split', 'threshold')

        for key in X_train_:  
            ## Train model
            rf_.fit(X_train_[key], y_train_[key])
            ## Predict
            if 'bin' not in tgn_: 
                pred_vali_ = rf_.predict(X_vali_[key])
            elif 'bin' in tgn_:
                pred_vali_ = rf_.predict_proba(X_vali_[key])[:,1]
                ### Calculate ROC curve
                fpr, tpr, thresholds = roc_curve(y_vali_[key], pred_vali_)                
                ### Make dataframe with index thresholds
                roc_auc_df = pd.DataFrame({'TPR': tpr, 'FPR': fpr}, index = thresholds)
                ### Reverse index
                roc_auc_df = roc_auc_df.reindex(index = roc_auc_df.index[::-1])
                ### Interpolate thresholds to universal linspace, converst to dataset & expand dimensions
                roc_auc_df_interp = interpolate(roc_auc_df, new_thresholds).to_xarray().expand_dims('inner_split')                
                ### G-Mean lines for each threshold
                gmean = np.sqrt(roc_auc_df_interp.TPR * (1 - roc_auc_df_interp.FPR))
                ds = xr.Dataset(data_vars = {'TPR': (dims, roc_auc_df_interp.TPR),'FPR': (dims, roc_auc_df_interp.FPR), 'G-Mean': (dims, gmean)},
                                coords = {'inner_split': [key], 'threshold': new_thresholds})
                bin_classi_scores.append(ds)                
            ## Compute score used to optimize the hyperparameters (RMSE or ROC AUC)
            scores[key] = compute_score(optimize_metric_, y_vali_[key], pred_vali_)
                
        ## Take means 
        mean_score.append(np.mean(list(scores.values())))
        if 'bin' in tgn_:
            mean_bin_classi_score.append(xr.concat(bin_classi_scores, dim = 'inner_split').mean(dim = 'inner_split').expand_dims('hp_set').assign_coords(hp_set = [i]))
        ## Show    
        if dictionary['verbosity'] > 3: print(hp_set, 'yield ', optimize_metric_, ' {:.3f}'.format(mean_score[i]))  
    
    ## Compare to previous scores & choose the best score to optimize the hyperparameters
    if optimize_metric_ == 'RMSE': 
        score_max = np.min(mean_score)
        score_max_ix = np.argmin(mean_score)
    else: 
        score_max = np.max(mean_score)
        score_max_ix = np.argmax(mean_score)
    best_hp_set = search_hp_grid[score_max_ix]
        
    ## Choose best probability threshold
    if 'bin' in tgn_:
        ### Select binary classi scores corresponding to best hyperparameters
        best_mean_bin_classi_score = xr.concat(mean_bin_classi_score, dim = 'hp_set').sel(hp_set = score_max_ix)       
        ### Maximize geometric mean to find maximum classification score
        bin_classi_score_max = best_mean_bin_classi_score['G-Mean'].max()
        ### Find corresponding best probability threshold
        best_th_ix = int(best_mean_bin_classi_score['G-Mean'].argmax())        
        best_th = new_thresholds[best_th_ix]
        ### Plot ROC AUC curve and mark best threshold
        if dictionary['verbosity'] > 1: plot_roc_curve(best_mean_bin_classi_score.to_dataframe()['FPR'], best_mean_bin_classi_score.to_dataframe()['TPR'], best_th_ix, 'vali', tgn_, 'RFC', _lead_time_, outer_split_num_ = outer_split_num___) 
    
    # Best hyperparameters
    ## Print the best set of hyperparameters
    if dictionary['verbosity'] > 1: print('The best hyperparameter set: ', best_hp_set, 'yields ', optimize_metric_, ' {:.3f}'.format(score_max))  
    ## Initialize model with best hyperparameters & train it with the full training set 
    ### Initialize model
    if 'bin' in tgn_:                 
        rf_ = RandomForestClassifier(n_estimators = best_hp_set['n_estimators'], 
                                     max_depth = best_hp_set['max_depth'], 
                                     min_samples_split = 2 * best_hp_set['min_samples_leaf'], 
                                     min_samples_leaf = best_hp_set['min_samples_leaf'], 
                                     bootstrap = True, 
                                     random_state = seed, 
                                     class_weight = 'balanced',
                                     criterion = 'gini', 
                                     n_jobs = dictionary['n_cores'])
        
    elif 'bin' not in tgn_:
        rf_ = RandomForestRegressor(n_estimators = best_hp_set['n_estimators'], 
                                    max_depth = best_hp_set['max_depth'], 
                                    min_samples_split = 2 * best_hp_set['min_samples_leaf'], 
                                    min_samples_leaf = best_hp_set['min_samples_leaf'], 
                                    bootstrap = True, 
                                    random_state = seed, 
                                    criterion = 'mse', 
                                    n_jobs = dictionary['n_cores'])
        best_th = None
            
    ### Train with full_train
    rf_.fit(X_train_full_, y_train_full_)
    ## Make dataset with best hyperparameters
    if 'bin' in tgn_: forecast_names = ['RFC_' + balance_type_]
    elif 'bin' not in tgn_: forecast_names = ['RFR']
    ### Build dataset
    best_hp_set_list = list(best_hp_set.values())
    best_hp_names = ['best_' + x for x in hp_grid.keys()]
    best_hp_set_list.append(best_th); best_hp_names.append('best_threshold')
    best_rf_hyperparameters_dset_ = xr.Dataset({'value': (('forecast', 'best_hyperparameter', 'lead_time'), np.reshape(best_hp_set_list, (1, len(best_hp_names), 1)))},
                                               coords = {'forecast': forecast_names, 'best_hyperparameter': best_hp_names, 'lead_time': [_lead_time_]})
    
    end = time.time()
    print('Time needed for hyperparameter optimization with ', dictionary['num_hp_set_candidates'], ' candidates: ', (end - start)/60, ' min')
        
        
    return rf_, best_rf_hyperparameters_dset_


# In[10]:


def quick_choose_random_forest_hyperparameters(X_train_full_, y_train_full_, optimize_metric_, tgn_, _lead_time_, balance_type_, outer_split_num_ = None):
    
    """
    inputs
    ------
    X_train_full_        pd.Dataframe : matrix of attributes (time x lagged features) for final training (after the hyperparameters are fixed)
    y_train_full_           pd.Series : array containing the target variable (time) for final training (after the hyperparameters are fixed)
    optimize_metric_              str : metric to optimize: Regression ('Corr', 'RMSE'), Classification ('ROC AUC', 'PR AUC')
    tgn_                          str : name of target variable
    _lead_time_                   int : lead time of prediction in weeks
    balance_type_                 str : name of balance type 'undersampling', 'oversampling' or 'none'
    " outer_split_num_              int : counter of outer split for nested CV "

    outputs
    -------
    rf_                      function : initialized random forest model
    best_th                     float : best threshold for the binarization of heat wave probability 
    
    """
    
    if dictionary['verbosity'] > 1: print('** Quick RF hyperparameter choice **')
    
    # Initialize
    seed = 7
    
    # Load best hyperparameters from last iteration of the optimize_random_forest_hyperparameters function
    ## Specifiy file to read
    path = dictionary['path_hyperparam']
    if outer_split_num_ is not None:
        file_name = path + 'best_rf_hyperparam_' + balance_type_ + '_' + tgn_ + '_lead_time_' + str(_lead_time_) + '_outer_split_' + str(outer_split_num_) + '.nc'
    else:
        file_name = path + 'best_rf_hyperparam_' + balance_type_ + '_' + tgn_ + '_lead_time_' + str(_lead_time_) + '.nc'
    ## Read
    hyperparam_dset = xr.open_dataset(file_name)
    ## Extract hyperparameters
    n_estimators_ = int(hyperparam_dset.sel(best_hyperparameter = 'best_n_estimators').value)
    max_depth_float = float(hyperparam_dset.sel(best_hyperparameter = 'best_max_depth').value)
    if not np.isnan(max_depth_float): max_depth_ = int(max_depth_float)
    else: max_depth_ = None
    min_samples_leaf_ = int(hyperparam_dset.sel(best_hyperparameter = 'best_min_samples_leaf').value)
    min_samples_split_ = 2 * min_samples_leaf_
    ## Print hyperparameters
    if dictionary['verbosity'] > 1: print('(n_estimators, max_depth, min_samples_split, min_samples_leaf) : (', n_estimators_, ', ', max_depth_, ', ', min_samples_split_, ', ', min_samples_leaf_, ')')        
    ## Extract best threshold 
    if 'bin' in tgn_:
        best_th = float(hyperparam_dset.sel(best_hyperparameter = 'best_threshold').value)
        if dictionary['verbosity'] > 1: print('Best threshold for heat wave probability binarization: ', best_th)
    
    # Initialize & train model
    ## Initialize with best hyperparameters
    if 'bin' in tgn_:                 
        rf_ = RandomForestClassifier(n_estimators = n_estimators_, 
                                     max_depth = max_depth_, 
                                     min_samples_split = min_samples_split_, 
                                     min_samples_leaf = min_samples_leaf_, 
                                     bootstrap = True, 
                                     random_state = seed, 
                                     class_weight = 'balanced',
                                     criterion = 'gini', 
                                     n_jobs = dictionary['n_cores'])
    elif 'bin' not in tgn_:
        rf_ = RandomForestRegressor(n_estimators = n_estimators_, 
                                    max_depth = max_depth_, 
                                    min_samples_split = min_samples_split_, 
                                    min_samples_leaf = min_samples_leaf_, 
                                    bootstrap = True, 
                                    random_state = seed, 
                                    criterion = 'mse', 
                                    n_jobs = dictionary['n_cores'])
    ## Train model
    rf_.fit(X_train_full_, y_train_full_)
                
    if 'bin' in tgn_: return rf_, best_th
    elif 'bin' not in tgn_: return rf_, None


# In[11]:


def optimize_ridge_hyperparameters(X_train_, y_train_, X_vali_, y_vali_, tgn_):
    
    
    """
    inputs
    ------
    X_train_             pd.Dataframe : matrix of attributes (time x lagged features) for training (to select the hyperparameters)
    y_train_                pd.Series : array containing the target variable (time) for training (to select the hyperparameters)
    X_vali_              pd.Dataframe : matrix of attributes (time x lagged features) for validating
    y_vali_                 pd.Series : array containing the target variable (time) for validating
    tgn_                          str : name of target variable

    outputs
    -------
    model                                Trained Ridge Regression model
    
    """
    
    if dictionary['verbosity'] > 1: print('** Ridge hyperparameter optimization **')
    
    # Train Ridge that optimizes a given score
    ## Initialize score_max with its minumum
    if 'bin' not in tgn_ and dictionary['metric_regr'] == 'RMSE': score_max = 1000.
    else: score_max = 0.
    ## Iterate over all hyperparameters and choose the best set    
    for alpha in np.round(np.arange(0., 1.05, 0.05), decimals = 2):
        ### Initialize model
        if 'bin' in tgn_:
            model = RidgeClassifierwithProba(alpha = alpha, normalize = False, max_iter = 1000)
        elif 'bin' not in tgn_:
            model = Ridge(alpha = alpha, normalize = False, max_iter = 1000)
                  
        score_new = {}
        for key in X_train_:            
            ### Train model
            model.fit(X_train_[key], y_train_[key])
            ### Predict
            if 'bin' in tgn_: pred_vali_ = model.predict_proba(X_vali_[key])[:,1]
            else: pred_vali_ = model.predict(X_vali_[key])
            ### Compute score
            if 'bin' in tgn_: score_new[key] = compute_score(dictionary['metric_classi'], y_vali_[key], pred_vali_)
            else: score_new[key] = compute_score(dictionary['metric_regr'], y_vali_[key], pred_vali_)
            print('Score for ', key, ' is: ', score_new[key])
        score_new = np.mean(list(score_new.values()))      
        print('Mean is: ', score_new) 
        if dictionary['verbosity'] > 3: 
            if 'bin' in tgn_: print('alpha: ', alpha, 'has ', dictionary['metric_classi'], ' score: ', score_new)
            else: print('alpha: ', alpha, 'has ', dictionary['metric_regr'], ' score: ', score_new)

        ### Compare to previous scores & choose the best one
        if 'bin' not in tgn_ and dictionary['metric_regr'] == 'RMSE':
            if (score_new < score_max): 
                score_max = score_new
                best_alpha = alpha
        else:
            if (score_new > score_max): 
                score_max = score_new
                best_alpha = alpha                
            
    if dictionary['verbosity'] > 1: print('Best alpha for Ridge is: ', best_alpha) 
        
    # Initialize model with best hyperparameters & full training set 
    if 'bin' in tgn_:
        model = RidgeClassifierwithProba(alpha = best_alpha, normalize = False, max_iter = 1000)
    elif 'bin' not in tgn_:
        model = Ridge(alpha = best_alpha, normalize = False, max_iter = 1000)
        
    return model


# In[12]:


def optimize_elastic_net_hyperparameters(X_train_, y_train_, X_vali_, y_vali_, tgn_):
    
    """
    inputs
    ------
    X_train_             pd.Dataframe : matrix of attributes (time x lagged features) for training (to select the hyperparameters)
    y_train_                pd.Series : array containing the target variable (time) for training (to select the hyperparameters)   
    X_vali_              pd.Dataframe : matrix of attributes (time x lagged features) for validating
    y_vali_                 pd.Series : array containing the target variable (time) for validating
    tgn_                          str : name of target variable


    outputs
    -------
    model                                Trained ElasticNet Regression model
    
    """
    
    if dictionary['verbosity'] > 1: print('** Elastic Net hyperparameter optimization **')
    
    # Train the Elastic net that optimizes a given score
    ## Initialize score_max with its minumum
    if dictionary['metric_regr'] == 'Corr': score_max = 0.
    elif dictionary['metric_regr'] == 'RMSE': score_max = 1000.
    ## Iterate over all hyperparameters and choose the best set    
    for alpha in np.round(np.arange(0., 1.05, 0.05), decimals = 2):
        for l1_ratio in np.round(np.arange(0., 1.05, 0.05), decimals = 2):
            ### Initialize model
            model = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, normalize = True, max_iter = 1000, tol = 1e-4)
        
            score_new = {}
            for key in X_train_:          
                ### Train model
                model.fit(X_train_[key], y_train_[key])
                ### Predict
                if 'bin' in tgn_: pred_vali_ = model.predict_proba(X_vali_[key])[:,1]
                else: pred_vali_ = model.predict(X_vali_[key])
                ### Compute score
                score_new = compute_score(dictionary['metric_regr'], y_vali_[key], pred_vali_)
            print('Score for ', key, ' is: ', score_new[key])
            score_new = np.mean(list(score_new.values()))      
            print('Mean is: ', score_new)
            if dictionary['verbosity'] > 3: print('(alpha, l1_ratio): (', alpha, ', ', l1_ratio, '), have ', dictionary['metric_regr'], ' score: ', score_new)
            ### Compare to previous scores & choose the best one
            if dictionary['metric_regr'] == 'Corr':
                if (score_new > score_max): 
                    if dictionary['verbosity'] > 3: print('Better')
                    score_max = score_new
                    best_alpha = alpha
                    best_l1_ratio = l1_ratio
            elif dictionary['metric_regr'] == 'RMSE':
                if (score_new < score_max): 
                    if dictionary['verbosity'] > 3: print('Better')
                    score_max = score_new
                    best_alpha = alpha
                    best_l1_ratio = l1_ratio
            
    if dictionary['verbosity'] > 1: print('Best (alpha, l1_ratio) tuple for the Elastic Net (MLR) is: (', best_alpha, ', ', best_l1_ratio, ')')        
    # Initialize model with best hyperparameters & full training set 
    model = ElasticNet(alpha = best_alpha, l1_ratio = best_l1_ratio, normalize = True, max_iter = 1000, tol = 1e-4)
        
    return model


# In[13]:


def train_test_multilinear_regression(X_trf, y_trf, X_te, pred_names, feature_names, tgn, lt, 
                                      outer_split_num__ = None,
                                      X_trf_loo = None, y_trf_loo = None):
    
    """
    inputs
    ------
    X_trf                        xr.Dataset : predictors in full training time period " (with its leave one out ensemble (loo) as optional argument) "
    y_trf                      xr.DataArray : target in full training time period " (with its leave one out ensemble (loo) as optional argument) "
    X_te                         xr.Dataset : predictors in testing time period
    pred_names                         list : (of strings) names of all predictors ordered like in X_trf and X_te
    feature_names                      list : (of strings) names of all features ordered like in X_trf and X_te
    tgn                                 str : name of target
    lt                                  int : lead time of prediction in weeks
    " outer_split_num__                 int : counter of outer split for nested CV "

    
    outputs
    -------
    pred_te                       pd.Series : time series predicted by the MLR model for the testing time period " (with its leave one out ensemble (loo) as optional argument) "         
    pred_trf                      pd.Series : time series predicted by the MLR model for the full training time period " (with its leave one out ensemble (loo) as optional argument) "   
    
    """   
    
    if dictionary['verbosity'] > 0: print_model_name('Multilinear Regression')
    # Set outer split number to None for non-nested CV
    if dictionary['cv_type'] == 'none': outer_split_num__ = None
    
    ## Train linear regression model
    if dictionary['reg_regr'] is False:
        mlr = LinearRegression()
    elif dictionary['reg_regr'] is True:
        #if dictionary['verbosity'] > 1: print('************************************* OPTIMIZE ON VALIDATION DATA *****************************************')    
        #mlr = optimize_ridge_hyperparameters(X_tr, y_tr, X_v, y_v, tgn)
        #mlr = optimize_elastic_net_hyperparameters(X_tr, y_tr, X_v, y_v, tgn)
        mlr = Ridge(alpha = 1.0, normalize = False, max_iter = 1000)
        #mlr = ElasticNet(alpha = 1.0, l1_ratio = 0.5, normalize = False)
    # Fit with all data
    mlr.fit(X_trf, y_trf)
    ## Plot regression coefficients
    if dictionary['verbosity'] > 1: show_coeffs(mlr, 'MLR', pred_names, feature_names, tgn, lt, outer_split_num__)

    ## Predict
    if dictionary['verbosity'] > 1: print('******************************************* PREDICT TEST DATA *********************************************')
    pred_te = mlr.predict(X_te)
    if dictionary['cv_type'] == 'none':
        pred_te_ensemble = np.stack([mlr.fit(X_trf_loo[key], y_trf_loo[key]).predict(X_te) for key in X_trf_loo])
        import matplotlib.pyplot as plt
        plt.figure(figsize = (20,5))
        for i in np.arange(len(pred_te_ensemble)):       
            plt.plot(np.arange(len(pred_te_ensemble[i])), pred_te_ensemble[i], linewidth = 1)
        plt.plot(np.arange(len(pred_te)), pred_te, linewidth = 3, color = 'black')
        plt.show()
        plt.close()
    ### Save time series
    save_time_series(pred_te, tgn, 'MLR', 'test', lt, outer_split_num__)
    if dictionary['cv_type'] == 'none':
        save_time_series(pred_te_ensemble, tgn, 'MLR_ensemble', 'test', lt, outer_split_num__)
    
    if dictionary['verbosity'] > 1: print('**************************************** PREDICT TRAIN FULL DATA ******************************************')
    # Fit with all data again
    mlr.fit(X_trf, y_trf)
    pred_trf = mlr.predict(X_trf)
    if dictionary['cv_type'] == 'none':
        pred_trf_ensemble = np.stack([mlr.fit(X_trf_loo[key], y_trf_loo[key]).predict(X_trf) for key in X_trf_loo])
    ### Save time series
    save_time_series(pred_trf, tgn, 'MLR', 'train_full', lt, outer_split_num__)
    if dictionary['cv_type'] == 'none':
        save_time_series(pred_trf_ensemble, tgn, 'MLR_ensemble', 'train_full', lt, outer_split_num__)
    
    if dictionary['cv_type'] == 'none': return pred_te, pred_te_ensemble, pred_trf, pred_trf_ensemble
    elif dictionary['cv_type'] == 'nested': return pred_te, pred_trf
    


# In[14]:


def train_test_random_forest_regressor(X_tr, y_tr, X_trf, y_trf, X_v, y_v, X_te, pred_names, feature_names, tgn, lt, bal, outer_split_num__ = None):

    """
    inputs
    ------
    X_tr                         xr.Dataset : predictors in training time period
    y_tr                       xr.DataArray : target in training time period
    X_trf                        xr.Dataset : predictors in full training time period
    y_trf                      xr.DataArray : target in full training time period
    X_v                          xr.Dataset : predictors in validation time period
    y_v                        xr.DataArray : target in validation time period
    X_te                         xr.Dataset : predictors in testing time period
    pred_names                         list : (of strings) names of all predictors ordered like in X_trf and X_te
    feature_names                      list : (of strings) names of all features ordered like in X_trf and X_te
    tgn                                 str : name of target
    lt                                  int : lead time of prediction in weeks
    bal                                 str : balance type ('undersampling' or 'oversampling')
    " outer_split_num__                 int : counter of outer split for nested CV "

    
    outputs
    -------
    pred_te                       pd.Series : time series predicted by the MLR model for the testing time period                 
    pred_trf                      pd.Series : time series predicted by the MLR model for the full training time period  
    best_hp                      xr.Dataset : hyperparameters for the Random Forest that optimize the metric_regr specified in the dictionary
    
    """   
    
    if dictionary['verbosity'] > 0: print_model_name('Random Forest Regressor')
    
    # Set outer split number to None for non-nested CV
    if dictionary['cv_type'] == 'none': outer_split_num__ = None
    
    ## Train Random Forest Regressor model
    if dictionary['optimize_rf_hyperparam'] is False:
        rfr, trivial_th = quick_choose_random_forest_hyperparameters(X_trf, y_trf, dictionary['metric_regr'], tgn, lt, bal, outer_split_num_ = outer_split_num__) 
    elif dictionary['optimize_rf_hyperparam'] is True:
        if dictionary['verbosity'] > 1: print('************************************** OPTIMIZE ON VALIDATION DATA ****************************************')
        rfr, best_hp = optimize_random_forest_hyperparameters(X_tr, y_tr, X_trf, y_trf, X_v, y_v, dictionary['metric_regr'], tgn, lt, bal, outer_split_num___ = outer_split_num__)

    if dictionary['verbosity'] > 1: 
        print('Depth of trees :')
        print('Max: ', max([estimator.get_depth() for estimator in rfr.estimators_]))
        print('Min: ', min([estimator.get_depth() for estimator in rfr.estimators_]))
    ## Show feature importance
    if dictionary['verbosity'] > 1: show_rf_imp(rfr, pred_names, feature_names, tgn, lt, outer_split_num__)

    ## Predict
    if dictionary['verbosity'] > 1: print('******************************************* PREDICT TEST DATA *********************************************')
    pred_te = rfr.predict(X_te)
    pred_te_ensemble = np.stack([pr.predict(X_te) for pr in rfr.estimators_])
    ### Save time series
    save_time_series(pred_te, tgn, 'RFR', 'test', lt, outer_split_num__)
    save_time_series(pred_te_ensemble, tgn, 'RFR_ensemble', 'test', lt, outer_split_num__)
    
    if dictionary['verbosity'] > 1: print('***************************************** PREDICT TRAIN FULL DATA ******************************************')
    pred_trf = rfr.predict(X_trf)
    pred_trf_ensemble = np.stack([pr.predict(X_trf) for pr in rfr.estimators_])
    ### Save time series
    save_time_series(pred_trf, tgn, 'RFR', 'train_full', lt, outer_split_num__)
    save_time_series(pred_trf_ensemble, tgn, 'RFR_ensemble', 'train_full', lt, outer_split_num__)
    
    if dictionary['optimize_rf_hyperparam'] is True: return pred_te, pred_te_ensemble, pred_trf, pred_trf_ensemble, best_hp
    else: return pred_te, pred_te_ensemble, pred_trf, pred_trf_ensemble, None


# In[6]:


def train_test_ridge_classifier(X_tr, y_tr, X_trf, y_trf, X_trfb, y_trfb, X_v, y_v, X_te, y_te, pred_names, feature_names, tgn, lt, bal, bal_sign, 
                                outer_split_num__ = None,
                                X_trf_loo = None, y_trf_loo = None,
                                X_trfb_loo = None, y_trfb_loo = None):

    """
    inputs
    -------
    X_tr                         xr.Dataset : predictors in training time period
    y_tr                       xr.DataArray : target in training time period
    X_trf                        xr.Dataset : predictors in full training time period " (with its leave one out ensemble (loo) as optional argument) "
    y_trf                      xr.DataArray : target in full training time period " (with its leave one out ensemble (loo) as optional argument) "
    X_trfb                       xr.Dataset : balanced predictors in full training time period " (with its leave one out ensemble (loo) as optional argument) "
    y_trfb                     xr.DataArray : balanced target in full training time period " (with its leave one out ensemble (loo) as optional argument) "
    X_v                          xr.Dataset : predictors in validation time period
    y_v                        xr.DataArray : target in validation time period
    X_te                         xr.Dataset : predictors in testing time period
    y_te                       xr.DataArray : target in testing time period
    pred_names                         list : (of strings) names of all predictors ordered like in X_trf and X_te
    feature_names                      list : (of strings) names of all features ordered like in X_trf and X_te
    tgn                                 str : name of target
    lt                                  int : lead time of prediction in weeks
    bal                                 str : balance type ('undersampling' or 'oversampling')
    bal_sign                            str : '+' (oversampled) or '-' (undersampled)
    " outer_split_num__                 int : counter of outer split for nested CV "
    " inner_split_num__                 int : counter of inner split for nested CV "

    
    outputs
    -------
    pred_te                       pd.Series : binary time series predicted by the MLR model for the testing time period                 
    pred_proba_te                 pd.Series : probability time series predicted by the MLR model for the testing time period 
    pred_trf                      pd.Series : binary time series predicted by the MLR model for the full training time period 
    pred_proba_trf                pd.Series : probability time series predicted by the MLR model for the full training time period 
    
    """   
        
    # Set outer split number to None for non-nested CV
    if dictionary['cv_type'] == 'none': outer_split_num__ = None
    
    ## Choose model
    if dictionary['verbosity'] > 0: print_model_name('Ridge Classifier')
    #if dictionary['verbosity'] > 1: print('************************************** OPTIMIZE ON VALIDATION DATA ****************************************')        
    #rc = optimize_ridge_hyperparameters(X_tr, y_tr, X_v, y_v, tgn)
    rc = RidgeClassifierwithProba(alpha = 1.0, normalize = False, max_iter = 1000)  
    
    ## Find best threshold to binarize probability prediction
    if dictionary['verbosity'] > 1: print('**************************** CHOOSE PROBABILITY THRESHOLD ****************************') 
    if bal != 'none': best_th_rc = choose_best_proba_threshold_classi(rc, 'RC', X_tr, y_tr, X_v, y_v, tgn, lt, outer_split_num__ = outer_split_num__)
    else: best_th_rc = choose_best_proba_threshold_classi(rc, 'RC', X_tr, y_tr, X_v, y_v, tgn, lt, outer_split_num__ = outer_split_num__)
   
    ## Train model
    if bal != 'none': rc.fit(X_trfb, y_trfb)
    else: rc.fit(X_trf, y_trf)
    ## Plot coefficients
    if dictionary['verbosity'] > 1: show_coeffs(rc, 'RC', pred_names, feature_names, tgn, lt, outer_split_num__)

    if dictionary['verbosity'] > 1: print('******************************************* PREDICT TEST DATA *********************************************')
    pred_te, pred_proba_te = predict_classi(rc, X_te, best_th_rc)
    if dictionary['cv_type'] == 'none':
        if bal != 'none': 
            pred_te_ensemble_both = np.stack([predict_classi(rc.fit(X_trfb_loo[key], y_trfb_loo[key]), X_te, best_th_rc) for key in X_trfb_loo])
        else: 
            pred_te_ensemble_both = np.stack([predict_classi(rc.fit(X_trf_loo[key], y_trf_loo[key]), X_te, best_th_rc) for key in X_trf_loo])
        pred_te_ensemble = pred_te_ensemble_both[:,0]
        pred_proba_te_ensemble = pred_te_ensemble_both[:,1]    
    ### Save time series
    model_name = 'RC' + bal_sign
    save_time_series(pred_te, tgn, model_name, 'test', lt, outer_split_num__) 
    save_time_series(pred_proba_te, tgn, model_name, 'test_proba', lt, outer_split_num__) 
    if dictionary['cv_type'] == 'none':
        save_time_series(pred_te_ensemble, tgn, model_name + '_ensemble', 'test', lt, outer_split_num__) 
        save_time_series(pred_proba_te_ensemble, tgn, model_name + '_ensemble', 'test_proba', lt, outer_split_num__) 
    
    ### Plot curves
    if dictionary['metric_classi'] == 'ROC AUC':
        fpr, tpr, thresholds = roc_curve(y_te, pred_proba_te)
        plot_roc_curve(fpr, tpr, None, 'test', tgn, 'RC', lt, outer_split_num_ = outer_split_num__)       
    elif dictionary['metric_classi'] == 'PR AUC':
        precision, recall, thresholds = precision_recall_curve(y_te, pred_proba_te)
        plot_pr_curve(precision, recall, None, 'test', tgn, 'RC', lt, outer_split_num_ = outer_split_num__) 
        
    if dictionary['verbosity'] > 1: print('***************************************** PREDICT TRAIN FULL DATA ******************************************')
    ## Train model again
    if bal != 'none': rc.fit(X_trfb, y_trfb)
    else: rc.fit(X_trf, y_trf)
    pred_trf, pred_proba_trf = predict_classi(rc, X_trf, best_th_rc)
    if dictionary['cv_type'] == 'none':
        if bal != 'none': 
            pred_trf_ensemble_both = np.stack([predict_classi(rc.fit(X_trf_loo[key], y_trf_loo[key]), X_trf, best_th_rc) for key in X_trf_loo])
        else: 
            pred_trf_ensemble_both = np.stack([predict_classi(rc.fit(X_trfb_loo[key], y_trfb_loo[key]), X_trf, best_th_rc) for key in X_trfb_loo])
        pred_trf_ensemble = pred_trf_ensemble_both[:,0]
        pred_proba_trf_ensemble = pred_trf_ensemble_both[:,1]
    ### Save time series
    save_time_series(pred_trf, tgn, model_name, 'train_full', lt, outer_split_num__) 
    save_time_series(pred_proba_trf, tgn, model_name, 'train_full_proba', lt, outer_split_num__) 
    if dictionary['cv_type'] == 'none':
        save_time_series(pred_trf_ensemble, tgn, model_name + '_ensemble', 'train_full', lt, outer_split_num__) 
        save_time_series(pred_proba_trf_ensemble, tgn, model_name + '_ensemble', 'train_full_proba', lt, outer_split_num__) 
    
    if dictionary['cv_type'] == 'none': return pred_te, pred_te_ensemble, pred_proba_te, pred_proba_te_ensemble, pred_trf, pred_trf_ensemble, pred_proba_trf, pred_proba_trf_ensemble
    elif dictionary['cv_type'] == 'nested': return pred_te, pred_proba_te, pred_trf, pred_proba_trf
            


# In[16]:


def train_test_random_forest_classifier(X_tr, y_tr, X_trf, y_trf, X_trfb, y_trfb, X_v, y_v, X_te, y_te, pred_names, feature_names, tgn, lt, bal, bal_sign, outer_split_num__ = None):
    
    """
    inputs
    -------
    X_tr                         xr.Dataset : predictors in training time period
    y_tr                       xr.DataArray : target in training time period
    X_trf                        xr.Dataset : predictors in full training time period
    y_trf                      xr.DataArray : target in full training time period
    X_trfb                       xr.Dataset : balanced predictors in full training time period
    y_trfb                     xr.DataArray : balanced target in full training time period
    X_v                          xr.Dataset : predictors in validation time period
    y_v                        xr.DataArray : target in validation time period
    X_te                         xr.Dataset : predictors in testing time period
    y_te                       xr.DataArray : target in testing time period
    pred_names                         list : (of strings) names of all predictors ordered like in X_trf and X_te
    feature_names                      list : (of strings) names of all features ordered like in X_trf and X_te
    tgn                                 str : name of target
    lt                                  int : lead time of prediction in weeks
    bal                                 str : balance type ('undersampling' or 'oversampling')
    bal_sign                            str : '+' (oversampled) or '-' (undersampled)
    " outer_split_num__                 int : counter of outer split for nested CV "s

    
    outputs
    -------
    pred_te                       pd.Series : binary time series predicted by the MLR model for the testing time period                 
    pred_proba_te                 pd.Series : probability time series predicted by the MLR model for the testing time period 
    pred_trf                      pd.Series : binary time series predicted by the MLR model for the full training time period 
    pred_proba_trf                pd.Series : probability time series predicted by the MLR model for the full training time period 
    best_hp                      xr.Dataset : hyperparameters for the Random Forest that optimize the metric_classi specified in the dictionary
    
    """  

    # Set outer split number to None for non-nested CV
    if dictionary['cv_type'] == 'none': outer_split_num__ = None
    
    ## Train Classification Random Forest model 
    if dictionary['verbosity'] > 0: print_model_name('Random Forest Classifier')
    if bal != 'none': 
        if dictionary['optimize_rf_hyperparam'] is False:
            rfc, best_rfc_th = quick_choose_random_forest_hyperparameters(X_trfb, y_trfb, dictionary['metric_classi'], tgn, lt, bal, outer_split_num_ = outer_split_num__)
        elif dictionary['optimize_rf_hyperparam'] is True:
            if dictionary['verbosity'] > 1: print('************************************** OPTIMIZE ON VALIDATION DATA ****************************************') 
            rfc, best_hp = optimize_random_forest_hyperparameters(X_tr, y_tr, X_trfb, y_trfb, X_v, y_v, dictionary['metric_classi'], tgn, lt, bal, outer_split_num___ = outer_split_num__)
            best_rfc_th = float(best_hp.sel(forecast = 'RFC_' + bal, best_hyperparameter = 'best_threshold', lead_time = lt).value)
            print('best_rfc_th: ', best_rfc_th)

    else:
        if dictionary['optimize_rf_hyperparam'] is False:
            rfc, best_rfc_th = quick_choose_random_forest_hyperparameters(X_trf, y_trf, dictionary['metric_classi'], tgn, lt, bal, outer_split_num_ = outer_split_num__)
        elif dictionary['optimize_rf_hyperparam'] is True:
            if dictionary['verbosity'] > 1: print('************************************** OPTIMIZE ON VALIDATION DATA ****************************************') 
            rfc, best_hp = optimize_random_forest_hyperparameters(X_tr, y_tr, X_trf, y_trf, X_v, y_v, dictionary['metric_classi'], tgn, lt, bal, outer_split_num___ = outer_split_num__)
            best_rfc_th = float(best_hp.sel(forecast = 'RFC_' + bal, best_hyperparameter = 'best_threshold', lead_time = lt).value)
            print('best_rfc_th: ', best_rfc_th)
        
    ## Print model characteristics    
    if dictionary['verbosity'] > 1: print('Depth of trees :')
    if dictionary['verbosity'] > 1: print('Max: ', max([estimator.get_depth() for estimator in rfc.estimators_]))
    if dictionary['verbosity'] > 1: print('Min: ', min([estimator.get_depth() for estimator in rfc.estimators_]))
    ## Show feature importance
    if dictionary['verbosity'] > 1: show_rf_imp(rfc, pred_names, feature_names, tgn, lt, outer_split_num__)

    if dictionary['verbosity'] > 1: print('******************************************* PREDICT TEST DATA *********************************************')
    pred_te, pred_proba_te = predict_classi(rfc, X_te, best_rfc_th)
    pred_te_ensemble_both = np.stack([predict_classi(pr, X_te, best_rfc_th) for pr in rfc.estimators_])
    pred_te_ensemble = pred_te_ensemble_both[:,0]
    pred_proba_te_ensemble = pred_te_ensemble_both[:,1]
    ### Save time series
    model_name = 'RFC' + bal_sign
    save_time_series(pred_te, tgn, model_name, 'test', lt, outer_split_num__) 
    save_time_series(pred_proba_te, tgn, model_name, 'test_proba', lt, outer_split_num__) 
    save_time_series(pred_te_ensemble, tgn, model_name + '_ensemble', 'test', lt, outer_split_num__) 
    save_time_series(pred_proba_te_ensemble, tgn, model_name + '_ensemble', 'test_proba', lt, outer_split_num__) 
    if dictionary['metric_classi'] == 'ROC AUC':
        fpr, tpr, thresholds = roc_curve(y_te, pred_proba_te)
        plot_roc_curve(fpr, tpr, None, 'test', tgn, 'RFC', lt, outer_split_num_ = outer_split_num__) 
    elif dictionary['metric_classi'] == 'PR AUC':
        precision, recall, thresholds = precision_recall_curve(y_te, pred_proba_te)
        plot_pr_curve(precision, recall, None, 'test', tgn, 'RFC', lt, outer_split_num_ = outer_split_num__)    
        
    if dictionary['verbosity'] > 1: print('***************************************** PREDICT TRAIN FULL DATA ******************************************')
    pred_trf, pred_proba_trf = predict_classi(rfc, X_trf, best_rfc_th)
    pred_trf_ensemble_both = np.stack([predict_classi(pr, X_trf, best_rfc_th) for pr in rfc.estimators_])
    pred_trf_ensemble = pred_trf_ensemble_both[:,0]
    pred_proba_trf_ensemble = pred_trf_ensemble_both[:,1]
    ### Save time series
    save_time_series(pred_trf, tgn, model_name, 'train_full', lt, outer_split_num__) 
    save_time_series(pred_proba_trf, tgn, model_name, 'train_full_proba', lt, outer_split_num__) 
    save_time_series(pred_trf_ensemble, tgn, model_name + '_ensemble', 'train_full', lt, outer_split_num__) 
    save_time_series(pred_proba_trf_ensemble, tgn, model_name + '_ensemble', 'train_full_proba', lt, outer_split_num__) 
    
    if dictionary['optimize_rf_hyperparam'] is True: return pred_te, pred_te_ensemble, pred_proba_te, pred_proba_te_ensemble, pred_trf, pred_trf_ensemble, pred_proba_trf, pred_proba_trf_ensemble, best_hp
    else: return pred_te, pred_te_ensemble, pred_proba_te, pred_proba_te_ensemble, pred_trf, pred_trf_ensemble, pred_proba_trf, pred_proba_trf_ensemble, None


# In[18]:


def pred_algorithm(X_train_, y_train_, 
                   X_train_full_, y_train_full_, 
                   X_train_full_bal_, y_train_full_bal_, 
                   X_vali_, y_vali_, 
                   X_test_, y_test_, 
                   persist_train_full_, clim_forecast_train_full_, 
                   persist_test_, clim_forecast_test_, ecmwf_forecast_test_, 
                   start_date_test_, predictor_names_, feature_names_, tg_name_, balance_, balance_sign_, _lead_time_, 
                   outer_split_num = None,
                   X_train_full_loo = None, y_train_full_loo = None, 
                   X_train_full_loo_bal = None, y_train_full_loo_bal = None):
        
    """
    inputs
    ------
    X_train_                             xr.Dataset : balanced (in the case of classification only) predictors in training time period
    y_train_                           xr.DataArray : balanced (in the case of classification only) target in training time period
    X_train_full_                        xr.Dataset : predictors in full training time period " (with its leave one out ensemble (loo) as optional argument) "
    y_train_full_                      xr.DataArray : target in full training time period " (with its leave one out ensemble (loo) as optional argument) "
    X_train_full_bal_                    xr.Dataset : balanced predictors in full training time period " (with its leave one out ensemble (loo) as optional argument) "
    y_train_full_bal_                  xr.DataArray : balanced target in full training time period " (with its leave one out ensemble (loo) as optional argument) "
    X_vali_                              xr.Dataset : predictors in validation time period
    y_vali_                            xr.DataArray : target in validation time period
    X_test_                              xr.Dataset : predictors in testing time period
    y_test_                            xr.DataArray : target in testing time period
    persist_train_full_                xr.DataArray : persistence forecast of target for full training time period
    clim_forecast_train_full_          xr.DataArray : climatology forecast of target for full training time period
    persist_test_                      xr.DataArray : persistence forecast of target for testing time period
    clim_forecast_test_                xr.DataArray : climatology forecast of target for testing time period
    ecmwf_forecast_test_               xr.DataArray : ECMWF forecast of target for testing time period
    start_date_test_                            str : start date of test time period
    predictor_names_                           list : (of strings) names of all predictors ordered like in X_trf and X_te
    feature_names_                             list : (of strings) names of all predictors ordered like in X_trf and X_te
    tg_name_                                    str : name of target
    balance_                                    str : balance type ('undersampling' or 'oversampling')
    balance_sign_                               str : '+' (oversampled) or '-' (undersampled)
    _lead_time_                                 int : lead time of prediction in weeks
    " outer_split_num                             int : counter of outer splits "


    outputs
    -------
    None. Plots ROC AUC and Precision-Recall curves
          Saves best hyperparameters to file if hyperparameter optimization is True
          Saves time series to file
          Plots metrics table
          Saves metrics to file
          Plots time series and lagged correlation plots for regression
    
    """
    
    # **Bias/variance trade off:**
    # 
    #     a) Bias: "The bias error is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting)."
    # 
    #     b) Variance: "The variance is an error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting)."
    # 
    # **Solutions:** 
    # 
    #     a) Smaller weights
    # 
    #     b) Optimum degree of polynomial (n) in case of non-linear (>1) or of depth for a tree (hyperparameterisation)
    # 
    # **Model choice:** Choose models at different end of the bias-variance trade off:
    # 
    #     1. Regression // 2. Classification
    # 
    #         1.1. Multilinear Regression // 2.1. Ridge Classifier (high bias, low variance)
    # 
    #         1.2. Random Forest Regressor // 2.2. Random Forest Classifier (low bias, high variance)
    
    # Set outer split number to None for the case of non-nested CV
    if dictionary['cv_type'] == 'none': outer_split_num = None
    # Set train_full_loo ensembles to None for the case of nested CV
    if dictionary['cv_type'] == 'nested':
        X_train_full_loo = None
        y_train_full_loo = None
        X_train_full_loo_bal = None
        y_train_full_loo_bal = None
    
    
    #Train models
    if dictionary['train_models'] is True: 

        ## 1. Regression
        # **For continuous variables: t2m**
        if 'bin' not in tg_name_:
            ### Quick linearity check with scatter diagrams   
            if dictionary['verbosity'] > 4: plot_scatter_diagrams(X_train_full_, y_train_full_, tg_name_)
            ### 1.1. Multilinear Regression (MLR)
            if dictionary['cv_type'] == 'none':
                pred_mlr_test, pred_mlr_test_ensemble, pred_mlr_train_full, pred_mlr_train_full_ensemble = train_test_multilinear_regression(X_train_full_, y_train_full_, X_test_, predictor_names_, feature_names_, tg_name_, _lead_time_,
                                                                                                                                            outer_split_num__ = outer_split_num,
                                                                                                                                            X_trf_loo = X_train_full_loo, y_trf_loo = y_train_full_loo)
            elif dictionary['cv_type'] == 'nested':
                pred_mlr_test, pred_mlr_train_full = train_test_multilinear_regression(X_train_full_, y_train_full_, X_test_, predictor_names_, feature_names_, tg_name_, _lead_time_,
                                                                                        outer_split_num__ = outer_split_num)
                                                                                                                            
            ### 1.2. Random Forest Regressor (RFR)
            pred_rfr_test, pred_rfr_test_ensemble, pred_rfr_train_full, pred_rfr_train_full_ensemble, best_hyperparam_rf = train_test_random_forest_regressor(X_train_, y_train_, 
                                                                                                                                                                X_train_full_, y_train_full_, 
                                                                                                                                                                X_vali_, y_vali_, 
                                                                                                                                                                X_test_, 
                                                                                                                                                                predictor_names_, feature_names_, tg_name_, _lead_time_, 
                                                                                                                                                                balance_, outer_split_num__ = outer_split_num)

        ## 2. Classification 
        # **For binary variables: hw_bin_2in7, hw_bin_1SD, hw_bin_15SD**
        elif 'bin' in tg_name_:
            ### 2.1. Ridge Classifier (RC)
            if dictionary['cv_type'] == 'none': 
                (pred_rc_test, pred_rc_test_ensemble, 
                 pred_proba_rc_test, pred_proba_rc_test_ensemble, 
                 pred_rc_train_full, pred_rc_train_full_ensemble, 
                 pred_proba_rc_train_full, pred_proba_rc_train_full_ensemble) = train_test_ridge_classifier(X_train_, y_train_, 
                                                                                                            X_train_full_, y_train_full_, 
                                                                                                            X_train_full_bal_, y_train_full_bal_, 
                                                                                                            X_vali_, y_vali_, 
                                                                                                            X_test_, y_test_, 
                                                                                                            predictor_names_, feature_names_, tg_name_, _lead_time_, 
                                                                                                            balance_, balance_sign_, outer_split_num__ = outer_split_num,
                                                                                                            X_trf_loo = X_train_full_loo, y_trf_loo = y_train_full_loo,
                                                                                                            X_trfb_loo = X_train_full_loo_bal, y_trfb_loo = y_train_full_loo_bal)
            elif dictionary['cv_type'] == 'nested':
                (pred_rc_test,
                 pred_proba_rc_test,
                 pred_rc_train_full, 
                 pred_proba_rc_train_full) = train_test_ridge_classifier(X_train_, y_train_, 
                                                                        X_train_full_, y_train_full_, 
                                                                        X_train_full_bal_, y_train_full_bal_, 
                                                                        X_vali_, y_vali_, 
                                                                        X_test_, y_test_, 
                                                                        predictor_names_, feature_names_, tg_name_, _lead_time_, 
                                                                        balance_, balance_sign_, outer_split_num__ = outer_split_num)
            ### 2.2. Random Forest Classifier (RFC)
            (pred_rfc_test, pred_rfc_test_ensemble, 
             pred_proba_rfc_test, pred_proba_rfc_test_ensemble, 
             pred_rfc_train_full, pred_rfc_train_full_ensemble, 
             pred_proba_rfc_train_full, pred_proba_rfc_train_full_ensemble, 
             best_hyperparam_rf) = train_test_random_forest_classifier(X_train_, y_train_, 
                                                                        X_train_full_, y_train_full_, 
                                                                        X_train_full_bal_, y_train_full_bal_, 
                                                                        X_vali_, y_vali_, 
                                                                        X_test_, y_test_, 
                                                                        predictor_names_, feature_names_, tg_name_, _lead_time_, 
                                                                        balance_, balance_sign_, outer_split_num__ = outer_split_num) 

        ## 3. Best hyperparameters
        # Save best hyperparameters for RF's 
        if dictionary['optimize_rf_hyperparam'] is True:            
            path = dictionary['path_hyperparam']
            if outer_split_num is not None:
                save_name = 'best_rf_hyperparam_' + balance_ + '_' + tg_name_ + '_lead_time_' + str(_lead_time_) + '_outer_split_' + str(outer_split_num) + '.nc'
            else:
                save_name = 'best_rf_hyperparam_' + balance_ + '_' + tg_name_ + '_lead_time_' + str(_lead_time_) + '.nc'
            save_to_file(best_hyperparam_rf, path, save_name, 'nc')

    else: 
        if 'bin' not in tg_name_:
            if dictionary['cv_type'] == 'none':
                (pred_mlr_test, pred_mlr_test_ensemble, pred_rfr_test, pred_rfr_test_ensemble, 
                pred_mlr_train_full, pred_mlr_train_full_ensemble, pred_rfr_train_full, pred_rfr_train_full_ensemble) = read_old_regr_ml_forecasts(tg_name_, _lead_time_, outer_split_num_ = outer_split_num)
            elif dictionary['cv_type'] == 'nested':
                (pred_mlr_test, pred_rfr_test, pred_rfr_test_ensemble, 
                pred_mlr_train_full, pred_rfr_train_full, pred_rfr_train_full_ensemble) = read_old_regr_ml_forecasts(tg_name_, _lead_time_, outer_split_num_ = outer_split_num)
        if 'bin' in tg_name_:
            if dictionary['cv_type'] == 'none':
                (pred_proba_rc_test, pred_proba_rc_test_ensemble, pred_proba_rfc_test, pred_proba_rfc_test_ensemble, 
                pred_rc_test, pred_rc_test_ensemble, pred_rfc_test, pred_rfc_test_ensemble, 
                pred_proba_rc_train_full, pred_proba_rc_train_full_ensemble, pred_proba_rfc_train_full, pred_proba_rfc_train_full_ensemble, 
                pred_rc_train_full, pred_rc_train_full_ensemble, pred_rfc_train_full, pred_rfc_train_full_ensemble) = read_old_classi_ml_forecasts(tg_name_, _lead_time_, balance_sign_, outer_split_num_ = outer_split_num)
            elif dictionary['cv_type'] == 'nested':
                (pred_proba_rc_test, pred_proba_rfc_test, pred_proba_rfc_test_ensemble, 
                pred_rc_test, pred_rfc_test, pred_rfc_test_ensemble, 
                pred_proba_rc_train_full, pred_proba_rfc_train_full,  pred_proba_rfc_train_full_ensemble,
                pred_rc_train_full, pred_rfc_train_full, pred_rfc_train_full_ensemble) = read_old_classi_ml_forecasts(tg_name_, _lead_time_, balance_sign_, outer_split_num_ = outer_split_num)


    if dictionary['cv_type'] == 'nested':
        pred_mlr_test_ensemble = None
        pred_mlr_train_full_ensemble = None
        pred_proba_rc_test_ensemble = None
        pred_rc_test_ensemble = None
        pred_proba_rc_train_full_ensemble = None
        pred_rc_train_full_ensemble = None
        
        
    ## 4. Metrics
    ### 9.1. Metrics regr
    if 'bin' not in tg_name_:
        # Print metrics table
        if dictionary['verbosity'] > 1: print('*********************************************** METRICS TEST DATA *****************************************')
        metrics_regr_test = build_metrics_regr(y_test_, pred_mlr_test, pred_rfr_test, pred_rfr_test_ensemble, 
                                               persist_test_, clim_forecast_test_, tg_name_, _lead_time_, 'test', 
                                               predictions_mlr_ensemble = pred_mlr_test_ensemble, ecmwf = ecmwf_forecast_test_, outer_split_num_ = outer_split_num)
        if dictionary['verbosity'] > 1: print('********************************************* METRICS TRAIN FULL DATA **************************************')
        metrics_regr_train = build_metrics_regr(y_train_full_, pred_mlr_train_full, pred_rfr_train_full, pred_rfr_train_full_ensemble, 
                                                persist_train_full_, clim_forecast_train_full_, tg_name_, _lead_time_, 'train', 
                                                predictions_mlr_ensemble = pred_mlr_train_full_ensemble, outer_split_num_ = outer_split_num)

        # Save metrics
        save_metrics(metrics_regr_test, 'regr', 'test', tg_name_, _lead_time_, balance_, outer_split_num)
        save_metrics(metrics_regr_train, 'regr', 'train', tg_name_, _lead_time_, balance_, outer_split_num)

    ### 4.2. Metrics classi
    if 'bin' in tg_name_:
        # Print metrics table
        if dictionary['verbosity'] > 1: print('*********************************************** METRICS TEST DATA *****************************************')
        proba_metrics_classi_test = build_metrics_proba_classi(y_test_, pred_proba_rc_test, pred_proba_rfc_test, pred_proba_rfc_test_ensemble, 
                                                               persist_test_, clim_forecast_test_, tg_name_, _lead_time_, 'test', balance_, 
                                                               predictions_proba_rc_ensemble = pred_proba_rc_test_ensemble, ecmwf = ecmwf_forecast_test_, outer_split_num_ = outer_split_num)   
        metrics_classi_test = build_metrics_classi(y_test_, pred_rc_test, pred_rfc_test, pred_rfc_test_ensemble, 
                                                   persist_test_, clim_forecast_test_, tg_name_, _lead_time_, 'test', balance_, 
                                                   predictions_rc_ensemble = pred_rc_test_ensemble, ecmwf = ecmwf_forecast_test_, outer_split_num_ = outer_split_num)
        if dictionary['verbosity'] > 1: print('********************************************* METRICS TRAIN FULL DATA **************************************')
        proba_metrics_classi_train = build_metrics_proba_classi(y_train_full_, pred_proba_rc_train_full, pred_proba_rfc_train_full, pred_proba_rfc_train_full_ensemble, 
                                                                persist_train_full_, clim_forecast_train_full_, tg_name_, _lead_time_, 'train', balance_, 
                                                                predictions_proba_rc_ensemble = pred_proba_rc_train_full_ensemble, outer_split_num_ = outer_split_num)
        metrics_classi_train = build_metrics_classi(y_train_full_, pred_rc_train_full, pred_rfc_train_full, pred_rfc_train_full_ensemble, 
                                                    persist_train_full_, clim_forecast_train_full_, tg_name_, _lead_time_, 'train', balance_, 
                                                    predictions_rc_ensemble = pred_rc_train_full_ensemble, outer_split_num_ = outer_split_num)

        # Save metrics
        save_metrics(metrics_classi_test, 'proba_classi', 'test', tg_name_, _lead_time_, balance_, outer_split_num_ = outer_split_num)
        save_metrics(proba_metrics_classi_test, 'classi', 'test', tg_name_, _lead_time_, balance_, outer_split_num_ = outer_split_num)
        save_metrics(metrics_classi_train, 'classi', 'train', tg_name_, _lead_time_, balance_, outer_split_num_ = outer_split_num)
        save_metrics(proba_metrics_classi_train, 'proba_classi', 'train', tg_name_, _lead_time_, balance_, outer_split_num_ = outer_split_num)


    ### 5. Plots regr
    if 'bin' not in tg_name_:
        ## Plot prediction against test data
        plot_pred_time_series(y_test_, pred_mlr_test, pred_rfr_test, persist_test_, clim_forecast_test_, ecmwf_forecast_test_, _lead_time_, tg_name_, outer_split_num_ = outer_split_num)
        plot_zoomed_in_pred_time_series(y_test_, pred_mlr_test, pred_rfr_test, persist_test_, clim_forecast_test_, ecmwf_forecast_test_, start_date_test_, _lead_time_, tg_name_, outer_split_num_ = outer_split_num)   
        ## Plot cross-correlation between predicted and real value
        plot_cross_corr(y_test_, pred_mlr_test.flatten(), pred_rfr_test.flatten(), ecmwf_forecast_test_, persist_test_, _lead_time_, tg_name_, outer_split_num_ = outer_split_num)


# In[5]:


def prepro_part2_and_prediction(tg_name, balance, lead_time_):

        
    """
    inputs
    ------
    tg_name                   str : name of target
    balance                   str : type of balance e.g. 'undersampling'
    lead_time_                int : lead time of prediction


    outputs
    -------

    None. Saves metrics, plots, best hyperparameters and time series to files. 
    
    """

    #---------------------------------------------------------------------------------------------------------------------------------------#    
    #---------------------------------------------------------------------------------------------------------------------------------------#  
    # Preprocessing part 2

    ## Note: This preprocessing part must be inside the lead_time loop, since it's lead_time-dependent
    ## 1. Load data
    if dictionary['verbosity'] > 2: print('Using ', dictionary['dataset'], ' data')
    # Load data from predictors
    data = xr.open_dataset(dictionary['path_input_prepro_pred'] + 'input_' + dictionary['timestep'] + '_' + dictionary['mean_type'] + '_mean_sa.nc')
    data = data.dropna('time')   
    # Load data from targets    
    access_targets_file = dictionary['path_input_targ'] + 'targets_' + dictionary['timestep'] +  '_' + dictionary['mean_type'] + '_mean.nc'
    targets = xr.open_dataset(access_targets_file).load().drop('dayofyear')
    if 'bin' in tg_name:
        if dictionary['verbosity'] > 4: print('Heat wave index: ', targets[tg_name])        
    # Add target as new variable to dataset
    data = data.assign(tg_name = targets[tg_name])
    data = data.rename({'tg_name': tg_name})    
    # Clean up
    ## Drop non-relevant variables
    if dictionary['drop_predictors'] is True:
        data = data.drop('na_index').drop('sst_med').drop('sst_semed').drop('snao').drop('sst_nwna').drop('sst_sena').drop('pcd_index').drop('sst_enso').drop('sst_baltic').drop('sm_deeper').drop('sm_deepest')
    ## Save predictor names
    predictor_names = list(data.drop(tg_name))
    ## Remove unnecessary dimension
    data = data.drop('mode', errors = 'ignore')
       
    if dictionary['verbosity'] > 2: print('Length of initial full dataset: ', len(data.time))
    if dictionary['verbosity'] > 4: print('Full dataset: ', data)
    
    #---------------------------------------------------------------------------------------------------------------------------------------#  
    ## 2. Lagged features   
    ### A. shifts the start by LT + num_lags to make sure it has the corresponding lagged predictors available
    ### B. Leaves a gap of LT + num_lags width between train_full & test set to avoid overlapping
    data_lagged_pred = lagged_features(data, tg_name, lead_time_)

    #---------------------------------------------------------------------------------------------------------------------------------------#      
    ## 3. Outer split
    # Split into train full and test (make sure that the splitting is in winter!) -> output: train_full, test
    train_full, start_date_train_full, end_date_train_full = split_sets(data_lagged_pred, 'train_full', lead_time_)
    test, start_date_test, end_date_test = split_sets(data_lagged_pred, 'test', lead_time_)   

    #---------------------------------------------------------------------------------------------------------------------------------------#  
    ## 4. Feature selection with Random Forest 
    if dictionary['feature_selection'] is True: 
        train_full, test = rf_feature_selection(train_full, test, tg_name)
   
    #---------------------------------------------------------------------------------------------------------------------------------------#  
    ## 5. Principal Component Analysis for dimensionality reduction
    if dictionary['pca'] is True: 
        train_full, test = PCA(train_full, test, tg_name, lead_time_)
        
    ### Save variable names
    feature_names = list(train_full)
    if dictionary['verbosity'] > 2: print('# features in dataset: ', len(feature_names))
        
    #---------------------------------------------------------------------------------------------------------------------------------------#  
    ## 6. Inner split
    # Split into train and validation (make sure that the splitting is in winter!) -> output: train, vali
    train, start_date_train, end_date_train = split_sets(train_full, 'train', lead_time_)
    vali, start_date_vali, end_date_vali = split_sets(train_full, 'vali', lead_time_) 
    
    #---------------------------------------------------------------------------------------------------------------------------------------#        
    ## 7. Season selection
    train_full = select_season(train_full, 'train_full')
    test = select_season(test, 'test')
    train = select_season(train, 'train')
    vali = select_season(vali, 'vali')
   
    #---------------------------------------------------------------------------------------------------------------------------------------#  
    ## 8. Leave one out train_full subsets for Jackknifes
    train_full_loo = loo_ensemble_rdm(train_full)
       
    #---------------------------------------------------------------------------------------------------------------------------------------#  
    ## 9. Balance out the data    
    ## Save balance sign
    if balance == 'undersampling': balance_sign = '-'
    elif balance == 'oversampling': balance_sign = '+'
    elif balance == 'none': balance_sign = ''
        
    # Use a rdm undersampling method to reduce the number of non-hw's in the hw indices to the number of hw's (50/50)
    ## Important to do the balancing only on the train set (also not on train_full, since it includes the validation set 
    ## -> create extra train_full_bal, but keep train_full intact), since otherwise the model can get biased or data 
    ## leakage can arise

    ## Needed for binary heatwave indices only
    if balance != 'none':
        if dictionary['verbosity'] > 2: print('Train')
        train = data_balance(balance, train, tg_name, feature_names)
        if dictionary['verbosity'] > 2: print('Train full')
        train_full_bal = data_balance(balance, train_full, tg_name, feature_names)
        train_full_loo_bal = {}
        for key in train_full_loo:
            train_full_loo_bal[key] = data_balance(balance, train_full_loo[key], tg_name, feature_names, show = False)                        
    else: 
        # Drop time index for training data
        train = train.reset_index().drop(columns = 'time', axis = 1) 
        train_full = train_full.reset_index().drop(columns = 'time', axis = 1) 
        for key in train_full_loo:
            train_full_loo[key] = train_full_loo[key].reset_index().drop(columns = 'time', axis = 1) 
    
        
    #---------------------------------------------------------------------------------------------------------------------------------------#      
    ## 10. Split up the data in target (y) and predictors (X): do it for train_full, train, vali, test and train_full_bal sets
    y_train_full, y_train, y_vali, y_test = train_full[tg_name], train[tg_name], vali[tg_name], test[tg_name]
    X_train_full, X_train, X_vali, X_test = train_full.drop(tg_name, axis = 1), train.drop(tg_name, axis = 1), vali.drop(tg_name, axis = 1), test.drop(tg_name, axis = 1)
    if balance != 'none':
        y_train_full_bal = train_full_bal[tg_name]
        X_train_full_bal = train_full_bal.drop(tg_name, axis = 1)
    else:
        X_train_full_bal = None
        y_train_full_bal = None
        
    y_train_full_loo = {}  
    X_train_full_loo = {}  
    y_train_full_loo_bal = {}  
    X_train_full_loo_bal = {}  
    for key in train_full_loo:
        y_train_full_loo[key] = train_full_loo[key][tg_name]
        X_train_full_loo[key] = train_full_loo[key].drop(tg_name, axis = 1)
        if balance != 'none':
            y_train_full_loo_bal[key] = train_full_loo_bal[key][tg_name]
            X_train_full_loo_bal[key] = train_full_loo_bal[key].drop(tg_name, axis = 1)
        else:
            X_train_full_loo_bal[key] = None
            y_train_full_loo_bal[key] = None
            
    ### Save feature names to variable
    feature_names = list(X_train_full)
    
    ### Save ground truth time series and index to file
    save_time_series(y_test, tg_name, 'GT', 'test', lead_time_)
    save_time_series(y_test.index, tg_name, 'index', 'test', lead_time_)    
    
    #---------------------------------------------------------------------------------------------------------------------------------------#    
    #---------------------------------------------------------------------------------------------------------------------------------------#    
    # Reference forecasts 
    persist_train_full, clim_train_full, persist_test, clim_test, ecmwf_test = compute_reference_forecasts(data, 
                                                                                                           y_train_full, y_test, 
                                                                                                           start_date_train_full, start_date_test, 
                                                                                                           end_date_train_full, end_date_test, 
                                                                                                           tg_name, 
                                                                                                           lead_time_)

    #---------------------------------------------------------------------------------------------------------------------------------------#    
    #---------------------------------------------------------------------------------------------------------------------------------------#  
    # Prediction
    ## Adapt formats of train and vali to fit nested CV
    key = 'inner_split_1'
    X_train, y_train, X_vali, y_vali = {key: X_train}, {key: y_train}, {key: X_vali}, {key: y_vali}
    ## Call prediction algorithm
    pred_algorithm(X_train, y_train, 
                   X_train_full, y_train_full, 
                   X_train_full_bal, y_train_full_bal, 
                   X_vali, y_vali, 
                   X_test, y_test, 
                   persist_train_full, clim_train_full, persist_test, clim_test, ecmwf_test, 
                   start_date_test,
                   predictor_names, feature_names, tg_name, 
                   balance, balance_sign, 
                   lead_time_,
                   X_train_full_loo = X_train_full_loo, y_train_full_loo = y_train_full_loo, 
                   X_train_full_loo_bal = X_train_full_loo_bal, y_train_full_loo_bal = y_train_full_loo_bal) 
    


# In[20]:


def prepro_part2_and_prediction_nested_cv(tg_name, balance, lead_time_):

        
    """
    inputs
    ------
    tg_name                   str : name of target
    balance                   str : type of balance e.g. 'undersampling'
    lead_time_                int : lead time of prediction


    outputs
    -------

    None. Saves metrics, plots, best hyperparameters and time series to files. 
    
    """

    #---------------------------------------------------------------------------------------------------------------------------------------#    
    #---------------------------------------------------------------------------------------------------------------------------------------#  
    # Preprocessing part 2
        
    ## Note: This preprocessing part must be inside the lead_time loop, since it's lead_time-dependent
    ## 1. Load data
    if dictionary['verbosity'] > 2: print('Using ', dictionary['dataset'], ' data')
    # Load data from predictors
    data = xr.open_dataset(dictionary['path_input_prepro_pred'] + 'input_' + dictionary['timestep'] + '_' + dictionary['mean_type'] + '_mean_sa.nc')
    data = data.dropna('time') 
    # Load data from targets    
    access_targets_file = dictionary['path_input_targ'] + 'targets_' + dictionary['timestep'] +  '_' + dictionary['mean_type'] + '_mean.nc'
    targets = xr.open_dataset(access_targets_file).load().drop('dayofyear')
    if 'bin' in tg_name:
        if dictionary['verbosity'] > 4: print('Heat wave index: ', targets[tg_name])        
    # Add target as new variable to dataset
    data = data.assign(tg_name = targets[tg_name])
    data = data.rename({'tg_name': tg_name})    
    # Clean up
    ## Drop non-relevant variables
    if dictionary['drop_predictors'] is True:
        data = data.drop('na_index').drop('sst_med').drop('sst_semed').drop('snao').drop('sst_nwna').drop('sst_sena').drop('pcd_index').drop('sst_enso').drop('sst_baltic').drop('sm_deeper').drop('sm_deepest')
    ## Save predictor names
    predictor_names = list(data.drop(tg_name))
    ## Remove unnecessary dimension
    data = data.drop('mode', errors = 'ignore')
       
    if dictionary['verbosity'] > 2: print('Length of initial full dataset: ', len(data.time))
    if dictionary['verbosity'] > 4: print('Full dataset: ', data)
    
    #---------------------------------------------------------------------------------------------------------------------------------------#  
    ## 2. Lagged features   
    ### A. shifts the start by LT + num_lags to make sure it has the corresponding lagged predictors available
    ### B. Leaves a gap of LT + num_lags width between train_full & test set to avoid overlapping
    data_lagged_pred = lagged_features(data, tg_name, lead_time_)

    #---------------------------------------------------------------------------------------------------------------------------------------#      
    ## 3. Outer split
    # Split into train full and test (make sure that the splitting is in winter!) -> output: train_full, test
    kfold = KFold(n_splits = dictionary['num_outer_folds'], shuffle = False)
    i = 1
    for train_full_ix, test_ix in kfold.split(data_lagged_pred):
        print('>> Outer split number ', i)        
        train_full, test, start_date_train_full, end_date_train_full, start_date_test, end_date_test, mid_end_date_train_full, mid_start_date_train_full = split_sets_nested_cv(data_lagged_pred, 
                                                                                                                                                                                train_full_ix, 
                                                                                                                                                                                test_ix, 
                                                                                                                                                                                'outer', 
                                                                                                                                                                                lead_time_)
        if not mid_end_date_train_full: mid_end_date_train_full = None
        if not mid_start_date_train_full: mid_start_date_train_full = None
        print('\n')    

        #---------------------------------------------------------------------------------------------------------------------------------------#  
        ## 4. Feature selection with Random Forest 
        if dictionary['feature_selection'] == True: 
            train_full, test = rf_feature_selection(train_full, test, tg_name)

        #---------------------------------------------------------------------------------------------------------------------------------------#  
        ## 5. Principal Component Analysis for dimensionality reduction
        if dictionary['pca'] == True: 
            train_full, test = PCA(train_full, test, tg_name, lead_time_, outer_split_num_ = i)
            
        ### Save variable names
        feature_names = list(train_full)
        if dictionary['verbosity'] > 2: print('# features in dataset: ', len(feature_names))
        
        #---------------------------------------------------------------------------------------------------------------------------------------#  
        ## 6. Inner split
        # Split into train and validation (make sure that the splitting is in winter!) -> output: train, vali
        kfold = KFold(n_splits = dictionary['num_inner_folds'], shuffle = False)
        ## Initialize
        j = 1
        train = {}
        vali = {}
        start_date_train = {}
        end_date_train = {}
        start_date_vali = {}
        end_date_vali = {}
        mid_end_date_train = {} 
        mid_start_date_train = {}
        ## Splits
        for train_ix, vali_ix in kfold.split(train_full):
            # Initialize train_full with its backup again
            #train_full = train_full_backup
            key = 'inner_split_{}'.format(j)
            print('>> Inner split number ', j)   
            (train[key], vali[key], 
            start_date_train[key], end_date_train[key], 
            start_date_vali[key], end_date_vali[key], 
            mid_end_date_train[key], mid_start_date_train[key]) = split_sets_nested_cv(train_full, 
                                                                                       train_ix, 
                                                                                       vali_ix, 
                                                                                       'inner', 
                                                                                       lead_time_)
            
            print('\n')
            j += 1 

        #---------------------------------------------------------------------------------------------------------------------------------------#        
        ## 7. Season selection
        train_full = select_season(train_full, 'train_full')
        test = select_season(test, 'test')
        for key in train:
            train[key] = select_season(train[key], 'train')
            vali[key] = select_season(vali[key], 'vali')

        #---------------------------------------------------------------------------------------------------------------------------------------#  
        ## 8. Balance train_full and train sets
        ### Use a rdm undersampling method to reduce the number of non-hw's in the hw indices to the number of hw's (50/50)
        ### Important to do the balancing only on the train set (also not on train_full, since it includes the validation set 
        ### -> create extra train_full_bal, but keep train_full intact), since otherwise the model can get biased or data 
        ### leakage can arise

        ### Save balance sign
        if balance == 'undersampling': balance_sign = '-'
        elif balance == 'oversampling': balance_sign = '+'
        elif balance == 'none': balance_sign = ''

        if balance != 'none':
            ### Balance needed for binary heatwave indices only
            if dictionary['verbosity'] > 2: print('Train full')
            train_full_bal = data_balance(balance, train_full, tg_name, feature_names)
            if dictionary['verbosity'] > 2: print('Train')
            for key in train:
                train[key] = data_balance(balance, train[key], tg_name, feature_names)
        else: 
            ### Drop time index
            train_full = train_full.reset_index().drop(columns = 'time', axis = 1)   
            for key in train:
                train[key] = train[key].reset_index().drop(columns = 'time', axis = 1) 

        #---------------------------------------------------------------------------------------------------------------------------------------#      
        ## 9. Split up the data in target (y) and predictors (X): do it for train_full, train, vali, test and train_full_bal sets
        y_train_full, y_test = train_full[tg_name], test[tg_name]
        X_train_full, X_test = train_full.drop(tg_name, axis = 1), test.drop(tg_name, axis = 1)
        
        y_train= {}
        y_vali = {}
        X_train = {}
        X_vali = {}
        for key in train:
            y_train[key], y_vali[key] = train[key][tg_name], vali[key][tg_name]
            X_train[key], X_vali[key] = train[key].drop(tg_name, axis = 1), vali[key].drop(tg_name, axis = 1)
               
        if balance != 'none':
            y_train_full_bal = train_full_bal[tg_name]
            X_train_full_bal = train_full_bal.drop(tg_name, axis = 1)
        else:
            X_train_full_bal = None
            y_train_full_bal = None
        
        ### Save predictor names to variable
        feature_names = list(X_train_full)

        ### Save ground truth time series and index to file
        save_time_series(y_test, tg_name, 'GT', 'test', lead_time_, outer_split_num_ = i)
        save_time_series(y_test.index, tg_name, 'index', 'test', lead_time_, outer_split_num_ = i)

        #---------------------------------------------------------------------------------------------------------------------------------------#    
        #---------------------------------------------------------------------------------------------------------------------------------------#    
        # Reference forecasts 
        persist_train_full, clim_train_full, persist_test, clim_test, ecmwf_test = compute_reference_forecasts(data, 
                                                                                                           y_train_full, y_test, 
                                                                                                           start_date_train_full, end_date_train_full, 
                                                                                                           start_date_test, end_date_test, 
                                                                                                           tg_name, 
                                                                                                           lead_time_,
                                                                                                           mid_end_date_train_full, mid_start_date_train_full)

        #---------------------------------------------------------------------------------------------------------------------------------------#    
        #---------------------------------------------------------------------------------------------------------------------------------------#  
        # Prediction
        pred_algorithm(X_train, y_train, 
                       X_train_full, y_train_full, 
                       X_train_full_bal, y_train_full_bal, 
                       X_vali, y_vali, 
                       X_test, y_test, 
                       persist_train_full, clim_train_full, persist_test, clim_test, ecmwf_test, 
                       start_date_test,
                       predictor_names, feature_names, tg_name, 
                       balance, balance_sign, 
                       lead_time_,
                       i)         

        # Increase outer split number 
        i += 1


# In[ ]:




