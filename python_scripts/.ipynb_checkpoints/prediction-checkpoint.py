""" HWAI prediction """

# Author: Elizabeth Weirich Benet 
# License: MIT License 2022


# Import packages
import matplotlib                             
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import random
import time
## Models
import sklearn
from sklearn.utils.extmath import softmax
from sklearn.linear_model import (LinearRegression, 
                                  Ridge, 
                                  RidgeClassifier)
from sklearn.ensemble import (RandomForestRegressor, 
                              RandomForestClassifier)
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV 

# Import own functions
## Utils
from utils import (save_to_file, 
                   print_model_name, 
                   save_time_series, 
                   traverse, 
                   interpolate, 
                   read_old_regr_ml_forecasts,
                   read_old_classi_ml_forecasts)                                 
## Plotting
from plotting import (show_coeffs, 
                      show_rf_imp, 
                      classification_plots,
                      plot_proba_histogram,
                      plot_pred_time_series,
                      plot_zoomed_in_pred_time_series,
                      plot_cross_corr)
## Metrics
from metrics import (compute_score,
                     frequency_bias_all_th,
                     threat_score_all_th,
                     build_metrics_regr,
                     build_metrics_classi,
                     build_metrics_proba_classi,
                     save_metrics)
## Reference forecasts
from reference_forecasts import compute_reference_forecasts
## Preprocessing 2
from preprocessing_part2 import (lagged_features, 
                                 split_sets, 
                                 split_sets_nested_cv, 
                                 select_season, 
                                 bootstrap_ensemble, 
                                 loo_ensemble_rdm)

# Import constants
from const import dictionary

# **----------------------------------------------------------------------------------------------------------------------------------------------**
# Class definitions 

# Ridge classifier which can make a probabilistic classification forecast
class RidgeClassifierwithProba(RidgeClassifier):
    def predict_proba(self, X):
        d = self.decision_function(X)
        d_2d = np.c_[-d, d]
        return softmax(d_2d)
    
    
# **----------------------------------------------------------------------------------------------------------------------------------------------**

# Function definitions 

def binarize(proba, threshold):
    
    """
    inputs
    ------
    proba                           pd.Series : probabilities of belonging to class 1 
    threshold                           float : probability threshold that marks the probability above which we define class 1


    outputs
    -------
    binary                          pd.Series : binary classification forecast
     
    """
    
    ## Binarize probability prediction with threshold
    binary = np.zeros(len(proba))
    for i in np.where(proba >= threshold):
        binary[i] = 1     
    
    return binary



def predict_classi(model1, model2, X, th):
    
    """
    inputs
    ------
    model1                         function : trained ML model
    model2                         function : trained and calibrated ML model (only for calibration is True, otherwise, equal to model)
    X                          pd.Dataframe : matrix of lagged predictors in the time period we want to forecast (time x lagged features)
    th                                float : probability threshold to binarize probabilistic output      


    outputs
    -------
    pred                          pd.Series : binary classification prediction by model1
    pred_proba                    pd.Series : probabilities of belonging to the class 1 (being a HW) by model2 (calibrated if calibration is True)
    pred_proba_4bin               pd.Series : probabilities of belonging to the class 1 (being a HW) by model1 (non-calibrated)
     
    """
    
    ## Predict
    pred_proba = model2.predict_proba(X)[:,1]
    pred_proba_4bin = model1.predict_proba(X)[:,1]
    
    ## Binarize model1 probability prediction with best threshold
    pred = binarize(pred_proba_4bin, th)  
    
    return pred, pred_proba, pred_proba_4bin




def choose_best_proba_threshold_classi(model, model_name, X, y, tgn_, _lead_time_, outer_split_num__ = None):
    
    """
    inputs
    ------
    model                           function : ML model (not trained)
    model_name                           str : 'RC' or 'RFC'
    X                     dict of xr.Dataset : predictors in train, train_full, vali, and test time periods (+ ensembles if no CV)
    y                   dict of xr.DataArray : target in train, train_full, vali, and test time periods (+ ensembles if no CV)
    tgn_                                 str : target name
    _lead_time_                          int : lead time of prediction in units of timestep
    " outer_split_num__                  int : counter for outer splits (only for nested CV case) "

    outputs
    -------
    best_threshold                     float : probability threshold that either sets B=1 or maximizes TS (user-defined setting under metric_th_sel
                                               in const.py)
    
    """

    if dictionary['verbosity'] > 1: print('**************************** CHOOSE PROBABILITY THRESHOLD ON VALIDATION SET ****************************') 
    
    # Choose training and validation sets
    ## No-CV: Use 30 CV-like train/vali splits created via leave-9y-out (loo) to stabilize the selection of the probability threshold
    if dictionary['cv_type'] == 'none':
        outer_split_num__ = None
        ens = '_th_ens'
    ## Nested-CV: Use 2 inner splits of the train_full set
    elif dictionary['cv_type'] == 'nested': ens = ''
    ## Select sets
    (X_train, y_train) = (X['train' + ens], y['train' + ens])
    (X_vali, y_vali) = (X['vali' + ens], y['vali' + ens])
        
    # Initialize 
    pred_proba = {}
    b = {}
    ts = {}
    
    # Loop over all 30 train/vali splits
    for key in X_train:  
        # Fit again with train period for further hyperparameter optimization step
        model.fit(X_train[key], y_train[key])
        # Predict heat wave probability
        pred_proba[key] = model.predict_proba(X_vali[key])[:,1]      
        # Frequency bias
        b_is, thresholds = frequency_bias_all_th(y_vali[key], pred_proba[key])  
        # Threat score
        ts_is = threat_score_all_th(y_vali[key], pred_proba[key], thresholds)     
        ## Make dataframe with index thresholds
        df = pd.DataFrame({'B': b_is, 'TS': ts_is}, index = thresholds)
        ## Reverse index
        df = df.reindex(index = df.index[::-1])
        ## Interpolate thresholds to universal linspace
        new_thresholds = np.linspace(0., 1., 600)
        df_interp = interpolate(df, new_thresholds)
        b[key] = df_interp.B
        ts[key] = df_interp.TS  

    # Mean over inner split
    b_mean = np.mean(list(b.values()), axis = 0) 
    ts_mean = np.mean(list(ts.values()), axis = 0) 

    # Select threshold
    if dictionary['metric_th_sel'] == 'B': 
        best_th_ix = np.nanargmin((np.abs(b_mean - 1)))  
    elif dictionary['metric_th_sel'] == 'TS': 
        best_th_ix = np.argmax(ts_mean)        
    # Index
    best_threshold = new_thresholds[best_th_ix]

    # Plot histogram with selected threshold for validation set
    if dictionary['verbosity'] > 3:
        for key in X_vali:  
            print('>> For ', key)
            plot_proba_histogram(pred_proba[key], 
                                 'vali', 
                                 tgn_, 
                                 model_name, 
                                 _lead_time_, 
                                 best_threshold = best_threshold, 
                                 outer_split_num_ = outer_split_num__, 
                                 inner_split_num_ = None)

    if dictionary['verbosity'] > 1: 
        print('Best Threshold = %f with (mean) frequency bias = %f and threat score = %f' %(best_threshold, b_mean[best_th_ix], ts_mean[best_th_ix]))  

    return best_threshold



def optimize_random_forest_hyperparameters(X, y, optimize_metric_, tgn_, _lead_time_, outer_split_num___ = None):
    
    """
    inputs
    ------
    
    X                    dict of xr.Dataset : predictors in train, train_full, vali, and test time periods (+ ensembles if no CV)
    y                  dict of xr.DataArray : target in train, train_full, vali, and test time periods (+ ensembles if no CV)
    optimize_metric_                    str : metric to optimize: Regression ('Corr' or 'RMSE'), Classification ('ROC AUC', 'PR AUC', or 'BS')
    tgn_                                str : name of target variable
    _lead_time_                         int : lead time of prediction in units of timestep 
    " outer_split_num___                int : counter for outer splits (only for nested CV case) "
    

    outputs
    -------
    best_rf                        function : random forest model initialized with the best hyperparameters (not trained)
    best_hyperparameters_dset_   xr.Dataset : hyperparameter set that optimizes optimize_metric_ when a single non-calibrated RF is trained on 
                                              the training set and forecasts the validation set. For nested CV, two training and validation sets
                                              are used for each outer split and the score is averaged. 
    
    """
    
    start = time.time()
    
    if dictionary['verbosity'] > 1: print('** ', dictionary['hp_search_type'], ' RF hyperparameter optimization **')
    
    if dictionary['cv_type'] == 'none':
        outer_split_num___ = None
        
    
    # Initialize
    ## Random state
    seed = 7
    ## score_max with its minumum (high for loss metrics, low for all other metrics)
    if optimize_metric_ in ['RMSE', 'BS']: score_max = 1000.
    else: score_max = 0.
    
    # Hyperparameter grid
    ## Number of trees in random forest
    n_estimators = [50, 100, 200, 400, 600]
    ## Minimum number of samples required at each leaf node
    first_guess_min_samples_leaf = int(len(X['train_full'])/100)
    step_min_samples_leaf = int(first_guess_min_samples_leaf/5)
    if step_min_samples_leaf < 1: step_min_samples_leaf = 1
    min_samples_leaf = np.arange(first_guess_min_samples_leaf - 5*step_min_samples_leaf, 
                                 first_guess_min_samples_leaf + 10*step_min_samples_leaf, 
                                 step_min_samples_leaf)
    ### Remove entries smaller than 1
    min_samples_leaf = min_samples_leaf[min_samples_leaf >= 1]
    ## Maximum number of levels in tree
    max_depth = np.arange(5, 15, 1)
    ## Maximum number of samples selected to train each estimator 
    max_samples = [None]
    ## Create the hyperparameter grid
    hp_grid = {'n_estimators': n_estimators,           
               'min_samples_leaf': min_samples_leaf.tolist(),           
               'max_depth': max_depth.tolist(),
               'max_samples': max_samples}
    
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
                                         max_samples = hp_set['max_samples'],
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
                                        max_samples = hp_set['max_samples'],
                                        random_state = seed, 
                                        criterion = 'squared_error', 
                                        n_jobs = dictionary['n_cores'])
            
        scores, pred_vali_ = [{} for i in range(2)]
        for key in X['train']: 
            ## Copy un-trained model
            rf = sklearn.base.clone(rf_)            
            ## Train model
            ### Use all training data for training
            rf.fit(X['train'][key], y['train'][key])
            ## Predict validation
            if 'bin' not in tgn_: pred_vali_ = rf.predict(X['vali'][key])
            elif 'bin' in tgn_: pred_vali_ = rf.predict_proba(X['vali'][key])[:,1]              
            ## Compute score used to optimize the hyperparameters 
            scores[key] = compute_score(optimize_metric_, y['vali'][key], pred_vali_)
                
        ## Take means 
        mean_score.append(np.mean(list(scores.values())))
        ## Show    
        if dictionary['verbosity'] > 3: print(hp_set, 'yield ', optimize_metric_, ' {:.3f}'.format(mean_score[i]))  
    
    ## Compare to previous scores & choose the best score to optimize the hyperparameters
    if optimize_metric_ in ['RMSE', 'BS']: 
        score_max = np.min(mean_score)
        score_max_ix = np.argmin(mean_score)
    else: 
        score_max = np.max(mean_score)
        score_max_ix = np.argmax(mean_score)
    best_hp_set = search_hp_grid[score_max_ix]
    
    # Best hyperparameters
    ## Print the best set of hyperparameters
    if dictionary['verbosity'] > 1: print('The best hyperparameter set: ', best_hp_set, 'yields ', optimize_metric_, ' {:.3f}'.format(score_max))  
    ## Initialize model with best hyperparameters & train it with the full training set 
    ### Initialize model
    if 'bin' in tgn_:     
        best_rf = RandomForestClassifier(n_estimators = best_hp_set['n_estimators'], 
                                     max_depth = best_hp_set['max_depth'], 
                                     min_samples_split = 2 * best_hp_set['min_samples_leaf'], 
                                     min_samples_leaf = best_hp_set['min_samples_leaf'], 
                                     bootstrap = True, 
                                     max_samples = best_hp_set['max_samples'],
                                     random_state = seed, 
                                     class_weight = 'balanced',
                                     criterion = 'gini', 
                                     n_jobs = dictionary['n_cores'])
        
    elif 'bin' not in tgn_:
        best_rf = RandomForestRegressor(n_estimators = best_hp_set['n_estimators'], 
                                    max_depth = best_hp_set['max_depth'], 
                                    min_samples_split = 2 * best_hp_set['min_samples_leaf'], 
                                    min_samples_leaf = best_hp_set['min_samples_leaf'], 
                                    bootstrap = True, 
                                    max_samples = best_hp_set['max_samples'],
                                    random_state = seed, 
                                    criterion = 'squared_error', 
                                    n_jobs = dictionary['n_cores'])
            
    ## Make dataset with best hyperparameters
    if 'bin' in tgn_: forecast_names = ['RFC']
    elif 'bin' not in tgn_: forecast_names = ['RFR']
    ### Build dataset
    best_hp_set_list = list(best_hp_set.values())
    best_hp_names = ['best_' + x for x in hp_grid.keys()]
    best_rf_hyperparameters_dset_ = xr.Dataset({'value': (('forecast', 'best_hyperparameter', 'lead_time'), np.reshape(best_hp_set_list, (1, len(best_hp_names), 1)))},
                                               coords = {'forecast': forecast_names, 'best_hyperparameter': best_hp_names, 'lead_time': [_lead_time_]})
    
    end = time.time()
    if dictionary['verbosity'] > 2: print('Time needed for hyperparameter optimization with ', len(search_hp_grid), ' candidates: ', (end - start)/60, ' min')
        
        
    return best_rf, best_rf_hyperparameters_dset_



def quick_choose_random_forest_hyperparameters(tgn_, _lead_time_, outer_split_num_ = None):
    
    """
    inputs
    ------
    tgn_                          str : name of target variable
    _lead_time_                   int : lead time of prediction in units of timestep
    " outer_split_num_            int : counter for outer splits (only for nested CV case) "

    outputs
    -------
    rf_                      function : random forest model initialized with the best hyperparameters saved the last time that optimize_rf_hyperparam 
                                        was True (not trained)
    
    """
    
    if dictionary['verbosity'] > 1: print('** Choose saved RF hyperparameters **')
    
    # Initialize
    seed = 7
    
    # Load best hyperparameters from last iteration of the optimize_random_forest_hyperparameters function
    ## Specifiy file to read
    path = dictionary['path_hyperparam'] + 'data/rf/'
    if outer_split_num_ is not None:
        file_name = path + 'best_rf_hyperparam_' + tgn_ + '_lead_time_' + str(_lead_time_) + '_outer_split_' + str(outer_split_num_) + '.nc'
    else:
        file_name = path + 'best_rf_hyperparam_' + tgn_ + '_lead_time_' + str(_lead_time_) + '.nc'
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
    if dictionary['verbosity'] > 1: 
        print('(n_estimators, max_depth, min_samples_split, min_samples_leaf) : (', n_estimators_, ', ', max_depth_, ', ', min_samples_split_, ', ', min_samples_leaf_, ')')        
    
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
                                    criterion = 'squared_error', 
                                    n_jobs = dictionary['n_cores'])
                
    return rf_




def optimize_ridge_hyperparameters(X, y, tgn_):
    
    
    """
    inputs
    ------
    X                    dict of xr.Dataset : predictors in train, train_full, vali, and test time periods (+ ensembles if no CV)
    y                  dict of xr.DataArray : target in train, train_full, vali, and test time periods (+ ensembles if no CV)
    tgn_                                str : name of target variable

    outputs
    -------
    best_model                     function : Ridge Regression model initialized with the best hyperparameters (not trained)
    best_alpha                        float : best regularization coefficient (alpha). Determines the amount of shrinkage of the 
                                              regression coefficients:
                                              1:maximum regularization
                                              0:no regularization
    
    """
    
    if dictionary['verbosity'] > 1: print('** Ridge hyperparameter optimization **')
    
    # Train Ridge that optimizes a given score
    ## Find metric
    if 'bin' in tgn_: optimize_metric_ = dictionary['metric_classi']
    else: optimize_metric_ = dictionary['metric_regr']
    
    ## Initialize score_max with its minumum
    if optimize_metric_ in ['RMSE', 'BS']: score_max = 1000.
    else: score_max = 0.
    ## Iterate over all hyperparameters and choose the best set    
    for alpha in np.round(np.arange(0., 1.05, 0.05), decimals = 2):
        ### Initialize model
        if 'bin' in tgn_:
            model_ = RidgeClassifierwithProba(alpha = alpha, max_iter = 1000)
        elif 'bin' not in tgn_:
            model_ = Ridge(alpha = alpha, max_iter = 1000)
                       
        score_new = {}
        for key in X['train']: 
            ## Copy un-trained model
            model = sklearn.base.clone(model_)
            ## Use all training data to train the model
            model.fit(X['train'][key], y['train'][key])
            ## Predict validation
            if 'bin' in tgn_: pred_vali_ = model.predict_proba(X['vali'][key])[:,1]
            else: pred_vali_ = model.predict(X['vali'][key])
            ### Compute score
            score_new[key] = compute_score(optimize_metric_, y['vali'][key], pred_vali_)
        score_new = np.mean(list(score_new.values()))      
        if dictionary['verbosity'] > 3: print('alpha: ', alpha, 'has ', optimize_metric_, ' score: ', score_new)

        ### Compare to previous scores & choose the best one
        if optimize_metric_ in ['RMSE', 'BS']:
            if (score_new < score_max): 
                score_max = score_new
                best_alpha = alpha
        else:
            if (score_new > score_max): 
                score_max = score_new
                best_alpha = alpha                
            
    if dictionary['verbosity'] > 1: print('Best alpha for Ridge is: ', best_alpha, 'with', optimize_metric_, 'score: ', score_max) 
        
    # Initialize model with best hyperparameters
    if 'bin' in tgn_:
        best_model = RidgeClassifierwithProba(alpha = best_alpha, max_iter = 1000)
    elif 'bin' not in tgn_:
        best_model = Ridge(alpha = best_alpha, max_iter = 1000)
        
    return best_model, best_alpha




def train_test_ridge_regressor(X, y, pred_names, feature_names, tgn, lt, outer_split_num__ = None):
    
    """
    inputs
    ------
    X                    dict of xr.Dataset : predictors in train, train_full, vali, and test time periods (+ ensembles if no CV)
    y                  dict of xr.DataArray : target in train, train_full, vali, and test time periods (+ ensembles if no CV)
    pred_names              list of strings : names of all predictors ordered like in X
    feature_names           list of strings : names of all features (lagged predictors) ordered like in X
    tgn                                 str : name of target
    lt                                  int : lead time of prediction in units of timestep
    " outer_split_num__                 int : counter for outer splits (only for nested CV case) "

    
    outputs
    -------
    pred                  dict of pd.Series : time series predicted by the RR model for the train_full and test time periods  
    " pred_ensemble       dict of pd.Series : ensemble for uncertainty estimation of time series predicted by the RR model for the train_full 
                                              and test time periods (only for no CV case) " 
    " best_a                    dict of int : best alpha for Ridge (only if optimize_linear_hyperparameters) "
    
    """   
    
    print_model_name('Ridge Regressor')
    
    ## Set outer split number to None for non-nested CV
    if dictionary['cv_type'] == 'none': outer_split_num__ = None
    
    # Initialize linear regression model and its ensemble
    if dictionary['optimize_linear_hyperparam'] is False: rr = Ridge(alpha = 1.0, max_iter = 1000)
    else: rr, best_a = optimize_ridge_hyperparameters(X, y, tgn)
    ### Initialize ensemble for uncertainty estimation
    if dictionary['cv_type'] == 'none': rr_ens = {key:sklearn.base.clone(rr) for key in X['train_full_bootstrap']}

    # Train with train_full
    if dictionary['verbosity'] > 1: 
        print('******************************************* TRAIN MODEL WITH TRAIN_FULL *********************************************')
    rr.fit(X['train_full'], y['train_full']) 
    ## Plot regression coefficients 
    if dictionary['verbosity'] > 1: show_coeffs(rr, 'RR', pred_names, feature_names, tgn, lt, outer_split_num__)
    
    # Predict    
    ## Initialize prediction dictionaries
    pred, pred_ensemble = [{'train_full': '', 'test': ''} for i in range(2)]    
    for subset in ['train_full', 'test']:   
        if dictionary['verbosity'] > 1: 
            print('******************************************* PREDICT', subset.upper(), 'DATA *********************************************')
        if dictionary['cv_type'] == 'none':
            ### Make ensenble of predictions based on models trained on subsets of train_full
            pred_ensemble[subset] = np.stack([rr_ens[key].fit(X['train_full_bootstrap'][key], y['train_full_bootstrap'][key]).predict(X[subset]) for key in X['train_full_bootstrap']])
            ### Final prediction is mean of all predictions
            pred[subset] = np.mean(pred_ensemble[subset], axis = 0)
            if dictionary['verbosity'] > 3:
                plt.figure(figsize = (20,5))
                for i in np.arange(len(pred_ensemble[subset])):       
                    plt.plot(np.arange(len(pred_ensemble[subset][i])), pred_ensemble[subset][i], linewidth = 1)
                plt.plot(np.arange(len(pred[subset])), pred[subset], linewidth = 3, color = 'black')
                plt.show()
                plt.close()
        elif dictionary['cv_type'] == 'nested': pred[subset] = rr.predict(X[subset])
        ## Save time series
        save_time_series(pred[subset], tgn, 'RR', subset, '', lt, outer_split_num__)
        if dictionary['cv_type'] == 'none': save_time_series(pred_ensemble[subset], tgn, 'RR_ensemble', subset, '', lt, outer_split_num__)

    if dictionary['cv_type'] == 'none': 
        if dictionary['optimize_linear_hyperparam']: return pred, pred_ensemble, best_a
        else: return pred, pred_ensemble, None
    elif dictionary['cv_type'] == 'nested': 
        if dictionary['optimize_linear_hyperparam']: return pred, {'train_full': None, 'test': None}, best_a
        else: return pred, {'train_full': None, 'test': None}, None




def train_test_random_forest_regressor(X, y, pred_names, feature_names, tgn, lt, outer_split_num__ = None):

    """
    inputs
    ------
    X                    dict of xr.Dataset : predictors in train, train_full, vali, and test time periods (+ ensembles if no CV)
    y                  dict of xr.DataArray : target in train, train_full, vali, and test time periods (+ ensembles if no CV)
    pred_names              list of strings : names of all predictors ordered like in X
    feature_names           list of strings : names of all features (lagged predictors) ordered like in X
    tgn                                 str : name of target
    lt                                  int : lead time of prediction in units of timestep
    " outer_split_num__                 int : counter for outer splits (only for nested CV case) "

    
    outputs
    -------       
    pred                  dict of pd.Series : time series predicted by the RFR model for the train_full and test time periods  
    " pred_ensemble       dict of pd.Series : ensemble for uncertainty estimation of time series predicted by the RFR model for the train_full 
                                              and test time periods (only for no CV case) " 
    " best_hp                   dict of int : best hyperparameters (only if optimize_rf_hyperparameters) "
    
    """   
    
    print_model_name('Random Forest Regressor')
    
    # Set outer split number to None for non-nested CV
    if dictionary['cv_type'] == 'none': outer_split_num__ = None
    
    ## Initialize Random Forest Regressor model with best hyperparameters
    if dictionary['optimize_rf_hyperparam'] is False:
        rfr = quick_choose_random_forest_hyperparameters(tgn, lt, outer_split_num_ = outer_split_num__) 
    elif dictionary['optimize_rf_hyperparam']:
        if dictionary['verbosity'] > 1: 
            print('************************************** OPTIMIZE ON VALIDATION DATA ****************************************')
        rfr, best_hp = optimize_random_forest_hyperparameters(X, y, dictionary['metric_regr'], tgn, lt, outer_split_num___ = outer_split_num__)
    ### Initialize ensemble for uncertainty estimation
    if dictionary['cv_type'] == 'none': rfr_ens = {key:sklearn.base.clone(rfr) for key in X['train_full_bootstrap']}

    # Train with train_full
    if dictionary['verbosity'] > 1: print('******************************************* TRAIN MODEL WITH TRAIN_FULL *********************************************')
    rfr.fit(X['train_full'], y['train_full']) 
    ## Show model's characteristics
    if dictionary['verbosity'] > 2: 
        print('Depth of trees :')
        print('Max: ', max([estimator.get_depth() for estimator in rfr.estimators_]))
        print('Min: ', min([estimator.get_depth() for estimator in rfr.estimators_]))
    ## Show feature importance
    if dictionary['verbosity'] > 1: show_rf_imp(rfr, pred_names, feature_names, tgn, lt, outer_split_num__)    
    
    # Predict    
    ## Initialize prediction dictionaries
    pred, pred_ensemble = [{'train_full': '', 'test': ''} for i in range(2)]
    
    for subset in ['train_full', 'test']:   
        if dictionary['verbosity'] > 1: 
            print('******************************************* PREDICT', subset.upper(), 'DATA *********************************************')
        if dictionary['cv_type'] == 'none':
            ### Make ensenble of predictions based on models trained on subsets of train_full
            pred_ensemble[subset] = np.stack([rfr_ens[key].fit(X['train_full_bootstrap'][key], y['train_full_bootstrap'][key]).predict(X[subset]) for key in X['train_full_bootstrap']])
            ### Final prediction is mean of all predictions
            pred[subset] = np.mean(pred_ensemble[subset], axis = 0)
            if dictionary['verbosity'] > 3:
                plt.figure(figsize = (20,5))
                for i in np.arange(len(pred_ensemble[subset])):       
                    plt.plot(np.arange(len(pred_ensemble[subset][i])), pred_ensemble[subset][i], linewidth = 1)
                plt.plot(np.arange(len(pred[subset])), pred[subset], linewidth = 3, color = 'black')
                plt.show()
                plt.close()
        elif dictionary['cv_type'] == 'nested': pred[subset] = rfr.predict(X[subset])
        ## Save time series
        save_time_series(pred[subset], tgn, 'RFR', subset, '', lt, outer_split_num__)
        if dictionary['cv_type'] == 'none': save_time_series(pred_ensemble[subset], tgn, 'RFR_ensemble', subset, '', lt, outer_split_num__)

    if dictionary['cv_type'] == 'none': 
        if dictionary['optimize_rf_hyperparam']: return pred, pred_ensemble, best_hp
        else: return pred, pred_ensemble, None
    else: 
        if dictionary['optimize_rf_hyperparam']: return pred, {'train_full': None, 'test': None}, best_hp
        else: return pred, {'train_full': None, 'test': None}, None



def train_test_ridge_classifier(X, y, pred_names, feature_names, tgn, lt, outer_split_num__ = None):

    """
    inputs
    -------
    X                    dict of xr.Dataset : predictors in train, train_full, vali, and test time periods (+ ensembles if no CV)
    y                  dict of xr.DataArray : target in train, train_full, vali, and test time periods (+ ensembles if no CV)
    pred_names              list of strings : names of all predictors ordered like in X
    feature_names           list of strings : names of all features (lagged predictors) ordered like in X
    tgn                                 str : name of target
    lt                                  int : lead time of prediction in units of timestep
    " outer_split_num__                 int : counter for outer splits (only for nested CV case) "
    " inner_split_num__                 int : counter for inner splits (only for nested CV case) "

    
    outputs
    -------
    pred                  dict of pd.Series : binary time series predicted by the RC model for the train_full and test time periods  
    " pred_ensemble       dict of pd.Series : ensemble for uncertainty estimation of binary time series predicted by the RC model for the train_full 
                                              and test time periods (only for no CV case) " 
    pred_proba            dict of pd.Series : probability time series predicted by the RC model for the train_full and test time periods  
    " pred_proba_ensemble dict of pd.Series : ensemble for uncertainty estimation of probability time series predicted by the RC model for the train_full 
                                              and test time periods (only for no CV case) " 
    " best_a                    dict of int : best alpha for Ridge (only if optimize_linear_hyperparameters) "
    
    """   
        
    print_model_name('Ridge Classifier') 
    
    # Set outer split number to None for non-nested CV
    if dictionary['cv_type'] == 'none': outer_split_num__ = None
    
    # Initialize linear classification model and its ensemble      
    if dictionary['optimize_linear_hyperparam'] is False: rc = RidgeClassifierwithProba(alpha = 1.0, normalize = False, max_iter = 1000)
    else: 
        if dictionary['verbosity'] > 1: print('************************************** OPTIMIZE ON VALIDATION DATA ****************************************')
        rc, best_a = optimize_ridge_hyperparameters(X, y, tgn)
    ## Initialize calibrated model
    if dictionary['calibrate_linear']: 
        model_type = 'calibrated' 
        rc_base = sklearn.base.clone(rc)
    else: model_type = 'non-calibrated'    
    ## Initialize ensemble for uncertainty estimation
    if dictionary['cv_type'] == 'none': 
        rc_ens = {}
        rc_ens_base = {}
        rc_ens_calib = {}
        for key in X['train_full_bootstrap']: 
            rc_ens[key] = sklearn.base.clone(rc)
            rc_ens_base[key] = sklearn.base.clone(rc)

    # Train    
    if dictionary['calibrate_linear']: 
        ## Train calibrated model for probabilistic prediction
        if dictionary['verbosity'] > 1: 
            print('******************************************* TRAIN', model_type.upper(), 'MODEL FOR PROBA PRED WITH TRAIN_FULL *********************************************')    
        ### Individual prediction
        for key in X['train']: # Key is not relevant in the case of no CV, only one train and vali sets (added for formatting reasons)
            rc_base.fit(X['train'][key], y['train'][key])
            rc_calib = CalibratedClassifierCV(rc_base, method = 'sigmoid', cv = 'prefit', n_jobs = dictionary['n_cores'])  
            rc_calib.fit(X['vali'][key], y['vali'][key]) 
        ### Ensemble prediction
        if dictionary['cv_type'] == 'none': 
            for key in X['train_full_bootstrap']:             
                rc_ens_base[key].fit(X['train_full_bootstrap'][key], y['train_full_bootstrap'][key])
                rc_ens_calib[key] = CalibratedClassifierCV(rc_ens_base[key], method = 'sigmoid', cv = 'prefit', n_jobs = dictionary['n_cores'])  
                rc_ens_calib[key].fit(X['vali_oob'][key], y['vali_oob'][key])         
    ## Train non-calibrated model
    if dictionary['verbosity'] > 1: 
        print('******************************************* TRAIN NON-CALIBRATED MODEL WITH TRAIN_FULL *********************************************')    
    ### Find best threshold to binarize probability prediction (use non-calibrated model which corresponds to the binary classification)
    best_th = choose_best_proba_threshold_classi(rc, 'RC', X, y, tgn, lt, outer_split_num__ = outer_split_num__)   
    ### Individual prediction
    rc.fit(X['train_full'], y['train_full'])
    ### Ensemble prediction
    if dictionary['cv_type'] == 'none': 
        for key in X['train_full_bootstrap']: rc_ens[key].fit(X['train_full_bootstrap'][key], y['train_full_bootstrap'][key]) 
    ### Plot coefficients
    if dictionary['verbosity'] > 1: show_coeffs(rc, 'RC', pred_names, feature_names, tgn, lt, outer_split_num__)
    
    # Predict    
    ## Initialize prediction dictionaries
    pred, pred_proba, pred_proba_4bin = [{'train_full': '', 'test': ''} for i in range(3)] 
    pred_ensemble, pred_proba_ensemble, pred_proba_4bin_ensemble = [{'train_full': [], 'test': []} for i in range(3)] 
    for subset in 'train_full', 'test':  
        if dictionary['verbosity'] > 1: 
            print('******************************************* PREDICT', subset.upper(), 'DATA *********************************************')    
        if dictionary['cv_type'] == 'none': 
            ### Make ensemble of predictions based on models trained on subsets of train_full
            for key in X['train_full_bootstrap']: 
                if dictionary['calibrate_linear']: 
                    ## Using calibrated model for probabilistic prediction and non-calibrated model for binary classification
                    pred_ensemble_key, pred_proba_ensemble_key, pred_proba_4bin_ensemble_key = predict_classi(rc_ens[key], rc_ens_calib[key], X[subset], best_th)
                else:
                    ## Using non-calibrated model for both problems
                    pred_ensemble_key, pred_proba_ensemble_key, pred_proba_4bin_ensemble_key = predict_classi(rc_ens[key], rc_ens[key], X[subset], best_th)
                pred_ensemble[subset].append(pred_ensemble_key)
                pred_proba_ensemble[subset].append(pred_proba_ensemble_key)
                pred_proba_4bin_ensemble[subset].append(pred_proba_4bin_ensemble_key)
            ### Final prediction is mean of all predictions
            pred_proba[subset] = np.mean(pred_proba_ensemble[subset], axis = 0)
            pred_proba_4bin[subset] = np.mean(pred_proba_4bin_ensemble[subset], axis = 0)
            pred[subset] = binarize(pred_proba_4bin[subset], best_th)
        elif dictionary['cv_type'] == 'nested': 
            if dictionary['calibrate_linear']: 
                ## Using calibrated model for probabilistic prediction and non-calibrated model for binary classification
                pred[subset], pred_proba[subset], pred_proba_4bin[subset] = predict_classi(rc, rc_calib, X[subset], best_th)
            else:
                ## Using non-calibrated model for both problems
                pred[subset], pred_proba[subset], pred_proba_4bin[subset] = predict_classi(rc, rc, X[subset], best_th)
                        
        # Show ensemble of predictions
        if dictionary['verbosity'] > 3 and dictionary['cv_type'] == 'none':
            plt.figure(figsize = (20,5))
            for i in range(len(pred_proba_ensemble[subset])):      
                plt.plot(np.arange(len(pred_proba_ensemble[subset][i])), pred_proba_ensemble[subset][i], linewidth = 1)
            plt.plot(np.arange(len(pred_proba[subset])), pred_proba[subset], linewidth = 3, color = 'black')
            plt.show()
            plt.close()
                
        ## Save time series
        model_name = 'RC'
        save_time_series(pred[subset], tgn, model_name, subset, '', lt, outer_split_num__) 
        save_time_series(pred_proba[subset], tgn, model_name, subset, '_proba', lt, outer_split_num__) 
        if dictionary['cv_type'] == 'none':
            save_time_series(pred_ensemble[subset], tgn, model_name + '_ensemble', subset, '', lt, outer_split_num__) 
            save_time_series(pred_proba_ensemble[subset], tgn, model_name + '_ensemble', subset, '_proba', lt, outer_split_num__) 
            
        ## Classification plots
        if dictionary['calibrate_linear']: 
            ### Probabilistic classification forecast
            if dictionary['verbosity'] > 0: print('\n\033[1m' + 'Calibrated probabilistic classification forecast' + '\033[0m') 
            classification_plots(y[subset], pred_proba[subset], subset, tgn, 'RC', lt, _outer_split_num_ = outer_split_num__)
            ### Binary classification forecast
            if dictionary['verbosity'] > 0: print('\n\033[1m' + 'Non-calibrated probabilities for binary classification forecast' + '\033[0m') 
            classification_plots(y[subset], pred_proba_4bin[subset], subset, tgn, 'RC', lt, best_th_ = best_th, _outer_split_num_ = outer_split_num__)
        else:       
            ### Classification probabilities coincide for both probabilistic and binary classification forecasts
            if dictionary['verbosity'] > 0: print('\n\033[1m' + 'Probabilities for probabilistic and binary classification forecast' + '\033[0m') 
            classification_plots(y[subset], pred_proba[subset], subset, tgn, 'RC', lt, best_th_ = best_th, _outer_split_num_ = outer_split_num__)
       
    
    if dictionary['cv_type'] == 'none': 
        if dictionary['optimize_linear_hyperparam']: return pred, pred_ensemble, pred_proba, pred_proba_ensemble, best_a
        else: return pred, pred_ensemble, pred_proba, pred_proba_ensemble, None
    elif dictionary['cv_type'] == 'nested': 
        if dictionary['optimize_linear_hyperparam']: return pred, {'train_full': None, 'test': None}, pred_proba, {'train_full': None, 'test': None}, best_a
        else: return pred, {'train_full': None, 'test': None}, pred_proba, {'train_full': None, 'test': None}, None
            



def train_test_random_forest_classifier(X, y, pred_names, feature_names, tgn, lt, outer_split_num__ = None):
    
    """
    inputs
    -------
    X                    dict of xr.Dataset : predictors in train, train_full, vali, and test time periods (+ ensembles if no CV)
    y                  dict of xr.DataArray : target in train, train_full, vali, and test time periods (+ ensembles if no CV)
    pred_names              list of strings : names of all predictors ordered like in X
    feature_names           list of strings : names of all features (lagged predictors) ordered like in X
    tgn                                 str : name of target
    lt                                  int : lead time of prediction in units of timestep
    " outer_split_num__                 int : counter for outer splits (only for nested CV case) "

    
    outputs
    -------
    pred                  dict of pd.Series : binary time series predicted by the RFC model for the train_full and test time periods  
    " pred_ensemble       dict of pd.Series : ensemble for uncertainty estimation of binary time series predicted by the RFC model for the train_full 
                                              and test time periods (only for no CV case) " 
    pred_proba            dict of pd.Series : probability time series predicted by the RFC model for the train_full and test time periods  
    " pred_proba_ensemble dict of pd.Series : ensemble for uncertainty estimation of probability time series predicted by the RFC model for the train_full 
                                              and test time periods (only for no CV case) "   
    " best_hp                   dict of int : best hyperparameters (only if optimize_rf_hyperparameters) "
    
    """  

    print_model_name('Random Forest Classifier')  
    
    # Set outer split number to None for non-nested CV
    if dictionary['cv_type'] == 'none': outer_split_num__ = None 
    
    # Initialize Classification Random Forest model with best hyperparameters      
    if dictionary['optimize_rf_hyperparam'] is False:
        rfc = quick_choose_random_forest_hyperparameters(tgn, lt, outer_split_num_ = outer_split_num__)
    else: 
        if dictionary['verbosity'] > 1: 
            print('************************************** OPTIMIZE ON VALIDATION DATA ****************************************')
        rfc, best_hp = optimize_random_forest_hyperparameters(X, y, dictionary['metric_classi'], tgn, lt, outer_split_num___ = outer_split_num__)      
    ## Initialize calibrated model
    if dictionary['calibrate_rf']: 
        model_type = 'calibrated' 
        rfc_base = sklearn.base.clone(rfc)
    else: model_type = 'non-calibrated'    
    ## Initialize ensemble for uncertainty estimation
    if dictionary['cv_type'] == 'none': 
        rfc_ens = {}
        rfc_ens_base = {}
        rfc_ens_calib = {}
        for key in X['train_full_bootstrap']: 
            rfc_ens[key] = sklearn.base.clone(rfc)
            rfc_ens_base[key] = sklearn.base.clone(rfc)

    # Train  
    if dictionary['calibrate_rf']: 
        ## Train calibrated model for probabilistic prediction
        if dictionary['verbosity'] > 1: 
            print('******************************************* TRAIN', model_type.upper(), 'MODEL FOR PROBA PRED WITH TRAIN_FULL *********************************************')       
        ### Individual prediction
        for key in X['train']: # Key is not relevant in the case of no CV, only one train and vali sets (added for formatting reasons)
            rfc_base.fit(X['train'][key], y['train'][key])
            rfc_calib = CalibratedClassifierCV(rfc_base, method = 'sigmoid', cv = 'prefit', n_jobs = dictionary['n_cores'])  
            rfc_calib.fit(X['vali'][key], y['vali'][key]) 
        ### Ensemble prediction
        if dictionary['cv_type'] == 'none': 
            for key in X['train_full_bootstrap']:             
                rfc_ens_base[key].fit(X['train_full_bootstrap'][key], y['train_full_bootstrap'][key])
                rfc_ens_calib[key] = CalibratedClassifierCV(rfc_ens_base[key], method = 'sigmoid', cv = 'prefit', n_jobs = dictionary['n_cores'])  
                rfc_ens_calib[key].fit(X['vali_oob'][key], y['vali_oob'][key]) 
    ## Train non-calibrated model
    if dictionary['verbosity'] > 1: 
        print('******************************************* TRAIN NON-CALIBRATED MODEL FOR BINARY PRED WITH TRAIN_FULL *********************************************')  
    ## Find best threshold to binarize probability prediction (use non-calibrated model which corresponds to the binary classification)
    best_th = choose_best_proba_threshold_classi(rfc, 'RFC', X, y, tgn, lt, outer_split_num__ = outer_split_num__) 
    ### Individual prediction
    rfc.fit(X['train_full'], y['train_full'])
    ### Ensemble prediction
    if dictionary['cv_type'] == 'none': 
        for key in X['train_full_bootstrap']: rfc_ens[key].fit(X['train_full_bootstrap'][key], y['train_full_bootstrap'][key])        
    ## Show model characteristics    
    if dictionary['verbosity'] > 2: 
        print('Depth of trees :')
        print('Max: ', max([estimator.get_depth() for estimator in rfc.estimators_]))
        print('Min: ', min([estimator.get_depth() for estimator in rfc.estimators_]))
    ## Show feature importance
    if dictionary['verbosity'] > 1: show_rf_imp(rfc, pred_names, feature_names, tgn, lt, outer_split_num__) 
    
        
    # Predict    
    ## Initialize prediction dictionaries
    pred, pred_proba, pred_proba_4bin = [{'train_full': '', 'test': ''} for i in range(3)] 
    pred_ensemble, pred_proba_ensemble, pred_proba_4bin_ensemble = [{'train_full': [], 'test': []} for i in range(3)] 
    for subset in 'train_full', 'test':  
        if dictionary['verbosity'] > 1: 
            print('******************************************* PREDICT', subset.upper(), 'DATA *********************************************')
        if dictionary['cv_type'] == 'none':
            ### Make ensenble of predictions based on models trained on subsets of train_full
            for key in X['train_full_bootstrap']: 
                if dictionary['calibrate_rf']: 
                    ## Using calibrated model for probabilistic prediction and non-calibrated model for binary classification
                    pred_ensemble_key, pred_proba_ensemble_key, pred_proba_4bin_ensemble_key = predict_classi(rfc_ens[key], rfc_ens_calib[key], X[subset], best_th)
                else:
                    ## Using non-calibrated model for both problems
                    pred_ensemble_key, pred_proba_ensemble_key, pred_proba_4bin_ensemble_key = predict_classi(rfc_ens[key], rfc_ens[key], X[subset], best_th)
                pred_ensemble[subset].append(pred_ensemble_key)
                pred_proba_ensemble[subset].append(pred_proba_ensemble_key)
                pred_proba_4bin_ensemble[subset].append(pred_proba_4bin_ensemble_key)
            ### Final prediction is mean of all predictions
            pred_proba[subset] = np.mean(pred_proba_ensemble[subset], axis = 0)
            pred_proba_4bin[subset] = np.mean(pred_proba_4bin_ensemble[subset], axis = 0) 
            pred[subset] = binarize(pred_proba_4bin[subset], best_th)
        elif dictionary['cv_type'] == 'nested': 
            if dictionary['calibrate_rf']: 
                ## Using calibrated model for probabilistic prediction and non-calibrated model for binary classification
                pred[subset], pred_proba[subset], pred_proba_4bin[subset] = predict_classi(rfc, rfc_calib, X[subset], best_th)
            else:
                ## Using non-calibrated model for both problems
                pred[subset], pred_proba[subset], pred_proba_4bin[subset] = predict_classi(rfc, rfc, X[subset], best_th)
                
        # Show ensemble of predictions
        if dictionary['verbosity'] > 3 and dictionary['cv_type'] == 'none':
            plt.figure(figsize = (20,5))
            for i in range(len(pred_proba_ensemble[subset])):      
                plt.plot(np.arange(len(pred_proba_ensemble[subset][i])), pred_proba_ensemble[subset][i], linewidth = 1)
            plt.plot(np.arange(len(pred_proba[subset])), pred_proba[subset], linewidth = 3, color = 'black')
            plt.show()
            plt.close()            
                               
        ## Save time series
        model_name = 'RFC'
        save_time_series(pred[subset], tgn, model_name, subset, '', lt, outer_split_num__) 
        save_time_series(pred_proba[subset], tgn, model_name, subset, '_proba', lt, outer_split_num__) 
        if dictionary['cv_type'] == 'none':
            save_time_series(pred_ensemble[subset], tgn, model_name + '_ensemble', subset, '', lt, outer_split_num__) 
            save_time_series(pred_proba_ensemble[subset], tgn, model_name + '_ensemble', subset, '_proba', lt, outer_split_num__) 
            
            
        ## Classification plots
        if dictionary['calibrate_rf']: 
            ### Probabilistic classification forecast
            if dictionary['verbosity'] > 0: print('\n\033[1m' + 'Calibrated probabilistic classification forecast' + '\033[0m') 
            classification_plots(y[subset], pred_proba[subset], subset, tgn, 'RFC', lt, _outer_split_num_ = outer_split_num__)
            ### Binary classification forecast
            if dictionary['verbosity'] > 0: print('\n\033[1m' + 'Non-calibrated probabilities for binary classification forecast' + '\033[0m') 
            classification_plots(y[subset], pred_proba_4bin[subset], subset, tgn, 'RFC', lt, best_th_ = best_th, _outer_split_num_ = outer_split_num__)
        else:       
            ### Classification probabilities coincide for both probabilistic and binary classification forecasts
            if dictionary['verbosity'] > 0: print('\n\033[1m' + 'Probabilities for probabilistic and binary classification forecast' + '\033[0m') 
            classification_plots(y[subset], pred_proba[subset], subset, tgn, 'RFC', lt, best_th_ = best_th, _outer_split_num_ = outer_split_num__)       

        
    if dictionary['cv_type'] == 'none':
        if dictionary['optimize_rf_hyperparam']: return pred, pred_ensemble, pred_proba, pred_proba_ensemble, best_hp
        else: return pred, pred_ensemble, pred_proba, pred_proba_ensemble, None
    else: 
        if dictionary['optimize_rf_hyperparam']: return pred, {'train_full': None, 'test': None}, pred_proba, {'train_full': None, 'test': None}, best_hp
        else: return pred, {'train_full': None, 'test': None}, pred_proba, {'train_full': None, 'test': None}, None
    
    


def pred_algorithm(X, y, persist_forecast, clim_forecast, ecmwf_forecast, 
                   start_date_test_, predictor_names_, feature_names_, tg_name_, _lead_time_, 
                   outer_split_num = None):
        
    """
    inputs
    ------
    X                            dict of xr.Dataset : predictors in train, train_full, vali, and test time periods (+ ensembles if no CV)
    y                          dict of xr.DataArray : target in train, train_full, vali, and test time periods (+ ensembles if no CV)
    persist_forecast                   xr.DataArray : persistence forecast of target for train_full and test time period
    clim_forecast                      xr.DataArray : climatology forecast of target for train_full and test time period
    ecmwf_forecast                     xr.DataArray : ECMWF forecast of target for test time period
    start_date_test_                            str : start date of test time period
    predictor_names_                list of strings : names of all predictors ordered like in X
    feature_names_                  list of strings : names of all features (lagged predictors) ordered like in X
    tg_name_                                    str : name of target
    _lead_time_                                 int : lead time of prediction in units of timestep
    " outer_split_num                           int : counter for outer splits (only for nested CV case) "


    outputs
    -------
    None. Plots ROC AUC, PR, and reliability curves for classification
          Plots probability histogram for classification
          Saves best hyperparameters to file if hyperparameter optimization is True
          Saves time series to file
          Plots metrics table
          Saves metrics to file
          Plots time series and lagged correlation plots for regression (for a particular lead time)
    
    """
    

    # Choose models at different end of the bias-variance trade off:
    # 
    #     1. Regression // 2. Classification
    # 
    #         1.1. Ridge Regressor // 2.1. Ridge Classifier (high bias, low variance)
    # 
    #         1.2. Random Forest Regressor // 2.2. Random Forest Classifier (low bias, high variance)
    
    # Set outer split number to None for the case of no CV
    if dictionary['cv_type'] == 'none': outer_split_num = None
    
    # Set ensembles to None for the case of nested CV
    if dictionary['cv_type'] == 'nested':
        for subset in ['train_full_bootstrap', 'vali_oob', 'train_th_ens', 'vali_th_ens']:
            X[subset], y[subset] = [None for i in range(2)]
       
    # Train models and make predictions
    if dictionary['train_models']: 
        ## 1. Regression
        # **For continuous targets: t2m**
        if 'bin' not in tg_name_:
            ### 1.1. Ridge Regressor (RR)
            pred_rr, pred_rr_ensemble, best_hyperparam_linear = train_test_ridge_regressor(X, 
                                                                                           y, 
                                                                                           predictor_names_, 
                                                                                           feature_names_, 
                                                                                           tg_name_, 
                                                                                           _lead_time_,
                                                                                           outer_split_num__ = outer_split_num)
                                                                                                                            
            ### 1.2. Random Forest Regressor (RFR)
            pred_rfr, pred_rfr_ensemble, best_hyperparam_rf = train_test_random_forest_regressor(X, 
                                                                                                 y, 
                                                                                                 predictor_names_, 
                                                                                                 feature_names_, 
                                                                                                 tg_name_, 
                                                                                                 _lead_time_, 
                                                                                                 outer_split_num__ = outer_split_num)

        ## 2. Classification 
        # **For binary targets: hw_bin_1SD, hw_bin_15SD**
        elif 'bin' in tg_name_:
            ### 2.1. Ridge Classifier (RC)
            pred_rc, pred_rc_ensemble, pred_proba_rc, pred_proba_rc_ensemble, best_hyperparam_linear = train_test_ridge_classifier(X, 
                                                                                                                                   y, 
                                                                                                                                   predictor_names_, 
                                                                                                                                   feature_names_, 
                                                                                                                                   tg_name_, 
                                                                                                                                   _lead_time_, 
                                                                                                                                   outer_split_num__ = outer_split_num)

            ### 2.2. Random Forest Classifier (RFC)
            pred_rfc, pred_rfc_ensemble, pred_proba_rfc, pred_proba_rfc_ensemble, best_hyperparam_rf = train_test_random_forest_classifier(X, 
                                                                                                                                           y, 
                                                                                                                                           predictor_names_, 
                                                                                                                                           feature_names_, 
                                                                                                                                           tg_name_, 
                                                                                                                                           _lead_time_, 
                                                                                                                                           outer_split_num__ = outer_split_num) 
        ## 3. Best hyperparameters
        # Save best hyperparameters for Ridge
        if dictionary['optimize_linear_hyperparam']:            
            if outer_split_num is not None:
                save_name = 'best_linear_hyperparam_' + tg_name_ + '_lead_time_' + str(_lead_time_) + '_outer_split_' + str(outer_split_num) + '.npy'
            else:
                save_name = 'best_linear_hyperparam_' + tg_name_ + '_lead_time_' + str(_lead_time_) + '.npy'
            save_to_file(best_hyperparam_linear, dictionary['path_hyperparam'] + 'data/linear/', save_name, 'np')        
        # Save best hyperparameters for RFs 
        if dictionary['optimize_rf_hyperparam']:            
            if outer_split_num is not None:
                save_name = 'best_rf_hyperparam_' + tg_name_ + '_lead_time_' + str(_lead_time_) + '_outer_split_' + str(outer_split_num) + '.nc'
            else:
                save_name = 'best_rf_hyperparam_' + tg_name_ + '_lead_time_' + str(_lead_time_) + '.nc'
            save_to_file(best_hyperparam_rf, dictionary['path_hyperparam'] + 'data/rf/', save_name, 'nc')
            
    # Read old saved predictions by the ML models (time series) if model training deactivated
    else: 
        if 'bin' not in tg_name_:
                 pred_rr, pred_rr_ensemble, pred_rfr, pred_rfr_ensemble = read_old_regr_ml_forecasts(tg_name_, 
                                                                                                     _lead_time_, 
                                                                                                     outer_split_num_ = outer_split_num)
        if 'bin' in tg_name_:
                 (pred_rc, pred_rc_ensemble, pred_proba_rc, pred_proba_rc_ensemble,
                  pred_rfc, pred_rfc_ensemble, pred_proba_rfc, pred_proba_rfc_ensemble) = read_old_classi_ml_forecasts(tg_name_, 
                                                                                                                       _lead_time_, 
                                                                                                                       outer_split_num_ = outer_split_num)
        
        
    ## 4. Metrics    
    for subset in ['train_full', 'test']:
        if dictionary['verbosity'] > 1: print('*********************************************** METRICS', subset.upper(), 'DATA *****************************************')
        ### 4.1. Plot metrics table
        if 'bin' not in tg_name_:
            pred_type = 'regr'            
            metrics = build_metrics_regr(y[subset], 
                                         pred_rr[subset], 
                                         pred_rfr[subset],
                                         persist_forecast[subset], 
                                         clim_forecast[subset], 
                                         tg_name_, 
                                         _lead_time_, 
                                         subset, 
                                         predictions_rr_ensemble = pred_rr_ensemble[subset], 
                                         predictions_rfr_ensemble = pred_rfr_ensemble[subset], 
                                         ecmwf = ecmwf_forecast[subset], 
                                         outer_split_num_ = outer_split_num)                   
        elif 'bin' in tg_name_:
            pred_type = 'classi'
            metrics_proba = build_metrics_proba_classi(y[subset], 
                                                       pred_proba_rc[subset], 
                                                       pred_proba_rfc[subset], 
                                                       persist_forecast[subset], 
                                                       clim_forecast[subset], 
                                                       tg_name_, 
                                                       _lead_time_, 
                                                       subset, 
                                                       predictions_proba_rc_ensemble = pred_proba_rc_ensemble[subset], 
                                                       predictions_proba_rfc_ensemble = pred_proba_rfc_ensemble[subset],
                                                       ecmwf = ecmwf_forecast[subset], 
                                                       outer_split_num_ = outer_split_num)   
            
            metrics = build_metrics_classi(y[subset], 
                                           pred_rc[subset], 
                                           pred_rfc[subset], 
                                           persist_forecast[subset], 
                                           clim_forecast[subset], 
                                           tg_name_, 
                                           _lead_time_, 
                                           subset, 
                                           predictions_rc_ensemble = pred_rc_ensemble[subset], 
                                           predictions_rfc_ensemble = pred_rfc_ensemble[subset],
                                           ecmwf = ecmwf_forecast[subset], 
                                           outer_split_num_ = outer_split_num)
        ### 4.2. Save metrics
        if 'bin' in tg_name_: 
            save_metrics(metrics_proba, 'proba_' + pred_type, subset, tg_name_, _lead_time_, outer_split_num_ = outer_split_num)
        save_metrics(metrics, pred_type, subset, tg_name_, _lead_time_, outer_split_num_ = outer_split_num)  
        
        
    ### 5. Regression plots for a particular lead time
    if 'bin' not in tg_name_:
        ## Plot prediction time series of test data
        plot_pred_time_series(y['test'], 
                              pred_rr['test'], 
                              pred_rfr['test'], 
                              persist_forecast['test'], 
                              clim_forecast['test'], 
                              ecmwf_forecast['test'], 
                              _lead_time_, 
                              tg_name_, 
                              outer_split_num_ = outer_split_num)
        plot_zoomed_in_pred_time_series(y['test'], 
                                        pred_rr['test'], 
                                        pred_rfr['test'], 
                                        persist_forecast['test'], 
                                        clim_forecast['test'], 
                                        ecmwf_forecast['test'], 
                                        start_date_test_, 
                                        _lead_time_, 
                                        tg_name_, 
                                        outer_split_num_ = outer_split_num)   
        
        ## Plot cross-correlation between ML forecasts and ground truth and ML forecasts and persistence forecast for test time period
        plot_cross_corr(y['test'], 
                        pred_rr['test'].flatten(), 
                        pred_rfr['test'].flatten(), 
                        ecmwf_forecast['test'], 
                        persist_forecast['test'], 
                        _lead_time_, 
                        tg_name_, 
                        outer_split_num_ = outer_split_num)



        

def prepro_part2_and_prediction(tg_name, lead_time_):

        
    """
    inputs
    ------
    tg_name                   str : name of target
    lead_time_                int : lead time of prediction in units of timestep


    outputs
    -------

    None. Calls reference forecasts and prediction function, which save time series, tables, and plots. 
    
    This function is called in the no CV case. 
    
    """

    #---------------------------------------------------------------------------------------------------------------------------------------#    
    #---------------------------------------------------------------------------------------------------------------------------------------#  
    # Preprocessing part 2
    ## Initialize dictionaries
    dset, X, y, start_date, end_date = [{'train': '', 
                                         'test': '', 
                                         'train_full': '', 
                                         'vali': '', 
                                         'train_full_bootstrap': {},
                                         'vali_oob': {},
                                         'train_th_ens': {},
                                         'vali_th_ens': {}} for i in range(5)]

    ## Note: This preprocessing part must be inside the lead_time loop, since it's lead_time-dependent
    ## 1. Load data
    # Load data from predictors
    data = xr.open_dataset(dictionary['path_data'] + 'preprocessed_predictors_7D_rolling_mean.nc')
    data = data.dropna('time')
    ## Choose predictors
    data = data[['t2m_x', 'z', 'rain', 'sm', 'sea', 'sst_nwmed', 'sst_cnaa']]  
    ## Save predictor names
    predictor_names = list(data)
    # Load data from targets    
    access_targets_file = dictionary['path_data'] + 'targets_7D_rolling_mean.nc'
    targets = xr.open_dataset(access_targets_file).load().drop('dayofyear')
    if 'bin' in tg_name:
        if dictionary['verbosity'] > 4: print('Heat wave index: ', targets[tg_name])        
    # Add target as new variable to dataset
    data = data.assign(tg_name = targets[tg_name])
    data = data.rename({'tg_name': tg_name})    
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
    # Split into train full and test (make sure that the splitting is in winter!) -> output: dset containing train_full and test
    for subset in ['train_full', 'test']:
        dset[subset], start_date[subset], end_date[subset] = split_sets(data_lagged_pred, subset, lead_time_) 
        
    ### Save variable names
    feature_names = list(dset['train_full'])
    if dictionary['verbosity'] > 2: print('# features in dataset: ', len(feature_names))
        
    #---------------------------------------------------------------------------------------------------------------------------------------#  
    ## 4. Inner split
    # Split into train and validation (make sure that the splitting is in winter!) -> output: train, vali
    for subset in ['train', 'vali']:
        dset[subset], start_date[subset], end_date[subset] = split_sets(dset['train_full'], subset, lead_time_)
    
    #---------------------------------------------------------------------------------------------------------------------------------------#        
    ## 5. Season selection
    for subset in ['train_full', 'test', 'train', 'vali']:
        dset[subset] = select_season(dset[subset], subset)
   
    #---------------------------------------------------------------------------------------------------------------------------------------#  
    ## 6. Bootstrapping
    dset['train_full_bootstrap'], dset['vali_oob'] = bootstrap_ensemble(dset['train_full'])
    
    #---------------------------------------------------------------------------------------------------------------------------------------#  
    ## 7. Ensemble for probability threshold selection
    dset['train_th_ens'], dset['vali_th_ens'] = loo_ensemble_rdm(dset['train_full'])
       
    #---------------------------------------------------------------------------------------------------------------------------------------#    
    ## 8. Reference forecasts 
    persist, clim, ecmwf = compute_reference_forecasts(data, 
                                                       dset, 
                                                       start_date, 
                                                       end_date, 
                                                       tg_name, 
                                                       lead_time_)
    
    #---------------------------------------------------------------------------------------------------------------------------------------#    
    ## 9. Split up the data in target (y) and predictors (X): do it for train_full, train, vali, and test sets
    ### Drop time index for training data
    for subset in ['train', 'train_full']:
        dset[subset] = dset[subset].reset_index().drop(columns = 'time', axis = 1)
        
    ### Split for individual forecasts
    for subset in ['train_full', 'test', 'train', 'vali']:
        y[subset] = dset[subset][tg_name]
        X[subset] = dset[subset].drop(tg_name, axis = 1) 
        
    ### Split for ensemble of forecasts      
    for subset in ['train_full_bootstrap', 'vali_oob', 'train_th_ens', 'vali_th_ens']:
        for key in dset[subset]:
            y[subset][key] = dset[subset][key][tg_name]
            X[subset][key] = dset[subset][key].drop(tg_name, axis = 1)  
            
    ### Save feature names to variable
    feature_names = list(X['train_full'])
    
    ### Save ground truth time series and index to file
    save_time_series(y['test'], tg_name, 'GT', 'test', '', lead_time_)
    save_time_series(y['test'].index, tg_name, 'index', 'test', '', lead_time_)    

    #---------------------------------------------------------------------------------------------------------------------------------------#    
    #---------------------------------------------------------------------------------------------------------------------------------------#  
    # Prediction
    ## Adapt formats of train and vali to fit nested CV (pred_algorithm must be able to handle inputs from both no CV and nested CV)
    key = 'inner_split_1'
    for subset in ['train', 'vali']:
        X[subset], y[subset] = {key: X[subset]}, {key: y[subset]}
    ## Call prediction algorithm
    pred_algorithm(X, y,
                   persist, clim, ecmwf, 
                   start_date['test'],
                   predictor_names, feature_names, tg_name, 
                   lead_time_) 
    


def prepro_part2_and_prediction_nested_cv(tg_name, lead_time_):

        
    """
    inputs
    ------
    tg_name                   str : name of target
    lead_time_                int : lead time of prediction in units of timestep


    outputs
    -------

    None. Calls reference forecasts and prediction function, which save time series, tables, and plots. 
    
    This function is called in the nested CV case.  
    
    """

    #---------------------------------------------------------------------------------------------------------------------------------------#    
    #---------------------------------------------------------------------------------------------------------------------------------------#  
    # Preprocessing part 2
    ## Initialize dictionaries
    dset, X, y, start_date, mid_end_date, mid_start_date, end_date = [{'train': {}, 
                                                                       'test': '', 
                                                                       'train_full': '', 
                                                                       'vali': {}, 
                                                                       'train_full_bootstrap': {},
                                                                       'vali_oob': {},
                                                                       'train_th_ens': {},
                                                                       'vali_th_ens': {}} for i in range(7)]
    
    
    ## Note: This preprocessing part must be inside the lead_time loop, since it's lead_time-dependent
    ## 1. Load data
    # Load data from predictors
    data = xr.open_dataset(dictionary['path_data'] + 'preprocessed_predictors_7D_rolling_mean.nc')
    data = data.dropna('time') 
    ## Choose predictors
    data = data[['t2m_x', 'z', 'rain', 'sm', 'sea', 'sst_nwmed', 'sst_cnaa']]        
    ## Save predictor names
    predictor_names = list(data)
    # Load data from targets    
    access_targets_file = dictionary['path_data'] + 'targets_7D_rolling_mean.nc'
    targets = xr.open_dataset(access_targets_file).load().drop('dayofyear')
    if 'bin' in tg_name:
        if dictionary['verbosity'] > 4: print('Heat wave index: ', targets[tg_name])        
    # Add target as new variable to dataset
    data = data.assign(tg_name = targets[tg_name])
    data = data.rename({'tg_name': tg_name})    
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
        if dictionary['verbosity'] > 0: print('>> Outer split number ', i)        
        (dset['train_full'], 
         dset['test'], 
         start_date['train_full'], 
         end_date['train_full'], 
         start_date['test'], 
         end_date['test'], 
         mid_end_date['train_full'], 
         mid_start_date['train_full']) = split_sets_nested_cv(data_lagged_pred,                                                                                                                                                                                                             train_full_ix, 
                                                              test_ix, 
                                                              'outer', 
                                                              lead_time_)
        if dictionary['verbosity'] > 0: print('\n')    
            
        ### Save variable names
        feature_names = list(dset['train_full'])
        if dictionary['verbosity'] > 2: print('# features in dataset: ', len(feature_names))

        #---------------------------------------------------------------------------------------------------------------------------------------#  
        ## 4. Inner split
        # Split into train and validation (make sure that the splitting is in winter!) -> output: train, vali
        kfold = KFold(n_splits = dictionary['num_inner_folds'], shuffle = False)
        ## Initialize
        j = 1
        ## Splits
        for train_ix, vali_ix in kfold.split(dset['train_full']):
            key = 'inner_split_{}'.format(j)
            if dictionary['verbosity'] > 0: print('>> Inner split number ', j)   
            (dset['train'][key], dset['vali'][key], 
            start_date['train'][key], end_date['train'][key], 
            start_date['vali'][key], end_date['vali'][key], 
            mid_end_date['train'][key], mid_start_date['train'][key]) = split_sets_nested_cv(dset['train_full'], 
                                                                                             train_ix, 
                                                                                             vali_ix, 
                                                                                             'inner', 
                                                                                             lead_time_)
            
            if dictionary['verbosity'] > 0: print('\n')
            j += 1 

        #---------------------------------------------------------------------------------------------------------------------------------------#        
        ## 5. Season selection
        for subset in ['train_full', 'test']:
            dset[subset] = select_season(dset[subset], subset)
        for subset in ['train', 'vali']:
            for key in dset[subset]:
                dset[subset][key] = select_season(dset[subset][key], subset)
      
        #---------------------------------------------------------------------------------------------------------------------------------------#    
        ## 6. Reference forecasts 
        persist, clim, ecmwf = compute_reference_forecasts(data, 
                                                           dset, 
                                                           start_date, 
                                                           end_date, 
                                                           tg_name, 
                                                           lead_time_, 
                                                           mid_end_date['train_full'], 
                                                           mid_start_date['train_full'])
                

        #---------------------------------------------------------------------------------------------------------------------------------------#      
        ## 7. Split up the data in target (y) and predictors (X): do it for train_full, train, vali, and test sets        
        ### Drop time index
        dset['train_full'] = dset['train_full'].reset_index().drop(columns = 'time', axis = 1)   
        for key in dset['train']:
            dset['train'][key] = dset['train'][key].reset_index().drop(columns = 'time', axis = 1) 
            
        ### Split individual forecast
        for subset in ['train_full', 'test']:
            y[subset] = dset[subset][tg_name]
            X[subset] = dset[subset].drop(tg_name, axis = 1)
            
        ### Split ensemble of forecasts     
        for subset in ['train', 'vali']:
            for key in dset[subset]:
                y[subset][key] = dset[subset][key][tg_name]
                X[subset][key] = dset[subset][key].drop(tg_name, axis = 1)           
        
        ### Save predictor names to variable
        feature_names = list(X['train_full'])

        ### Save ground truth time series and index to file
        save_time_series(y['test'], tg_name, 'GT', 'test', '', lead_time_, outer_split_num_ = i)
        save_time_series(y['test'].index, tg_name, 'index', 'test', '', lead_time_, outer_split_num_ = i)

        #---------------------------------------------------------------------------------------------------------------------------------------#    
        #---------------------------------------------------------------------------------------------------------------------------------------#  
        # Prediction
        pred_algorithm(X, y, 
                       persist, clim, ecmwf, 
                       start_date['test'],
                       predictor_names, feature_names, tg_name, 
                       lead_time_,
                       i)         

        # Increase outer split number 
        i += 1
