#!/usr/bin/env python
# coding: utf-8

# # Preprocessing part 2

# In[ ]:


# Import packages
import xarray as xr
import pandas as pd
import numpy as np
import random 

## Models 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

## Data balance
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

## Dimensionality reduction
from sklearn import preprocessing as prepro_sklearn
from sklearn.decomposition import PCA as PCA_model


# In[3]:


# Import own functions
from ipynb.fs.defs.utils import save_time_series, force_split_out_of_selected_season
from ipynb.fs.defs.plotting import plot_var_expl, show_coeffs, show_rf_imp


# In[4]:


# Import constants
from ipynb.fs.full.const import dictionary


# In[ ]:


def loo_ensemble_rdm(x):
    
    """
      inputs
      -------
        x                      pd.Dataframe : containing full time period


      outputs
      -------
        x_loo                    dictionary : dictionary with keys loo_i that contains all possibilities to leave-5-years-out from x in 600 randomly selected different ways

    """ 
    
    if dictionary['verbosity'] > 1: print('************************************* Creating random leave-5-years-out ensemble *************************************')  

    # Make list of years
    all_years = list(np.arange(x.index.year[0], x.index.year[-1] + 1, 1))
    # Add year column
    x = x.assign(year = x.index.year)
    # Initialize
    x_loo = {}
    j = 1
    for subset in np.arange(2):
        remove_years = random.sample(all_years, 5)
        key = 'loo_{}'.format(j)    
        x_loo[key] = x[~x['year'].isin(remove_years)].drop(columns = 'year')
        j = j + 1
           
    return x_loo


# In[2]:


def select_season(x, slice_name):
    
    """
      inputs
      -------
        x                      pd.Dataframe or xr.Dataset : containing full year
        slice_name                                    str : name of data slice (e.g., 'test', 'train', 'all')


      outputs
      -------
        seasonal_x             pd.Dataframe : containing selected months of each year

    """ 
    
    if dictionary['verbosity'] > 1: print('************************************* Selecting season for ', slice_name, '*************************************')   

    # For DataFrame
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        seasonal_x = x.loc[x.index.month <= dictionary['final_month']]
        seasonal_x = seasonal_x.loc[dictionary['initial_month'] <= seasonal_x.index.month]
        if dictionary['verbosity'] > 2: print('Length before: ', len(x.index), 'and after season selection: ', len(seasonal_x.index))
        if dictionary['verbosity'] > 4: print('Selected season slice is: ', seasonal_x.index)
    
    # For Dataset
    elif isinstance(x, xr.DataArray) or isinstance(x, xr.Dataset):
        seasonal_x = x.where(x.time.dt.month <= dictionary['final_month'], drop = True)
        seasonal_x = seasonal_x.where(dictionary['initial_month'] <= seasonal_x.time.dt.month, drop = True)
        if dictionary['verbosity'] > 2: print('Length before: ', len(x.time), 'and after season selection: ', len(seasonal_x.time))
        if dictionary['verbosity'] > 4: print('Selected season slice is: ', seasonal_x.time)
     
    return seasonal_x


# In[ ]:


def split_sets(data, slice_type, lead_time):
    
    """
      inputs
      -------
        data           pd.Dataframe or xr.Dataset : time series for overall period  and all features
        slice_type                            str : select whether you want the train (works only if giving train_full), vali (works only if giving train_full),
                                                    train_full (works only if giving full dataset) or test data (works only if giving full dataset)
        lead_time                             int : number of timesteps between each predictor value and the target

      outputs
      -------
        data_slice     pd.Dataframe or xr.Dataset : time series for selected time period (slice)
        start_date_corrected                  str : date when the selected slice starts (yyyy-mm-dd)
        end_date_corrected                    str : date when the selected slice end (yyyy-mm-dd)

    """  
    
    if dictionary['verbosity'] > 1: print('************** Enter split_sets for', slice_type, ' ******************') 

    # Compute start_year and end_year
    start_year = int(dictionary['splits'].sel(dataset = str(dictionary['dataset']), slice_type = slice_type, edge = 'start'))    
    end_year = int(dictionary['splits'].sel(dataset = str(dictionary['dataset']), slice_type = slice_type, edge = 'end'))  
    # Find start_date and end_date   
    ## For DataFrame
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.loc[(data.index.year >= start_year) & (data.index.year <= end_year)]
        start_date = data.index[0]
        end_date = data.index[-1]
    ## For Dataset
    elif isinstance(x, xr.DataArray) or isinstance(x, xr.Dataset):                      
        start_date = data.sel(time = str(start_year)).time[0]
        end_date = data.sel(time = str(end_year)).time[-1]
        ## Correct format
        start_date, end_date = str(start_date.time.values)[:10], str(end_date.time.values)[:10]
    if dictionary['verbosity'] > 2: print(slice_type, ' has start date, end date: ', start_date, ', ', end_date)

    ## Correct dates of target to account for corners (missing predictor data), intersection (overlapp of train_full & test sets
    ## -> data leakage issue)
    start_date_corrected = (pd.to_datetime(start_date) + pd.DateOffset(days = dictionary['timestep_num'] * (lead_time + dictionary['num_lags']))).strftime('%F')
    end_date_corrected = (pd.to_datetime(end_date)).strftime('%F')
    if dictionary['verbosity'] > 2: print(slice_type, ' has corrected start date, corrected end date: ' + start_date_corrected + ', ' + end_date_corrected)

    # Cut slice
    ## For DataFrame
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame): 
        ### Find corresponding index   
        start = data.index.searchsorted(pd.to_datetime(start_date_corrected))
        end = data.index.searchsorted(pd.to_datetime(end_date_corrected))
        ### Cut
        data_slice = data[start:end + 1]
    ## For Dataset
    elif isinstance(x, xr.DataArray) or isinstance(x, xr.Dataset): data_slice = data.sel(time = slice(start_date_corrected, end_date_corrected))

    # Show info    
    ## Save index
    ## For DataFrame
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame): index = data_slice.index
    ## For Dataset
    elif isinstance(x, xr.DataArray) or isinstance(x, xr.Dataset): index = data_slice.time
    ## Print info
    if dictionary['verbosity'] > 4: 
            if slice_type == 'train_full' or slice_type == 'test':
                print('Length of ', slice_type, ' set after train_full-test (outer) split: ', len(index)) 
            elif slice_type == 'train' or slice_type == 'vali':
                print('Length of ', slice_type, ' set after train-vali (inner) split: ', len(index)) 
            print(slice_type, ' set time index: ', index)
        
    
    return data_slice, start_date_corrected, end_date_corrected


# In[1]:


def split_sets_nested_cv(df, set1_ix, set2_ix, split_type, lead_time):
    
    """
      inputs
      -------
        df             pd.Dataframe or xr.Dataset : time series for overall period and all features
        set1_ix                              list : range of int indices created via kFold.split that characterise set1
        set2_ix                              list : range of int indices created via kFold.split that characterise set2
        split_type                            str : 'inner' or 'outer' -> refers to which loop of the nested CV is considered 
        lead_time                             int : number of timesteps between each predictor value and the target

      outputs
      -------
        set1                         pd.Dataframe : time series for selected time period (slice) for set1
        set2                         pd.Dataframe : time series for selected time period (slice) for set2
        start_date_set1_corrected             str : date when the selected slice for set1 starts (yyyy-mm-dd)
        mid_end_date_set1_corrected           str : date when the 1st part of the selected slice for set1 ends (yyyy-mm-dd)   
        mid_start_date_set1_corrected         str : date when the 2nd part of the selected slice for set1 starts (yyyy-mm-dd)
        end_date_set1_corrected               str : date when the selected slice for set1 ends (yyyy-mm-dd)
        start_date_set2_corrected             str : date when the selected slice for set2 starts (yyyy-mm-dd)
        end_date_set2_corrected               str : date when the selected slice for set2 ends (yyyy-mm-dd)

    """  
    
    if dictionary['verbosity'] > 1: print('************** Enter nested CV split_sets ******************') 
    
    
    # Compute names of sets
    if split_type == 'outer': set1_name = 'train_full'; set2_name = 'test'
    elif split_type == 'inner': set1_name = 'train'; set2_name = 'vali'
    
    # Compute preliminary sets
    pset1, pset2 = df.iloc[set1_ix], df.iloc[set2_ix]
    # Find jump in index (only relevant for train_full)
    ix_jump = [i for i in range(len(set1_ix) - 1) if set1_ix[i] != set1_ix[i + 1] - 1]
    # Extract start and end dates
    ## Set 1
    start_date_set1 = str(pset1.index[0])[:10]
    end_date_set1 = str(pset1.index[-1])[:10]
    if len(ix_jump) != 0:
        mid_end_date_set1 = str(pset1.index[ix_jump[0]])[:10]
        mid_start_date_set1 = str(pset1.index[ix_jump[0] + 1])[:10]
        if dictionary['verbosity'] > 2: print(set1_name, ' has start date, mid end date, mid start date, and end date: ', start_date_set1, ', ', mid_end_date_set1, ', ', mid_start_date_set1, ', ', end_date_set1)
    else: 
        mid_end_date_set1 = mid_start_date_set1 = None
        if dictionary['verbosity'] > 2: print(set1_name, ' has start date, and end date: ', start_date_set1, ', ', end_date_set1)
    ## Set 2
    start_date_set2 = str(pset2.index[0])[:10]
    end_date_set2 = str(pset2.index[-1])[:10]
    if dictionary['verbosity'] > 2: print(set2_name, ' has start date, end date: ', start_date_set2, ', ', end_date_set2)
    
    # Force splits out of selected season
    start_date_set1 = force_split_out_of_selected_season(start_date_set1, df)
    end_date_set1 = force_split_out_of_selected_season(end_date_set1, df)
    mid_end_date_set1 = force_split_out_of_selected_season(mid_end_date_set1, df)
    mid_start_date_set1 = force_split_out_of_selected_season(mid_start_date_set1, df)
    start_date_set2 = force_split_out_of_selected_season(start_date_set2, df)
    end_date_set2 = force_split_out_of_selected_season(end_date_set2, df)
    
    # Correct dates to account for corners (missing predictor data), intersection (overlapp of train_full & test sets
    # -> data leakage issue)
    ## Set 1
    start_date_set1_corrected = (pd.to_datetime(start_date_set1) + pd.DateOffset(days = dictionary['timestep_num'] * (lead_time + dictionary['num_lags']))).strftime('%F')
    end_date_set1_corrected = (pd.to_datetime(end_date_set1)).strftime('%F')
    if len(ix_jump) != 0:
        mid_end_date_set1_corrected = (pd.to_datetime(mid_end_date_set1)).strftime('%F')
        mid_start_date_set1_corrected = (pd.to_datetime(mid_start_date_set1) + pd.DateOffset(days = dictionary['timestep_num'] * (lead_time + dictionary['num_lags']))).strftime('%F')
        if dictionary['verbosity'] > 2: print(set1_name, ' has corrected start date, corrected mid end date, corrected mid start date, and corrected end date: ', start_date_set1_corrected, ', ', 
                                              mid_end_date_set1_corrected, ', ', mid_start_date_set1_corrected, ', ', end_date_set1_corrected)
    else: 
        mid_end_date_set1_corrected = mid_start_date_set1_corrected = None
        if dictionary['verbosity'] > 2: print(set1_name, ' has corrected start date, corrected end date: ', start_date_set1_corrected, ', ', end_date_set1_corrected)
    ## Set 2
    start_date_set2_corrected = (pd.to_datetime(start_date_set2) + pd.DateOffset(days = dictionary['timestep_num'] * (lead_time + dictionary['num_lags']))).strftime('%F')
    end_date_set2_corrected = (pd.to_datetime(end_date_set2)).strftime('%F')
    if dictionary['verbosity'] > 2: print(set2_name, ' has corrected start date, corrected end date: ', start_date_set2_corrected, ', ', end_date_set2_corrected)

    # Cut slice
    ## Find corresponding index  
    ### Set 1
    start_set1 = df.index.searchsorted(pd.to_datetime(start_date_set1_corrected))
    end_set1 = df.index.searchsorted(pd.to_datetime(end_date_set1_corrected))

    if len(ix_jump) != 0:
        mid_end_set1 = df.index.searchsorted(pd.to_datetime(mid_end_date_set1_corrected))
        mid_start_set1 = df.index.searchsorted(pd.to_datetime(mid_start_date_set1_corrected))
    ### Set 2
    start_set2 = df.index.searchsorted(pd.to_datetime(start_date_set2_corrected))
    end_set2 = df.index.searchsorted(pd.to_datetime(end_date_set2_corrected))
    ## Cut
    ### Set 1
    if len(ix_jump) != 0:
        set1_part1 = df[start_set1:mid_end_set1 + 1]
        set1_part2 = df[mid_start_set1:end_set1 + 1]
        set1 = pd.concat([set1_part1, set1_part2], axis = 0) 
    else: set1 = df[start_set1:end_set1 + 1]
    ### Set 2
    set2 = df[start_set2:end_set2 + 1]
    
    #if mid_end_date_set1_corrected is not None and mid_start_date_set1_corrected is not None:
        #return set1, set2, start_date_set1_corrected, end_date_set1_corrected, start_date_set2_corrected, end_date_set2_corrected, mid_end_date_set1_corrected, mid_start_date_set1_corrected
    #else: return set1, set2, start_date_set1_corrected, end_date_set1_corrected, start_date_set2_corrected, end_date_set2_corrected

    return set1, set2, start_date_set1_corrected, end_date_set1_corrected, start_date_set2_corrected, end_date_set2_corrected, mid_end_date_set1_corrected, mid_start_date_set1_corrected


# In[9]:


def lagged_features(data, target_name, lead_time):
    
    """
      inputs
      -------
        data         xarray.Dataset : time series from all variables (non-lagged)
        target_name             str : name of target variable 
        lead_time               int : number of timesteps between each predictor value and the target

      outputs
      -------
        targ_pred      pd.DataFrame : target and predictors with coordinate time of target


    """  
    
    if dictionary['verbosity'] > 1: print('************** Enter lagged_features ******************') 

    # Predictors
    ## Initialize targ_pred with the target only
    targ_pred = data[target_name].to_dataframe()
    if dictionary['verbosity'] > 4: print('Index for targ_pred right after being initialized :', targ_pred.index, 'with len :', len(targ_pred.index)) 
    ## Save time index
    time_index = targ_pred.index
    ## Save start and end date
    start_date = time_index[0]
    end_date = time_index[-1]
    ## Create a full set of lagged predictors for each lag time
    ### Relative lag (rel_lag) = lag - lead time i.e. for lead time 1, lag 1 has rel_lag 0
    for rel_lag in range (0, dictionary['num_lags']):
        ## Compute offset days
        offset_days = dictionary['timestep_num'] * (lead_time + rel_lag)
        ## Find start and end date for lagged predictors
        start_date_lag = (pd.to_datetime(start_date) - pd.DateOffset(days = offset_days)).strftime('%F')       
        end_date_lag = (pd.to_datetime(end_date) - pd.DateOffset(days = offset_days)).strftime('%F')
        ## Extract selected time series slice from predictors
        X_lag = data.drop(target_name)
        X_lag = X_lag.sel(time = slice(start_date_lag, end_date_lag)).to_dataframe()       
        ## Add the corresponding overall lag i.e. lead time (LT) + relative lag (0 - num_lags) to the predictor variable names
        x_lag_names = list(X_lag)
        for name in x_lag_names:
            X_lag = X_lag.rename(columns = {name : name + '_' + dictionary['timestep'] + '_lag' + str(lead_time + rel_lag)})           
        ## Combine all lagged predictors and the target into a matrix 
        ### e.g. for lag_num = 4: targ_pred = ([target: {Y = X_0}, predictors: {X_res_LT, X_res_LT+1, X_res_LT+2, X_res_LT+3}])
        cut_time_index = time_index[-len(X_lag.index):]
        X_lag = X_lag.reset_index(drop = True)
        X_lag = X_lag.set_index(cut_time_index)
        targ_pred = pd.concat([targ_pred, X_lag], axis = 1)
        
    if dictionary['all_lags_and_means'] == True:
        ### Add soil moisture and SSTs at coarser time resolutions
        ## Extract soil moisture and SST variable names 
        sm_sst_var_names = [var_name for var_name in list(data) if 'sm' in var_name or 'sst' in var_name]
        ## Extract selected time series slice from predictors
        X = targ_pred.drop(columns = [target_name])
        ## Extract soil moisture and SSTs predictors
        sm_sst_cols = [col for col in X.columns if 'sm' in col or 'sst' in col]
        X_sm_sst = X.filter(sm_sst_cols)
        for res_factor in [2,3,4]:
            if dictionary['verbosity'] > 4: print('res_factor: ', res_factor)
            ## Take means of res_factor columns and make lag_num/res_factor new columns containing means 
            for rel_lag in range (0, int(dictionary['num_lags']/res_factor)):
                if dictionary['verbosity'] > 4: print('rel_lag: ', rel_lag)
                lags_to_mean = [lead_time + res_factor * rel_lag + i for i in range(0, res_factor)]
                for var_name in sm_sst_var_names:                
                    cols_to_mean = [var_name + '_' + dictionary['timestep'] + '_lag' + str(lag) for lag in lags_to_mean]
                    if dictionary['verbosity'] > 4: print('cols_to_mean: ', cols_to_mean)
                    X_sm_sst[var_name + '_' + str(dictionary['timestep_num'] * res_factor) + dictionary['timestep'][-1] + '_lag' + str(lead_time + rel_lag)] = X_sm_sst[cols_to_mean].mean(axis = 1)
        ## Combine all lagged predictors and the target into a matrix 
        ### e.g. for lag_num = 4: targ_pred = ([target: {Y = X_0}, predictors: {X_res_factor x res_LT, X_factor x res_LT+1}])
        targ_pred = pd.concat([targ_pred.drop(columns = sm_sst_cols), X_sm_sst], axis = 1)

    if dictionary['verbosity'] > 4: print('Targ_pred columns:', list(targ_pred))
    
    return targ_pred


# In[20]:


def data_balance(balance_type, data, target_name, lagged_variable_names, show = True): 
    
    """
    inputs
    ------
    balance_type              str : specifies whether under-/ or oversampling should be performed
    data             pd.Dataframe : contains time series for target and predictors at lagged times
    target_name               str : name of target variable 
    variable_names    list of str : names of all variables in data
    show                     bool : if True, print comments

    outputs
    -------
    data             pd.Dataframe : balanced time series for target and predictors
    
    """
    
    if dictionary['verbosity'] > 1 and show is True: print('*********************** Enter data_balance ************************') 
    ## Split dataset into target and features
    target = data[target_name].values
    features = data.drop(target_name, axis = 1).to_xarray().to_array().transpose()
    if dictionary['verbosity'] > 2 and show is True: print('Before (class index, number of samples): ', sorted(Counter(target).items()))

    ## Controlled undersampling step
    if balance_type == 'undersampling':
        rus = RandomUnderSampler(random_state = 0)
        features_resampled, target_resampled = rus.fit_resample(features, target)
        if dictionary['verbosity'] > 2 and show is True: print('...Undersampling data...')
            
    ## Controlled oversampling step
    elif balance_type == 'oversampling':
        ros = RandomOverSampler(random_state = 0)
        features_resampled, target_resampled = ros.fit_resample(features, target)
        if dictionary['verbosity'] > 2 and show is True: print('...Oversampling data...')
    if dictionary['verbosity'] > 2 and show is True: print('After (class index, number of samples): ', sorted(Counter(target_resampled).items()))
        
    ## Dataframe out of resampled data in chronological order
    data_resampled = np.column_stack((target_resampled.transpose(), features_resampled))
    df_resampled = pd.DataFrame(data = data_resampled[:,:],    
          columns = lagged_variable_names).sort_index()  
    
    ## Prints 
    if dictionary['verbosity'] > 2 and show is True:
        print('Length before and after data balance: ', len(data), 'and:', len(df_resampled))
    if dictionary['verbosity'] > 3 and show is True: 
        print('# 0\'s before data balance: ', len(np.where(data[target_name] == 0)[0]))
        print('# 1\'s before data balance: ', len(np.where(data[target_name] == 1)[0]))
        print('# 0\'s after data balance: ', len(np.where(df_resampled[target_name] == 0)[0]))
        print('# 1\'s after data balance: ', len(np.where(df_resampled[target_name] == 1)[0]))            
    if dictionary['verbosity'] > 4 and show is True: print('Balanced dataframe: \n', df_resampled)
        
    return df_resampled     


# In[ ]:


def rf_feature_selection(yX_train_full, yX_test, tgn):

    """
    Feature selection that selects the minimum number of features whose Random Forest importances add up to sum_importances (numeric value specified in dictionary)
    
    inputs
    ------

    yX_train_full                          pd.Dataframe : target and lagged predictor features for the train_full time period
    yX_test                                pd.Dataframe : target and lagged predictor features for the test time period
    tgn                                             str : name of target variable 

    outputs
    -------
    yX_train_full_selected                 pd.Dataframe : target and lagged predictor features for the train_full time period after feature selection
    yX_test_selected                       pd.Dataframe : target and lagged predictor features for the test time period after feature selection
    
    """
    
    if dictionary['verbosity'] > 1: print('************** Feature selection with Random Forest ****************')
    
    # Split up data into target and features
    X_train_full = yX_train_full.drop(tgn, axis = 1)
    y_train_full = yX_train_full[tgn]
    
    # Run RF on train_full with default parameters
    ## Set default parameters
    n_estimators = 100
    max_depth = None
    min_samples_split = 2
    min_samples_leaf = 1
    min_samples_leaf = int(len(X_train_full)/100)
    ## Initialize model
    if 'bin' in tgn:                 
        rf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, random_state = 6, criterion = 'gini', n_jobs = dictionary['n_cores'])
    elif 'bin' not in tgn:
        rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, random_state = 6, criterion = 'mse', n_jobs = dictionary['n_cores'])
    ## Train model
    rf.fit(X_train_full, y_train_full)
    
    # Keep features whose importances sum sum_importances & drop the rest   
    ## Feature names
    feature_names = list(X_train_full)
    ## Get feature importances
    importances = list(rf.feature_importances_)
    feature_importances = [(feature, importance) for feature, importance in zip(feature_names, importances)]
    ## Sort feature importances
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    sorted_features, sorted_importances = zip(*feature_importances)
    ## Find minimum number of features whose importances sum 0.9
    sum_imps = 0  
    num_sel_features = 0
    while sum_imps < dictionary['min_sum_importances']:
        sum_imps = sum_imps + sorted_importances[num_sel_features]
        num_sel_features += 1
    ## Select one more to make sure the importances sum at least sum_importances
    selected_features = list(sorted_features[:num_sel_features + 1])        
    if dictionary['verbosity'] > 1: print('The ', num_sel_features + 1, ' most important features are: ', selected_features, ' and their importances add up to: ', round(sum_imps, 3))      
    ## Extract selected features from train_full and test datasets and recombine target and features (y & X = yX)
    ### Train_full
    X_train_full_selected = X_train_full[selected_features]
    yX_train_full_selected = X_train_full_selected.assign(tg_name = y_train_full.values)
    yX_train_full_selected = yX_train_full_selected.rename(columns = {'tg_name': tgn})
    first_column = yX_train_full_selected.pop(tgn)
    yX_train_full_selected.insert(0, tgn, first_column)
    ### Test
    yX_test_selected = yX_test[[tgn] + selected_features]
        
    return yX_train_full_selected, yX_test_selected


# In[ ]:


def PCA(yX_train_full, yX_test, tgn, lead_time, outer_split_num__ = None):
    

    """
    Dimensionality reduction via Principal Component Analysis
    
    inputs
    ------

    yX_train_full                          pd.Dataframe : target and lagged predictor features for the train_full time period
    yX_test                                pd.Dataframe : target and lagged predictor features for the test time period
    tgn                                             str : name of target variable 
    lead_time                                       int : number of timesteps between each predictor value and the target
    " outer_split_num__                             int : counter for outer splits "

    outputs
    -------
    yX_train_full_pc                       pd.Dataframe : target and lagged predictor features for the train_full time period after feature selection
    yX_test_pc                             pd.Dataframe : target and lagged predictor features for the test time period after feature selection
    
    """
    
    if dictionary['verbosity'] > 1: print('************** Dimensionality reduction via PCA ****************')
    
    # Split up data into target and features
    X_train_full = yX_train_full.drop(tgn, axis = 1)
    y_train_full = yX_train_full[tgn]
    X_test = yX_test.drop(tgn, axis = 1)
    y_test = yX_test[tgn]
    
    # Standarize data
    sc = prepro_sklearn.StandardScaler()
    X_train_full_std = sc.fit_transform(X_train_full)
    X_test_std = sc.transform(X_test)
    
    #PCA
    ## PCA on features from the full training set
    pca1 = PCA_model()
    pca1.fit(X_train_full_std)
    var_exp = pca1.explained_variance_ratio_
    print('Explained variance: ', np.round(var_exp, 3).tolist())
    cum_var_exp = np.cumsum(var_exp)
    print('Cumulative explained variance: ', np.round(cum_var_exp, 3).tolist())
    ## Find number of PCs neeeded (k) to get the min_cum_exp_var defined in the dictionary
    for i in range (np.size(cum_var_exp)): 
        if (cum_var_exp[i] >= dictionary['min_cum_exp_var']): 
            k = i + 1
            break
    ## PCA for dimensionality reduction n to k on features from the full training set
    pca = PCA_model(n_components = k)
    X_train_full_pc = pca.fit_transform(X_train_full_std)    
    print('The ', k, ' selected principal components have a cumulative explained variance of: ', round(cum_var_exp[k -1], 3))
    if outer_split_num__ is not None: plot_var_expl(var_exp, k, tgn, lead_time, outer_split_num_ = outer_split_num__)
    else: plot_var_expl(var_exp, k, tgn, lead_time)
    ## Project eigenvectors from PCA on the test set
    X_test_pc = pca.transform(X_test_std)
    
    # Construct dataframes & recombine target and features (y & X = yX)
    ## Train_full
    X_train_full_pc = pd.DataFrame(data = X_train_full_pc, index = y_train_full.index, columns = ['PC{}'.format(i) for i in range(1, k + 1)])
    yX_train_full_pc = X_train_full_pc.assign(tg_name = y_train_full.values)
    yX_train_full_pc = yX_train_full_pc.rename(columns = {'tg_name': tgn})
    first_column = yX_train_full_pc.pop(tgn)
    yX_train_full_pc.insert(0, tgn, first_column)
    ### Test
    X_test_pc = pd.DataFrame(data = X_test_pc, index = y_test.index, columns = ['PC{}'.format(i) for i in range(1, k + 1)])
    yX_test_pc = X_test_pc.assign(tg_name = y_test.values)
    yX_test_pc = yX_test_pc.rename(columns = {'tg_name': tgn})
    first_column = yX_test_pc.pop(tgn)
    yX_test_pc.insert(0, tgn, first_column)    
    
    
    return yX_train_full_pc, yX_test_pc

