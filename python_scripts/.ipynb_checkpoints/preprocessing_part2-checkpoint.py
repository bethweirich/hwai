""" HWAI preprocessing part 2 """

# Author: Elizabeth Weirich Benet 
# License: MIT License 2022

# Import packages
import xarray as xr
import pandas as pd
import numpy as np
import random 

# Import own functions
## Utils
from utils import force_split_out_of_selected_season

# Import constants
from const import dictionary


# **----------------------------------------------------------------------------------------------------------------------------------------------**

# Function definitions 

def bootstrap_ensemble(x):
    
    """
      inputs
      -------
        x                      pd.Dataframe : containing full time period


      outputs
      -------
        x_bootstrap              dictionary : each bootstrap_i corresponds to a choice of samples with replacement of size len(x) randomly drawn from x
        x_oob                    dictionary : each bootstrap_i contains the out of bag (OOB) years that were not used for the corresponding choice of samples

    """ 
    
    if dictionary['verbosity'] > 1: print('************************************* Creating bootstrapping ensemble *************************************')  

    # Make list of years
    all_years = list(np.arange(x.index.year[0], x.index.year[-1] + 1, 1))
    
    # Add year column
    x = x.assign(year = x.index.year)
    
    # Initialize
    x_bootstrap, x_oob = [{} for i in range(2)]
    if dictionary['full_run']: ensemble_size = 600
    else: ensemble_size = 2
    
    for i in np.arange(ensemble_size):
        ## Bootstrap sample with replacement of size len(x)
        years_bootstrap = random.choices(all_years, k = len(all_years))
        ## Out of bag years
        years_oob = list(set(all_years) - set(years_bootstrap))
        ## Dataframes
        key = 'bootstrap_{}'.format(i + 1)    
        x_bootstrap[key] = pd.DataFrame({'year': years_bootstrap, 'order': range(len(years_bootstrap))})
        x_bootstrap[key] = pd.merge(x_bootstrap[key], x, on = 'year').drop('order', axis = 1).drop(columns = 'year')
        x_oob[key] = x[x['year'].isin(years_oob)].drop(columns = 'year')
  
    return x_bootstrap, x_oob




def loo_ensemble_rdm(x):
    
    """
      inputs
      -------
        x                      pd.Dataframe : containing full time period


      outputs
      -------
        x_loo                    dictionary : each loo_i that contains 30 random possibilities to leave-9-years-out from x
        x_oob                    dictionary : each loo_i contains the 9 out of bag (OOB) years that were not used for the corresponding choice of samples in x_loo

    """ 
    
    if dictionary['verbosity'] > 1: print('************************************* Creating random 30-members leave-9-years-out ensemble *************************************')  

    # Make list of years
    all_years = list(np.arange(x.index.year[0], x.index.year[-1] + 1, 1))
    
    # Add year column
    x = x.assign(year = x.index.year)
    
    # Initialize
    x_loo, x_oob = [{} for i in range(2)]
    if dictionary['full_run']: ensemble_size = 30
    else: ensemble_size = 2
    
    for i in np.arange(ensemble_size):
        ## Define key name
        key = 'loo_{}'.format(i + 1)   
        ## Remove 9 random years
        remove_years = random.sample(all_years, 9)
        ## Leave-9-out (len(train_full - 9 years))        
        x_loo[key] = x[~x['year'].isin(remove_years)].drop(columns = 'year')
        ## Out-of-bag (9 years)
        x_oob[key] = x[x['year'].isin(remove_years)].drop(columns = 'year')   
       
    return x_loo, x_oob




def select_season(x, slice_name):
    
    """
      inputs
      -------
        x                      pd.Dataframe or xr.Dataset : containing full year
        slice_name                                    str : name of data slice (e.g., 'test', 'train', 'all')


      outputs
      -------
        seasonal_x                           pd.Dataframe : containing selected months of each year

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





def split_sets(data, slice_type, lead_time):
    
    """
      inputs
      -------
        data           pd.Dataframe or xr.Dataset : time series for overall period  and all features
        slice_type                            str : select whether you want the train (works only if giving train_full), vali (works only if giving train_full),
                                                    train_full (works only if giving full dataset) or test data (works only if giving full dataset)
        lead_time                             int : lead time of prediction in units of timestep

      outputs
      -------
        data_slice     pd.Dataframe or xr.Dataset : time series for selected time period (slice)
        start_date_corrected                  str : date when the selected slice starts (yyyy-mm-dd)
        end_date_corrected                    str : date when the selected slice end (yyyy-mm-dd)

    """  
    
    if dictionary['verbosity'] > 1: print('************** Enter split_sets for', slice_type, ' ******************') 

    # Compute start_year and end_year
    start_year = int(dictionary['splits'].sel(slice_type = slice_type, edge = 'start'))    
    end_year = int(dictionary['splits'].sel(slice_type = slice_type, edge = 'end'))  
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




def split_sets_nested_cv(df, set1_ix, set2_ix, split_type, lead_time):
    
    """
      inputs
      -------
        df             pd.Dataframe or xr.Dataset : time series for overall period and all features
        set1_ix                              list : range of int indices created via kFold.split that characterise set1
        set2_ix                              list : range of int indices created via kFold.split that characterise set2
        split_type                            str : 'inner' or 'outer' -> refers to which loop of the nested CV is considered 
        lead_time                             int : lead time of prediction in units of timestep

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

    return set1, set2, start_date_set1_corrected, end_date_set1_corrected, start_date_set2_corrected, end_date_set2_corrected, mid_end_date_set1_corrected, mid_start_date_set1_corrected



def lagged_features(data, target_name, lead_time):
    
    """
      inputs
      -------
        data         xarray.Dataset : time series from all variables (non-lagged)
        target_name             str : name of target variable 
        lead_time               int : lead time of prediction in units of timestep

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

    if dictionary['verbosity'] > 4: print('Targ_pred columns:', list(targ_pred))
    
    return targ_pred

