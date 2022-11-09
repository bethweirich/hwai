""" HWAI reference forecasts """

# Author: Elizabeth Weirich Benet 
# License: MIT License 2022

# Import packages
import xarray as xr
import pandas as pd
import numpy as np

# Import own functions
## Utils
from utils import (compute_ones_percentage, 
                   save_time_series, 
                   standardize_data, 
                   compute_hw_bin_sd)
## Preprocessing
from preprocessing_part2 import select_season

# Import constants
from const import dictionary


# **----------------------------------------------------------------------------------------------------------------------------------------------**

# Function definitions 

def compute_persistence_forecast(data, slice_type, start_date, end_date, target_name, _lead_time_, mid_end_date = None, mid_start_date = None):
    
    """
    inputs
    ------
    data                          xr.Dataset : full initial dataset
    slice_type                           str : select whether you want the train (works only if giving train_full), vali (works only if giving train_full),
                                               train_full (works only if giving full dataset) or test data (works only if giving full dataset)
    start_date                           str : date when the selected slice starts
    end_date                             str : date when the selected slice ends
    target_name                          str : name of target variable 
    _lead_time_                          int : number of timesteps between each predictor value and the target
    " mid_end_date                       str : date when the 1st part of the selected slice ends "
    " mid_start_date                     str : date when the 2nd part of the selected slice starts "

    outputs
    -------
    persistence_forecast           pd.Series : time series of persistence forecast 
    
    """
    
    if dictionary['verbosity'] > 1: print('************** Computing persistence', slice_type, 'forecast ****************')
     
    # Save Ground Truth index
    if mid_end_date != None and mid_start_date != None:
        gt_index_part1 = data[target_name].sel(time = slice(start_date, mid_end_date)).time.values
        gt_index_part2 = data[target_name].sel(time = slice(mid_start_date, end_date)).time.values
        gt_index = np.concatenate((gt_index_part1, gt_index_part2), axis = 0)
        
    else: gt_index = data[target_name].sel(time = slice(start_date, end_date)).time.values
    
    # Shift start_date and end_date by shift = lead time
    start_date_shifted = (pd.to_datetime(start_date) - pd.DateOffset(days = dictionary['timestep_num'] * _lead_time_)).strftime('%F')
    end_date_shifted = (pd.to_datetime(end_date) - pd.DateOffset(days = dictionary['timestep_num'] * _lead_time_)).strftime('%F')
    if mid_end_date != None and mid_start_date != None:
        mid_end_date_shifted = (pd.to_datetime(mid_end_date) - pd.DateOffset(days = dictionary['timestep_num'] * _lead_time_)).strftime('%F')
        mid_start_date_shifted = (pd.to_datetime(mid_start_date) - pd.DateOffset(days = dictionary['timestep_num'] * _lead_time_)).strftime('%F')
   
    # Select slice for persistence forecast
    if mid_end_date != None and mid_start_date != None:
        persistence_forecast_part1 = data[target_name].sel(time = slice(start_date_shifted, mid_end_date_shifted))
        persistence_forecast_part2 = data[target_name].sel(time = slice(mid_start_date_shifted, end_date_shifted))
        persistence_forecast = xr.concat([persistence_forecast_part1, persistence_forecast_part2], dim = 'time')
    else: persistence_forecast = data[target_name].sel(time = slice(start_date_shifted, end_date_shifted))
    
    # Adapt index to match set
    persistence_forecast = persistence_forecast.to_dataframe().reset_index(drop = True).set_index(gt_index).squeeze()
    
    # Select season
    persistence_forecast = select_season(persistence_forecast, 'persistence_' + slice_type)
    
    ix = persistence_forecast.index
    ix_jump = [i for i in range(len(ix) - 1) if int(str(ix[i])[:4]) + 1 < int(str(ix[i + 1])[:4])]    
    if len(ix_jump) != 0:
        mid_end_date = str(persistence_forecast.index[ix_jump[0]])[:10]
        mid_start_date= str(persistence_forecast.index[ix_jump[0] + 1])[:10]    
    if dictionary['verbosity'] > 2:
        if mid_end_date != None and mid_start_date != None: 
            print('Persistence ', slice_type, ' has (start date, mid end date, mid start date, end date): (', 
                  persistence_forecast.index[0], ', ', mid_end_date, ', ', mid_start_date, ', ', persistence_forecast.index[-1], ')')
        else: print('Persistence ', slice_type, ' has (start date, end date): (', persistence_forecast.index[0], ', ', persistence_forecast.index[-1], ')')
    
    # Save persistence forecast time series 
    save_time_series(persistence_forecast, target_name, 'persistence', slice_type, '', _lead_time_)
        
    return persistence_forecast




def compute_climatology_forecast(y, climatology_doy, tgn):
    
    """
    inputs
    ------
    y                              pd.Series : target time series
    climatology_doy                np.darray : 365 entries representing the climatology of the target variable
    tgn                                  str : name of target variable 

    outputs
    -------
    climatology_forecast           pd.Series : time series of climatology forecast 
    
    """
    
    if dictionary['verbosity'] > 1: print('************** Computing climatology forecast ****************')
        
    ## Initialize arrays
    y_doy = np.zeros(len(y), dtype = int)
    if 'bin' not in tgn:
        climatology_forecast = np.zeros(len(y), dtype = float)
    else:
        climatology_forecast = np.zeros(len(y), dtype = int)
    ## Run through and fill with corresponding value    
    for i in range(len(y)):
        ### Extract day of year (doy)
        y_doy[i] = pd.to_datetime(y.index[i]).timetuple().tm_yday
        ### 1. Regression: find corresponding temperature value in the climatology
        if 'bin' not in tgn:
            climatology_forecast[i] = float(climatology_doy[y_doy[i]])
        ### 2. Classification: fill climatology_forecast where 1's predominate in the doy's week's mean with 1's 
        ### -> get binary array representing majority class
        elif 'bin' in tgn:
            count_1_week = 0
            for j in np.where(climatology_doy[y_doy[i] - 3: y_doy[i] + 3].mean() >= 0.5)[0]:
                if dictionary['verbosity'] > 4: print(j)
                climatology_forecast[j] = 1
                count_1_week += 1
    if 'bin' in tgn and dictionary['verbosity'] > 4: print('# Entries with 1 being the majority class of the week: ', count_1_week)   
    
    ## Adapt index to match corresponding set
    climatology_forecast = pd.DataFrame(data = climatology_forecast, index = y.index, columns = [tgn])[tgn]
    
    return climatology_forecast




def compute_ecmwf_forecast(tgn, _lead_time_, start_date_, end_date_, time_period):

    """
    inputs
    ------
    tgn                                  str : name of target variable 
    _lead_time_                          int : number of timesteps between each predictor value and the target
    start_date_                          str : start date of ECMWF forecast time series in the format yyyy:mm:dd
    end_date_                            str : end date of ECMWF forecast time series in the format yyyy:mm:dd
    time_period                          str : name of time period (e.g., 'test')


    outputs
    -------
    ecmwf_forecast                 pd.Series : time series of ECMWF forecast 
    
    """
    
    if dictionary['verbosity'] > 1: print('************** Computing ECMWF forecast ****************')
        
    # Load data
    access_ecmwf_mean_file = dictionary['path_data'] + 'ecmwf/ensemble_mean/' + 't2m_ecmwf_2000-05-01_2019-09-30_lead_time_' + str(_lead_time_) + '_weeks.nc'
    access_ecmwf_mems_file = dictionary['path_data'] + 'ecmwf/ensemble_members/' + 't2m_ecmwf_2000-05-01_2019-09-30_lead_time_' + str(_lead_time_) + '_weeks_mems.nc'    
    ecmwf_mean_data = xr.open_dataset(access_ecmwf_mean_file).load()
    ecmwf_mems_data = xr.open_dataset(access_ecmwf_mems_file).load()
    
    # Merge ensemble mean and ensemble members to one dataset and rename columns 
    ecmwf_mems_data = ecmwf_mems_data.rename(dict(zip(['2t']+['2t_' + str(x) for x in range(2,12)], [tgn + '_' + str(x) for x in range(1,12)])))
    ecmwf_data = xr.merge([ecmwf_mean_data.rename({'t2m_daily': tgn + '_mean'}), ecmwf_mems_data])
           
    # Prepro step 1 done
    ##     1. CE box already selected and avg
    ##     2. Vertical level already selected
    ##     3. Detrending not needed for such a short time period
    ##     4. Anomalies already provided
    ##     5. 7D smoothing already done
    
    # Standardize ECMWF data for SD classification indices
    ecmwf_data_std = standardize_data(ecmwf_data)
    
    # Make ECMWF index depending on the target name
    ## Make regression index
    if tgn == 't2m': ecmwf_index = ecmwf_data
    ## Make 1SD index
    elif tgn == 'hw_bin_1SD': ecmwf_index = compute_hw_bin_sd(ecmwf_data_std, 1)
    ## Make 1.5SD index
    elif tgn == 'hw_bin_15SD': ecmwf_index = compute_hw_bin_sd(ecmwf_data_std, 1.5)
    
    # Do the preprocessing steps that are common to all indices (prepro step 2)  
    # Select test time period (start_date_, end_date_)
    if dictionary['verbosity'] > 2: print('ECMWF start date: ', start_date_, ' and end date: ', end_date_)
    ecmwf_index = ecmwf_index.sel(time = slice(start_date_, end_date_))
    if 'bin' in tgn and dictionary['verbosity'] > 4: compute_ones_percentage(ecmwf_index, 'ECMWF weekly heat wave index', 0)
    
    # Season selection
    ecmwf_index = select_season(ecmwf_index, 'ecmwf_' + time_period)
    if 'bin' in tgn and dictionary['verbosity'] > 4: compute_ones_percentage(ecmwf_index, 'ECMWF weekly heat wave index', 0)    
    
    # Adapt format
    ecmwf_forecast = ecmwf_index.to_dataframe()
    
    ### Save ECMWF time series & index
    save_time_series(ecmwf_forecast, tgn, 'ECMWF', time_period, '', _lead_time_)
    save_time_series(ecmwf_forecast.index, tgn, 'ECMWF_index', time_period, '', _lead_time_)

    return ecmwf_forecast




def compute_reference_forecasts(data_, dset, start_date, end_date, tg_name, lead_time_, mid_end_date_train_full_ = None, mid_start_date_train_full_ = None):
    
    
    """
    inputs
    ------ 
    data_                                xr.Dataset : target + predictors in full time period and seasons
    dset                          dict of pd.Series : target (Ground Truth) 
    start_date                          dict of str : date when the selected slice starts (yyyy-mm-dd)
    end_date                            dict of str : date when the selected slice ends (yyyy-mm-dd)
    tg_name                                     str : name of target variable 
    lead_time_                                  int : lead time of prediction in units of timestep
   " mid_end_date_train_full_                   str : date when the 1st part of the selected slice for train_full ends (yyyy-mm-dd) "  
   " mid_start_date_train_full_                 str : date when the 2nd part of the selected slice for train_full starts (yyyy-mm-dd) "

    outputs
    -------
    persist_forecast_          dict of xr.DataArray : persistence forecast of target for full train and test time period
    clim_forecast_             dict of xr.DataArray : climatology forecast of target for full train and test time period
    ecmwf_forecast_            dict of xr.DataArray : ECMWF forecast of target for test time period
    
    """
    
    # 1. Persistence forecast
    persist_forecast_ = {} 
    ## Same length than test or train_full, but shifted by lead time)
    for subset in ['train_full', 'test']:
        if subset == 'train_full' and mid_end_date_train_full_ is not None and mid_start_date_train_full_ is not None:
            persist_forecast_['train_full'] = compute_persistence_forecast(data_, 'train_full', start_date['train_full'], end_date['train_full'], tg_name, lead_time_, mid_end_date_train_full_, mid_start_date_train_full_)
        else: 
            persist_forecast_[subset] = compute_persistence_forecast(data_, subset, start_date[subset], end_date[subset], tg_name, lead_time_)
    
    
    # 2. Climatology forecast
    # Load data from targets    
    access_targets_file = dictionary['path_data'] + 'targets_7D_rolling_mean.nc'
    targets = xr.open_dataset(access_targets_file).load().drop('dayofyear')
    ## Load data from clim
    clim = xr.open_dataset(dictionary['path_data'] + 'climatology.nc')
    ## 2.1. For regression: use only zeros since the variable are the standardized anomalies w.r.t. climatology 
    ## -> their climatology must be 0
    zero_clim = np.zeros(len(clim.t2m))
    if 'bin' not in tg_name: 
        clim_doy = zero_clim
    ## 2.2. For classification: use the class with higher cardinality (Remark: it's also usually 0)
    else: 
        ### Compute mean for each doy: continuous value in range [0,1]
        cont_clim_doy = targets[tg_name].groupby('time.dayofyear').mean('time')
        ### Take running mean
        cont_clim_doy = cont_clim_doy.pad(dayofyear = 15, mode = 'wrap').rolling(dayofyear = 31, center = True).mean().dropna(dim = 'dayofyear')
        ### Initialize with zeros
        clim_doy = zero_clim.astype(int)
        ### Fill doy's where 1's predominate with 1's -> get binary array representing majority class
        count_1_doy = 0
        for i in np.where(cont_clim_doy >= 0.5)[0]:
            if dictionary['verbosity'] > 4: print(i)
            clim_doy[i] = 1
            count_1_doy += 1
        if dictionary['verbosity'] > 4: print('# Entries with 1 being the majority class of the doy: ', count_1_doy)
    ## Compute climatology forecast  
    clim_forecast_ = {}
    clim_forecast_['train_full'] = compute_climatology_forecast(dset['train_full'][tg_name], clim_doy, tg_name)
    clim_forecast_['test'] = compute_climatology_forecast(dset['test'][tg_name], clim_doy, tg_name)
    ## Save climatology forecast time series
    for subset in ['train_full', 'test']:
        save_time_series(clim_forecast_[subset], tg_name, 'climatology', subset, '', lead_time_)
        if dictionary['verbosity'] > 4: print('Clim forecast', subset, ': ', clim_forecast_[subset])

        
    # 3. Dynamical model: ECMWF forecast
    if dictionary['cv_type'] != 'nested':
        ecmwf_forecast_test = compute_ecmwf_forecast(tg_name, lead_time_, start_date['test'], end_date['test'], 'test')
    else: ecmwf_forecast_test = None
    # Adapt format
    ecmwf_forecast_ = {'train_full': None, 'test': ecmwf_forecast_test}
    
    
    return persist_forecast_, clim_forecast_, ecmwf_forecast_
    

