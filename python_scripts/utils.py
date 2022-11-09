""" HWAI utils """

# Author: Elizabeth Weirich Benet 
# License: MIT License 2022

# Import packages
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import matplotlib.colors as colors
import itertools
import os
from scipy.stats import norm

# Import constants
from const import dictionary


# **----------------------------------------------------------------------------------------------------------------------------------------------**
# Class definitions 

class text_format:
    
    ## Colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    TURQUOISE = '\033[96m'

    ## Highlights
    RED_HIGHLIGHT = '\033[1;30;41m'
    GREEN_HIGHLIGHT = '\033[1;30;42m'
    YELLOW_HIGHLIGHT = '\033[1;30;43m'
    BLUE_HIGHLIGHT = '\033[1;30;44m'
    PURPLE_HIGHLIGHT = '\033[1;30;45m'
    TURQUOISE_HIGHLIGHT = '\033[1;30;46m'

    GREY_HIGHLIGHT = '\033[1;37;40m'
    
    ## Formats
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    ## End
    END = '\033[0m'
    
    

# Set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    
    def __init__(self, vmin = None, vmax = None, midpoint = None, clip = False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip = None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    

# **----------------------------------------------------------------------------------------------------------------------------------------------**
# Function definitions 

def traverse(d):
    
    """
    inputs
    ------
    d               dict : dictionary      


    outputs
    -------
    product of all possible combinations of the dict elements
    
    """
       
    K,V = zip(*d.items())
    for v in itertools.product(*(v if isinstance(v,list) else traverse(v) for v in V)):
        yield dict(zip(K,v))




def interpolate(df, new_index):
    
    """
    inputs
    ------
    df               pd.DataFrame : original dataframe      
    new_index                       new index to interpolate df


    outputs
    -------
    df               pd.DataFrame : new dataframe with values interpolated to new_index  
    
    """
    
    df_out = pd.DataFrame(index = new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out


def force_split_out_of_selected_season(date_str, data):
    
    """
    inputs
    ------
    date_str               str : date that is possibly inside the selected season
    data          pd.Dataframe : data to be split


    outputs
    -------
    newdate_str            str : date with month closest to date_str but outside the selected season
    
    """
    
    if dictionary['verbosity'] > 4: print('Before forcing: ', date_str)
    if date_str != None:
        # Force the split to be in the non-selected season
        date = datetime.strptime(date_str, '%Y-%m-%d')
        if date.month >= dictionary['initial_month'] and date.month <= dictionary['final_month']: 
            ## If closer to inital month, force to first month BEFORE selected season
            newdate = date.replace(day = 1)
            if date.month - dictionary['initial_month'] < dictionary['final_month'] - date.month:
                newdate = newdate.replace(month = dictionary['initial_month'] - 1)
            ## If closer to final month, force to first month AFTER selected season    
            else:
                newdate = newdate.replace(month = dictionary['final_month'] + 1)
                
            # Make sure that the new_date is inside the dataframe's limits
            ## Find dataframe's limits
            min_date = data.index[0]
            max_date = data.index[-1]
            if newdate < min_date: newdate = min_date
            elif newdate > max_date: newdate = max_date          
            
            newdate_str = str(newdate)[:10]    
        else: newdate_str = date_str
        if dictionary['verbosity'] > 4: print('After forcing: ', newdate_str)
        
        return newdate_str
    
    else: return None



def mask_months(month):
    
    """
    inputs
    ------
    month       int : number of month in the year 1-12


    outputs
    -------
    binary      int : 0 if month not in selected months
                int : 1 if month in selected months
                      
    """
                      
    return (month >= dictionary['initial_month']) & (month <= dictionary['final_month'])



def final_day_of_month(month):

    """
    inputs
    ------
    month       int : number of month


    outputs
    -------
    final_day   int : number of the last day of that month
    
    """
    
    # January, March, May, July, August, October, and December have 31 days
    if month in [1, 3, 5, 7, 8, 10, 12]: final_day = 31
    # April, June, September and November have 30 days
    elif month in [4, 6, 9, 11]: final_day = 30
    # February has 28-29 days
    elif month == 2: final_day = 28
    
    return final_day



def print_datetime(script_name):
    
    """
    inputs
    ------
    script_name          str : name of script   


    outputs
    -------
    None. Prints the highlighted name of the script and datetime when it started running
    
    """

    if dictionary['verbosity'] > 0: print(text_format.YELLOW_HIGHLIGHT + 'Output from ', script_name, ' run at ', datetime.now(), text_format.END)

    


def print_title(tgn):
    
    """
    inputs
    ------
    tgn       str : name of target


    outputs
    -------
    None. Prints the highlighted name of the target
    
    """

    if dictionary['verbosity'] > 0:
        if tgn[-4:] == '2in7':
            print(text_format.RED_HIGHLIGHT + 'Summer binary heat wave index (at least 2 out of 7 days with HW)', text_format.END)
        elif tgn[-3:] == '1SD':
            print(text_format.RED_HIGHLIGHT + 'Summer binary heat wave index (weekly t2m_eobs > 1SD)', text_format.END)
        elif tgn[-4:] == '15SD':
            print(text_format.RED_HIGHLIGHT + 'Summer binary heat wave index (weekly t2m_eobs > 1.5SD)', text_format.END)
        elif tgn == 't2m':
            print(text_format.RED_HIGHLIGHT + 'Summer temperature', text_format.END)



def print_lead_time(lead_time):

    """
    inputs
    ------
    lead_time       int : lead time of prediction in units of timestep


    outputs
    -------
    None. Prints the highlighted lead time
    
    """

    if dictionary['verbosity'] > 0: print(text_format.TURQUOISE_HIGHLIGHT + 'Lead time: ' + str(lead_time) + ' week(s)', text_format.END)  



def print_model_name(model_name):

    """
    inputs
    ------
    model_name       str : machine Learning model name


    outputs
    -------
    None. Prints the highlighted model name
    
    """


    if dictionary['verbosity'] > 0: print(text_format.GREY_HIGHLIGHT + model_name, text_format.END)



def save_to_file(my_object, path, save_name, file_format):
    
    """
    inputs
    ------
    my_object               * : object to save; must be compatible with the format (e.g., xr.Dataset can be saved to 'nc')
    path                  str : path to directoy
    save_name             str : file name 
    file_format           str : 'nc' (NetCDF file), 'np' (numpy file) or 'csv' (plain text file)


    outputs
    -------
    None. Saves the object to a file at file_name. Reminder: for this to work, need to "Restart Kernel and Clear All Outputs" first.
    
    """
    ## Create directory if non-existent
    if not os.path.isdir(path):
        os.makedirs(path)
        
    # Remove old existing file (if existent) to enable overwritting
    if os.path.isfile(path + save_name):
        os.remove(path + save_name)
        if dictionary['verbosity'] > 2: print('Old file removed')

    if file_format == 'nc':
        # Convert to netcdf4 and save
        ## Reorder variables in dataset
        my_object.to_netcdf(path + save_name,'w','NETCDF4')

    elif file_format == 'np':    
        np.save(path + save_name, np.asarray(my_object))  
        
    elif file_format == 'csv':    
        my_object.to_csv(path + save_name)  
    
    if dictionary['verbosity'] > 2: print('New file ', save_name, 'saved to ', path)




def compute_ones_percentage(binary_object, object_name, show_position):
    
    """
      inputs
      -------
      binary_object           xr.Dataset, xr.DataArray, or np.array : object containing only 0s and 1s
      object_name                                               str : object name
      show_position                                            bool : if True, positions of the ones are shown


      outputs
      -------
      None. Prints a the percentage of one's in the binary object. 

    """ 

    if isinstance(binary_object, pd.DataFrame) or isinstance(binary_object, xr.Dataset):
        print('The percentage of 1s in ', object_name, ' is: \n')
        for var in list(binary_object):
            ## Compute percentage of one's in binary matrix
            percentage_1s = float(binary_object[var].sum()/binary_object[var].size*100)
            print(var, ': %.2f' %(percentage_1s), '% \n')
            if show_position == True: print('Positions of the 1s: \n', np.where(binary_object[var] == 1))
    
    else:
        ## Compute percentage of one's in binary matrix
        percentage_1s = float(binary_object.sum()/binary_object.size*100)
        print('The percentage of 1s in ', object_name, ' is: %.2f' %(percentage_1s), '%')
        if show_position == True: print('Positions of the 1s: \n', np.where(binary_object == 1))
        



def filter_dates_like(y, x):
 
    """
    inputs
    ------
    y                            pd.Series : full time series
    x                            pd.Series : discontinuous time series


    outputs
    -------
    y_filtered                   pd.Series : y time series filtered to have the same time entries than x
    
    """
    
    y_filtered = y.loc[x.index]
    y_filtered = y_filtered.astype(float)
    
    return y_filtered




def month_to_season(var_mon, sea):
    
    """
    inputs
    ------
    var_mon                        xr.Dataset : monthly variable
    sea                            str : season ('MAM', 'JJA', 'SON', 'DJF')


    outputs
    -------
    var_sea                        xr.Dataset : seasonal variable
    
    """
    
    # Mask other months with nan
    a = var_mon.where(var_mon['time.season'] == sea)
    # Rolling mean -> only Jan is not nan
    # however, we loose Jan/ Feb in the first year and Dec in the last
    b = a.rolling(min_periods = 1, center = True, time = 3).mean()
    # Take annual mean
    var_sea = b.groupby('time.year').mean('time')
    
    return var_sea



def flip_longitude_360_2_180(var_360, lon):
    
    """
    inputs
    ------
    var_380                        xr.Dataset : variable with longitude coordinates [0,360]
    lon                          xr.DataArray : longitude coordinates        
    
    outputs
    -------  
    var_180                        xr.Dataset : variable with longitude coordinates [-180,180]
    
    This function shifts the longitude dimension form [0,360] to [-180,180].
    
    """
    
    var_180 = var_360.assign_coords({'longitude': lon.where(lon <= 180, lon - 360)})
    var_180 = var_180.sortby(var_180.longitude)
    
    return var_180 



# Format ticks
def format_fn_y(tick_val, tick_pos):
    
    if int(tick_val) in range(N):
        return predictor_names[int(tick_val)]
    else:
        return ''



# Hatch non-significant cells
def hatch_non_significant_cells(ax, p_matrix, target_pos, tau_max):
    
    ## Define significance level
    alpha_level = 0.05
    ## Compute mask-type DataArray (1: hatch (non-significant, p_value > alpha_level), 0: don't hatch (significant, p_value < alpha_level))
    hatch = np.ma.masked_less(p_matrix, alpha_level)
    ## Integrate hatching into plot
    #h_data = np.flip(hatch[:, target_pos, 1:tau_max + 1], axis = 1)
    h_data = hatch[:, target_pos, 1:tau_max + 1]
    h = ax.pcolor(h_data, hatch = '///', color = 'grey',
                  facecolor = 'None', 
                  edgecolor = 'w', cmap = None, 
                  alpha = 0.5)
    
        
    
def standardize_data(dset):

        
    """
    inputs
    ------
    dset                 xr.Dataset : original dataset
    

    outputs
    -------   
    dset_std             xr.Dataset : standardized dataset

    
    """ 

    if dictionary['verbosity'] > 1: print('*************************** Standardizing the data *****************************')

    # Daily standard deviations
    dset_daily_std = dset.groupby('time.dayofyear').std('time')
    
    # Standardization step
    dset_std = (dset.groupby('time.dayofyear'))/dset_daily_std

    ## Drop extra coordinates
    dset_std = dset_std.drop(['dayofyear'])
    
    ## Replace infinite values with 0 (i.e., no anomaly)
    dset_std = dset_std.where(abs(dset_std) != np.inf).fillna(0.)
        
    return dset_std


def compute_hw_bin_sd(t2m_anom, sd_threshold):

        
    """
    inputs
    ------
    t2m_anom                       xr.DataArray : time series of detrended 2m air temperature anomalies w.r.t. climatology in CE
    sd_threshold                          float : threshold for index definition (e.g., 1.5, to mark as heatwaves the entries with 
                                                  temperature anomalies above 1.5 standard deviations)


    outputs
    -------   
    hw_bin_sd                      xr.DataArray : binary single measurement (extreme temperature only) weekly heat wave index.
                                                  0: no heatwave
                                                  1: heatwave
    
    """  

    if dictionary['verbosity'] > 1: print('************************************** Computing the hw_bin_' + str(sd_threshold) + 'SD index ****************************************')

    if isinstance(t2m_anom, xr.DataArray):
        ## Calculate standard deviation
        (mu, sigma) = norm.fit(t2m_anom)
        ## Output: binary array (1: the mean T anomaly (w.r.t. daily climatology) of the week is above sd_threshold, 0: else)
        hw_bin_sd = xr.where(t2m_anom > sd_threshold * sigma, 1, 0).load() 
        if dictionary['verbosity'] > 2: compute_ones_percentage(hw_bin_sd, 'hw_bin_sd', False)
    
    elif isinstance(t2m_anom, xr.Dataset):
        ## Calculate standard deviation
        hw_bin_sd = []
        for var in list(t2m_anom):
            ## Calculate standard deviation
            (mu, sigma) = norm.fit(t2m_anom[var])
            ## Output: binary dataset (1: the mean T anomaly (w.r.t. daily climatology) of the week is above sd_threshold, 0: else)            
            hw_bin_sd.append(xr.where(t2m_anom[var] > sd_threshold * sigma, 1, 0).load())
        hw_bin_sd = xr.merge(hw_bin_sd)
        if dictionary['verbosity'] > 2: compute_ones_percentage(hw_bin_sd, 'hw_bin_sd', False)
        
    return hw_bin_sd



def save_time_series(x, tgn, time_series_type, subset, pred_type_add, _lead_time_, outer_split_num_ = None):
    
    """
    inputs
    ------
    x                        pd.Series : time series to be saved
    tgn                            str : name of target
    time_series_type               str : 'GT', 'persistence', a certain ML forecast etc.
    subset                         str : 'vali', 'train', 'train_full', 'test'
    pred_type_add                  str : '_proba' if it saves the probabilistic classification forecast, '', otherwise
    _lead_time_                    int : lead time of prediction in units of timestep
    " outer_split_num_             int : counter for outer splits (only for nested CV case) "

    
    outputs
    -------
    None. Saves the time series x to a file.
    
    """    
    
    # Specify directory
    path = dictionary['path_time_series'] + subset + '/' + tgn + '/'
    
    # Specify file name
    if outer_split_num_ != None:
        if time_series_type in ['GT', 'index']:
            save_name = tgn + '_' + time_series_type + '_' + subset + pred_type_add + '_outer_split_' + str(outer_split_num_) + '.npy'
        else:
            save_name = tgn + '_' + time_series_type + '_' + subset + pred_type_add + '_lead_time_' + str(_lead_time_) + '_weeks_outer_split_' + str(outer_split_num_) + '.npy'
    else:
        if time_series_type in ['GT', 'index']:
            save_name = tgn + '_' + time_series_type + '_' + subset + pred_type_add + '.npy'
        else:
            save_name = tgn + '_' + time_series_type + '_' + subset + pred_type_add + '_lead_time_' + str(_lead_time_) + '_weeks.npy' 
            
    # Save    
    save_to_file(x, path, save_name, 'np')
    



def read_old_regr_ml_forecasts(tgn, _lead_time_, outer_split_num_ = None):
    
    """
    inputs
    ------
    tgn                            str : name of target
    _lead_time_                    int : lead time of prediction in units of timestep
    " outer_split_num_             int : counter for outer splits (only for nested CV case) "

    
    outputs
    -------
    pred_rr                       dict : regression forecast by RR model for train_full and test
    " pred_rr_ensemble            dict : ensemble of regression forecasts for uncertainty estimation by RR model for train_full and test (only for no CV case) "
    pred_rfr                      dict : regression forecast by RFR model for train_full and test
    " pred_rfr_ensemble           dict : ensemble of regression forecasts for uncertainty estimation by RFR model for train_full and test (only for no CV case) "
    
    """   
    

    # Make combinations of models and subsets
    if dictionary['cv_type'] == 'none':
        models = ['RR', 'RR_ensemble', 'RFR', 'RFR_ensemble']
    elif dictionary['cv_type'] == 'nested':
        models = ['RR', 'RFR']
    subsets = ['test', 'train_full']
    combinations = list(itertools.product(subsets, models))
    # Run over all combinations and load numpy files
    d = {}
    for i in range(len(combinations)):
        subset = combinations[i][0]
        model = combinations[i][1]
        # Specify directory and file
        path = dictionary['path_time_series'] + subset + '/' + tgn + '/'
        file = path + tgn + '_' + model + '_' + subset + '_lead_time_' + str(_lead_time_) 
        if dictionary['cv_type'] == 'nested': file = file + '_weeks_outer_split_' + str(outer_split_num_) + '.npy'
        elif dictionary['cv_type'] == 'none': file = file + '_weeks.npy'
        # Load
        forecast = np.load(file)
        model_lc = str.lower(model)
        d[model_lc + '_' +  subset] = forecast
        
    # Adapt format
    pred_rr = {'train_full': d['rr_train_full'], 'test': d['rr_test']}
    pred_rfr = {'train_full': d['rfr_train_full'], 'test': d['rfr_test']}
    if dictionary['cv_type'] == 'none':
        pred_rr_ensemble = {'train_full': d['rr_ensemble_train_full'], 'test': d['rr_ensemble_test']}
        pred_rfr_ensemble = {'train_full': d['rfr_ensemble_train_full'], 'test': d['rfr_ensemble_test']}
        
    if dictionary['cv_type'] == 'none': return pred_rr, pred_rr_ensemble, pred_rfr, pred_rfr_ensemble
    elif dictionary['cv_type'] == 'nested': return pred_rr, {'train_full': None, 'test': None}, pred_rfr, {'train_full': None, 'test': None}




def read_old_classi_ml_forecasts(tgn, _lead_time_, outer_split_num_ = None):
    
    """
    inputs
    ------
    tgn                            str : name of target
    _lead_time_                    int : lead time of prediction in units of timestep
    " outer_split_num_             int : counter for outer splits (only for nested CV case) "

    
    outputs
    -------
    pred_rc                       dict : binary classification forecast by RC model for train_full and test
    " pred_rc_ensemble            dict : ensemble of binary classification forecasts for uncertainty estimation by RC model for train_full and test (only for no CV case) "
    pred_proba_rc                 dict : probabilistic classification forecast by RC model for train_full and test
    " pred_proba_rc_ensemble      dict : ensemble of probabilistic classification forecasts for uncertainty estimation by RC model for train_full and test (only for no CV case) "
    pred_rfc                      dict : binary classification forecast by RFC model for train_full and test
    " pred_rfc_ensemble           dict : ensemble of binary classification forecasts for uncertainty estimation by RFC model for train_full and test (only for no CV case) "
    pred_proba_rfc                dict : probabilistic classification forecast by RFC model for train_full and test
    " pred_proba_rfc_ensemble     dict : ensemble of probabilistic classification forecasts for uncertainty estimation by RFC model for train_full and test (only for no CV case) "
    
    """    

    
    # Make combinations of models and subsets
    if dictionary['cv_type'] == 'none': models = ['RC', 'RC_ensemble', 'RFC', 'RFC_ensemble']
    elif dictionary['cv_type'] == 'nested': models = ['RC', 'RFC']
    pred_types = ['_proba', '']
    subsets = ['test', 'train_full']
    combinations = list(itertools.product(subsets, pred_types, models))
    # Run over all combinations and load numpy files
    d = {}
    for i in range(len(combinations)):
        subset = combinations[i][0]
        pred_type  = combinations[i][1]
        model = combinations[i][2]
        # Specify directory and file
        path = dictionary['path_time_series'] + subset + '/' + tgn + '/'
        file = path + tgn + '_' + model + '_' + subset + pred_type + '_lead_time_' + str(_lead_time_) 
        if dictionary['cv_type'] == 'nested': file = file + '_weeks_outer_split_' + str(outer_split_num_) + '.npy'
        elif dictionary['cv_type'] == 'none': file = file + '_weeks.npy'
        # Load
        forecast = np.load(file)
        model_lc = str.lower(model)
        d[model_lc + '_' +  subset + pred_type] = forecast   

    # Adapt format
    pred_rc = {'train_full': d['rc_train_full'], 'test': d['rc_test']}
    pred_proba_rc = {'train_full': d['rc_train_full_proba'], 'test': d['rc_test_proba']}
    pred_rfc = {'train_full': d['rfc_train_full'], 'test': d['rfc_test']}
    pred_proba_rfc = {'train_full': d['rfc_train_full_proba'], 'test': d['rfc_test_proba']}
    if dictionary['cv_type'] == 'none':
        pred_rc_ensemble = {'train_full': d['rc_ensemble_train_full'], 'test': d['rc_ensemble_test']}
        pred_proba_rc_ensemble = {'train_full': d['rc_ensemble_train_full_proba'], 'test': d['rc_ensemble_test_proba']}
        pred_rfc_ensemble = {'train_full': d['rfc_ensemble_train_full'], 'test': d['rfc_ensemble_test']}  
        pred_proba_rfc_ensemble = {'train_full': d['rfc_ensemble_train_full_proba'], 'test': d['rfc_ensemble_test_proba']}
    

    if dictionary['cv_type'] == 'none': return pred_rc, pred_rc_ensemble, pred_proba_rc, pred_proba_rc_ensemble, pred_rfc, pred_rfc_ensemble, pred_proba_rfc, pred_proba_rfc_ensemble
    elif dictionary['cv_type'] == 'nested': return pred_rc, {'train_full': None, 'test': None}, pred_proba_rc, {'train_full': None, 'test': None}, pred_rfc, {'train_full': None, 'test': None}, pred_proba_rfc, {'train_full': None, 'test': None}




