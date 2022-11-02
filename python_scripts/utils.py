#!/usr/bin/env python
# coding: utf-8

# # Utils

# In[1]:


# Import packages
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import Audio, display
import os, inspect
from datetime import datetime
import matplotlib.colors as colors
import itertools


# In[2]:


# Import constants
from ipynb.fs.full.const import dictionary


# In[33]:


# Define text format class
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


# In[ ]:


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
        


# In[3]:


# Notify after running a cell
def allDone():

    """
    inputs
    ------
    None


    outputs
    -------
    None. Makes notification sound 
    
    """
    
    display(Audio(url = 'https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand3.wav', autoplay = True))


# In[ ]:


def interpolate(df, new_index):
    
    """
    
    Returns a new DataFrame with all columns values interpolated
    to the new_index values
    
    """
    
    df_out = pd.DataFrame(index = new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out


# In[4]:


def find_nearest(array, value):
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    return array[idx]


# In[5]:


# Convert list to string
def list_to_string(s): 
    
    """
    inputs
    ------
    s       list : -


    outputs
    -------
    s'       str : with same content than s
    
    """
    
    ## Initialize an empty string
    str1 = " " 
    
    ## Return string  
    return (str1.join(s))


# In[6]:


def print_all(x):
    
    """
    inputs
    ------
    x       np.array : -


    outputs
    -------
    None. Prints all entries in the array even if it's long
    
    """
    
    for i in range (0,len(x)):
        print(x[i])


# In[86]:


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


# In[7]:


# Select summer months only (JJA)
def is_jja(month):

    """
    inputs
    ------
    month       int : number of month in the year 1-12


    outputs
    -------
    binary      int : 0 if month not in JJA
                      1 if month in JJA
    
    """
    
    return (month >= 6) & (month <= 8)


# In[8]:


# Select extended summer months only (MJJAS)
def is_mjjas(month):
    
    """
    inputs
    ------
    month       int : number of month in the year 1-12


    outputs
    -------
    binary      int : 0 if month not in MJJAS
                      1 if month in MJJAS
                      
    """
                      
    return (month >= 5) & (month <= 9)


# In[9]:


def compute_start_and_end_date(x):
    
    """
      inputs
      -------
        x                      pd.Dataframe or xr.Dataset  : containing time series


      outputs
      -------
        None. Prints start and end date of the time series. 

    """ 
    
    # For DataFrame
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        start_date = x.index[0]
        end_date = x.index[-1]
    
    # For Dataset
    elif isinstance(x, xr.DataArray) or isinstance(x, xr.Dataset):
        start_date = x.time[0]
        end_date = x.time[-1]
    
    if dictionary['verbosity'] > 2: print('Start_date: ', start_date,', end_date: ', end_date)        


# In[10]:


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


# In[7]:


def print_datetime(script_name):
    
    """
    inputs
    ------
    script_name          str : name of script   


    outputs
    -------
    None. Prints the highlighted name of the script and datetime when it started running
    
    """

    print(text_format.YELLOW_HIGHLIGHT + 'Output from ', script_name, ' run at ', datetime.now(), text_format.END)


# In[11]:


def print_title(tgn):
    
    """
    inputs
    ------
    tgn       str : name of target


    outputs
    -------
    None. Prints the highlighted name of the target
    
    """

    if tgn[-4:] == '2in7':
        print(text_format.RED_HIGHLIGHT + 'Summer binary heat wave index (at least 2 out of 7 days with HW)', text_format.END)
    elif tgn[-3:] == '1SD':
        print(text_format.RED_HIGHLIGHT + 'Summer binary heat wave index (weekly t2m_eobs > 1SD)', text_format.END)
    elif tgn[-4:] == '15SD':
        print(text_format.RED_HIGHLIGHT + 'Summer binary heat wave index (weekly t2m_eobs > 1.5SD)', text_format.END)
    elif tgn == 't2m':
        print(text_format.RED_HIGHLIGHT + 'Summer temperature', text_format.END)


# In[12]:


def print_balance(balance_type):
    
    """
    inputs
    ------
    balance_type       str : type of balance e.g. 'undersampling'


    outputs
    -------
    None. Prints the highlighted type of balance
    
    """
    
    print(text_format.BLUE_HIGHLIGHT + 'Balance: ' + balance_type, text_format.END)


# In[13]:


def print_lead_time(lead_time):

    """
    inputs
    ------
    lead_time       int : lead time of the prediction


    outputs
    -------
    None. Prints the highlighted lead time
    
    """

    print(text_format.TURQUOISE_HIGHLIGHT + 'Lead time: ' + str(lead_time) + ' week(s)', text_format.END)  


# In[14]:


def print_model_name(model_name):

    """
    inputs
    ------
    model_name       str : Machine Learning model


    outputs
    -------
    None. Prints the highlighted model name
    
    """


    print(text_format.GREY_HIGHLIGHT + model_name, text_format.END)


# In[15]:


def save_to_file(my_object, path, save_name, file_format):
    
    """
    inputs
    ------
    my_object               * : object to save 
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


# In[16]:


def find_var_name(var):

    """
    inputs
    ------
    var                  * : object 


    outputs
    -------
    var_name             * : name of object
    
    """
    
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


# In[87]:


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
        


# In[18]:


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


# In[19]:


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


# In[20]:


def flip_longitude_360_2_180(var_360, lon):
    
    """
    inputs
    ------
    var_380                        xr.Dataset : variable with longitude coordinates [0,360]
    lon                          xr.DataArray : longitude coordinates        
    
    outputs
    -------  
    var_180                        xr.Dataset : variable with longitude coordinates [-180,180]
    
    This function shifts the longitude dimension form [0,360] to [-180,180]
    """
    
    var_180 = var_360.assign_coords({'longitude': lon.where(lon <= 180, lon - 360)})
    var_180 = var_180.sortby(var_180.longitude)
    
    return var_180 


# In[32]:


def format_fn_y(tick_val, tick_pos):
    
    if int(tick_val) in range(N):
        return predictor_names[int(tick_val)]
    else:
        return ''


# In[6]:


# Hatch non-significant grid cells
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


# In[7]:


# Set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    
    def __init__(self, vmin = None, vmax = None, midpoint = None, clip = False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# In[5]:


def save_time_series(x, tgn, time_series_type, subset, _lead_time_, outer_split_num_ = None):
    
    """
    inputs
    ------
    x                        pd.Series : time series to be saved
    tgn                            str : name of target
    time_series_type               str : 'GT', 'persistence', a certain ML forecast etc.
    subset                         str : 'vali', 'train', 'train_full', 'test'
    _lead_time_                    int : lead time of prediction in weeks
    " outer_split_num_             int : counter for outer splits "

    
    outputs
    -------
    None                                 Saves the time series x to file
    
    """    
    
    # Reminder: for this to work, need to "Restart Kernel and Clear All Outputs" first
    ## Specify directory
    path = dictionary['path_time_series'] + tgn + '/'
    if outer_split_num_ != None:
        if time_series_type == 'GT':
            save_name = tgn + '_' + time_series_type + '_' + subset + '_outer_split_' + str(outer_split_num_) + '.npy'
        else:
            save_name = tgn + '_' + time_series_type + '_' + subset + '_lead_time_' + str(_lead_time_) + '_weeks_outer_split_' + str(outer_split_num_) + '.npy'
    else:
        if time_series_type == 'GT':
            save_name = tgn + '_' + time_series_type + '_' + subset + '.npy'
        else:
            save_name = tgn + '_' + time_series_type + '_' + subset + '_lead_time_' + str(_lead_time_) + '_weeks.npy'   
        
    save_to_file(x, path, save_name, 'np')
    


# In[5]:


def read_old_regr_ml_forecasts(tgn, _lead_time_, outer_split_num_ = None):
    
    """
    inputs
    ------
    tgn                            str : name of target
    _lead_time_                    int : lead time of prediction in weeks
    " outer_split_num_             int : counter for outer splits. This argument is optional. "

    
    outputs
    -------
    pred_mlr                      dict : forecast by MLR model for train_full and test
    " pred_mlr_ensemble           dict : ensemble of forecasts for uncertainty estimation by MLR model for train_full and test (only for no CV case) "
    pred_rfr                      dict : forecast by RFR model for train_full and test
    " pred_rfr_ensemble           dict : ensemble of forecasts for uncertainty estimation by RFR model for train_full and test (only for no CV case) "
    
    """   
    
    # Specify directory
    path = dictionary['path_time_series'] + tgn + '/'
    # Make combinations of models and subsets
    if dictionary['cv_type'] == 'none':
        models = ['MLR', 'MLR_ensemble', 'RFR', 'RFR_ensemble']
    elif dictionary['cv_type'] == 'nested':
        models = ['MLR', 'RFR']
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
        
    # Adapt format
    pred_mlr = {'train_full': d['mlr_train_full'], 'test': d['mlr_test']}
    pred_rfr = {'train_full': d['rfr_train_full'], 'test': d['rfr_test']}
    if dictionary['cv_type'] == 'none':
        pred_mlr_ensemble = {'train_full': d['mlr_ensemble_train_full'], 'test': d['mlr_ensemble_test']}
        pred_rfr_ensemble = {'train_full': d['rfr_ensemble_train_full'], 'test': d['rfr_ensemble_test']}
        
    if dictionary['cv_type'] == 'none': return pred_mlr, pred_mlr_ensemble, pred_rfr, pred_rfr_ensemble
    elif dictionary['cv_type'] == 'nested': return pred_mlr, {'train_full': None, 'test': None}, pred_rfr, {'train_full': None, 'test': None}


# In[6]:


def read_old_classi_ml_forecasts(tgn, _lead_time_, balance_sign_, outer_split_num_ = None):
    
    """
    inputs
    ------
    tgn                            str : name of target
    _lead_time_                    int : lead time of prediction in weeks
    balance_sign_                  str : '' (none), '+' (oversampled) or '-' (undersampled)
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
    elif dictionary['cv_type'] == 'nested': pred_rc, {'train_full': None, 'test': None}, pred_proba_rc, {'train_full': None, 'test': None}, pred_rfc, {'train_full': None, 'test': None}, pred_proba_rfc, {'train_full': None, 'test': None}


# In[ ]:




