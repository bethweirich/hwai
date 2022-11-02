#!/usr/bin/env python
# coding: utf-8

# # Preprocessing part 1

# In[1]:


# Import packages
import xarray as xr
from eofs.xarray import Eof
import numpy as np
import pandas as pd
import time
from scipy.stats import norm
from scipy import signal


# In[2]:


# Import own functions
from ipynb.fs.defs.utils import save_to_file, compute_ones_percentage, flip_longitude_360_2_180, month_to_season
from ipynb.fs.defs.plotting import show_multiple_time_series, show_individual_time_series, show_snao_sea_patterns, show_first_x_eofs


# In[3]:


# Import constants
from ipynb.fs.full.const import dictionary


# In[2]:


def import_data():
    
    """
    inputs
    ------
    None but the paths to the files with the raw data variables specified in Table 1 in [1] must be provided by the user.

    outputs
    -------
    t2m                     xr.DataArray : lat-lon grid x time series of 2m air temperature
    z                       xr.DataArray : lat-lon grid x time series of Geopotential at 500hPa  
    sm                      xr.Dataset   : 4 vertical levels x lat-lon grid x time series of Soil Water Volume Level
    sst                     xr.DataArray : lat-lon grid x time series of Sea Surface Temperature
    rain                    xr.DataArray : lat-lon grid x time series of total precipitation    
    
    """

    if dictionary['verbosity'] > 1: print('********************************************* Importing data ***********************************************')
     
    # Specify paths
    (access_z_eraint_file, 
     access_sm_era5land_file, 
     access_sst_hadisst_file, 
     access_rain_eobs_file, 
     access_t2m_eobs_file) = (
                                dictionary['path_eraint'] + 'phi/daymean/phi-eraint-37plevels-daymean-*',
                                dictionary['path_era5_land'] + 'sm/VSW_l1-4_1981-2019-daily.nc',
                                dictionary['path_hadisst'] + 'HadISST_monmean_1870-2019.nc',
                                dictionary['path_eobs'] + 'precip/daymean/rr_ens_mean_0.25deg_reg_v22.0e.nc',
                                dictionary['path_eobs'] + 't2m_mean/daymean/tg_ens_mean_0.25deg_reg_v22.0e.nc'
                            )

    # Load data
    ## Target
    t2m = xr.open_dataset(access_t2m_eobs_file).tg.load()    
    ## Predictors        
    (z, sm, sst, rain) =  (
                xr.open_mfdataset(access_z_eraint_file,combine = 'by_coords', parallel = True).phi.load().sel(level = dictionary['geopotential_level']),
                xr.open_dataset(access_sm_era5land_file).load(),
                xr.open_dataset(access_sst_hadisst_file).sst.load(),
                xr.open_dataset(access_rain_eobs_file).rr.load()
                            )
 
    return t2m, z, sm, sst, rain


# In[5]:


def compute_anomalies_wrt_climatology(dset):
    
    """
    inputs
    ------
    dset                 xr.Dataset : absolute values of each variable in their respective physical units
    

    outputs
    -------   
    dset_anom            xr.Dataset : anomalies w.r.t. climatology, smoothed out in time via a rolling mean
                                      
    
    """ 

    if dictionary['verbosity'] > 1: print('*************************** Computing anomalies w.r.t. climatology *****************************')

    # Daily climatology
    dset_clim = dset.groupby('time.dayofyear').mean('time')
    ## Smooth monthly
    dset_clim_sm = dset_clim.pad(dayofyear = 15, mode = 'wrap').rolling(dayofyear = 31, center = True).mean().dropna(dim = 'dayofyear')

    # Daymean anomalies
    dset_anom = dset.groupby('time.dayofyear') - dset_clim_sm
        
    return dset_anom  


# In[6]:


def compute_north_atlantic_pca(z500):
    
    """
    inputs
    ------    
    z500                              xr.DataArray : lat-lon grid x time series of Geopotential at 500hPa (1d resolution)     
    
    outputs
    -------
    snao_daily_std                    xr.DataArray : time series of Summer North Atlantic Oscillation
    sea_daily_std                     xr.DataArray : time series of Summer East Atlantic pattern 
    
    """


    # Geopotential data
    ## Convert from geopotential to height
    z500_height = z500/9.806 
    # Compute monthly mean
    z500_monthly = z500_height.resample(time = '1M').mean(dim = 'time')
    ## Seasonal summer (JJA) anomalies 
    z500_jja = month_to_season(z500_monthly, sea = 'JJA')
    z500_jja_anom = z500_jja - z500_jja.mean('year') 
    z500_jja_anom = z500_jja_anom.rename({'year': 'time'})
    ## Detrend 
    z500_jja_anom = z500_jja_anom.copy(data = signal.detrend(z500_jja_anom, axis = 0)) 
    ## Select the North Atlantic area
    z500_jja_anom = z500_jja_anom.sortby('latitude', ascending = False)
    z500_fliped =  flip_longitude_360_2_180(z500_jja_anom, z500_jja_anom.longitude)
    z500_JJA_na = z500_fliped.sel(latitude = slice(70,40),longitude = slice(-90, 30)) 

    # PCA
    num_eofs = 10
    ## Create a solver class, taking advantage of built-in weighting
    weights = np.sqrt(np.cos(np.deg2rad(z500_JJA_na.latitude))).squeeze()
    solver = Eof(z500_JJA_na.transpose('time', 'longitude', 'latitude'), center = True, weights = weights)
    ## Retrieve the first two EOFs from the solver class
    eofs = solver.eofs(neofs = num_eofs).transpose('mode', 'latitude', 'longitude')
    eofs_cov = solver.eofsAsCovariance(neofs = num_eofs, pcscaling = 1).transpose('mode', 'latitude', 'longitude')
    pcs = solver.pcs(npcs = num_eofs, pcscaling = 1)
    ## Calculate explained variance
    var_frac = solver.varianceFraction(neigs = num_eofs) 
    if dictionary['verbosity'] > 2: print('Variance fraction of ', num_eofs, ' first North Atlantic patterns EOFs is: \n', var_frac.values)
    ## Plot pattern of first 10 EOFs
    if dictionary['verbosity'] > 4: show_first_x_eofs(eofs, num_eofs)
    ## Attribute EOFs to Atlantic Patterns: 1st (mode = 0) to SNAO, 2nd (mode = 1) to SEA
    snao_eof = eofs.sel(mode = 0)
    sea_eof = eofs.sel(mode = 1)

    # Create a daily index based on the projection of the previous EOF patterns
    z500_daily_anom = compute_anomalies_wrt_climatology(z500_height)
    ## Select the North Atlantic area
    z500_daily_anom = z500_daily_anom.sortby('latitude', ascending = False)
    z500_fliped =  flip_longitude_360_2_180(z500_daily_anom, z500_daily_anom.longitude)
    z500_daily_na = z500_fliped.sel(latitude = slice(70,40), longitude = slice(-90,30)) 
    ## Projections (onto daily values)
    snao_daily = (z500_daily_na * snao_eof).mean('latitude').mean('longitude')
    sea_daily = (z500_daily_na * sea_eof).mean('latitude').mean('longitude')
    ## Standarize
    snao_daily_std = (snao_daily - snao_daily.mean('time')) / snao_daily.std('time')
    sea_daily_std = (sea_daily - sea_daily.mean('time')) / sea_daily.std('time')
    ## Drop extra dimensions
    (snao_daily_std, sea_daily_std) = (snao_daily_std.drop('mode'), sea_daily_std.drop('mode'))

    # Daily composite for verification
    z500_snao = z500_daily_na.where(snao_daily < 1.0).mean('time')
    z500_sea = z500_daily_na.where(sea_daily > 1.0).mean('time')
    
    # Show SNAO & SEA patterns
    if dictionary['verbosity'] > 2: show_snao_sea_patterns(snao_eof, sea_eof, z500_snao, z500_sea)
  
    return snao_daily_std, sea_daily_std


# In[4]:


def box_selection(t2m, z, sm, sst, rain):
    
    """
    inputs
    ------
    t2m                      xr.DataArray : lat-lon grid x time series of 2m air temperature (target & predictor)
    z                        xr.DataArray : lat-lon grid x time series of Geopotential at 500hPa (predictor)
    sm                       xr.Dataset   : 4 vertical levels x lat-lon grid x time series of Soil Water Volume Level (predictor)
    sst                      xr.DataArray : lat-lon grid x time series of Sea Surface Temperature (predictor)
    rain                     xr.DataArray : lat-lon grid x time series of total precipitation (predictor)

    outputs
    -------
    t2m_sa_                  xr.DataArray : time series of 2m air temperature in CE (target)   
    z_sa_                    xr.DataArray : time series of Geopotential in CE at 500hPa (predictor)
    sm_all_levels_sa_        xr.Dataset   : 4 vertical levels x time series of Soil Water Volume Level in CE (predictor)
    sst_nwmed_sa_            xr.DataArray : time series of Sea Surface Temperature in NWMED (predictor)
    sst_cnaa_sa_             xr.DataArray : time series of Sea Surface Temperature in CNAA (predictor)
    rain_sa_                 xr.DataArray : time series of total precipitation in CE (predictor)   
    
    sa = spatially averaged
    
    """ 
    
    if dictionary['verbosity'] > 1: print('******************************* Selecting boxes and spatially averaging data *******************************')

    # Select data in grid point boxes & average over each box
    ## CE for 2m Temperature, Geopotential, Precipitation and Soil Moisture 
    ## NWMED and CNAA for Sea Surface Temperature
    
    ## Target (& predictor)
    ### Temperature
    t2m_sa_ = t2m.sel(latitude = slice(dictionary['boxes'].sel(box='ce_obs',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce_obs',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce_obs',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce_obs',axis='longitude',edge='end'))).mean('latitude').mean('longitude')                   
    
    ## Predictors
    ### Geopotential, soil moisture, NWMED SST, CNAA SST, precipitation 
    (z_sa_, sm_all_levels_sa_, sst_nwmed_sa_, sst_cnaa_sa_, rain_sa_) =  (
        z.sel(latitude = slice(dictionary['boxes'].sel(box='ce',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
        sm.sel(latitude = slice(dictionary['boxes'].sel(box='ce',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
        sst.sel(latitude = slice(dictionary['boxes'].sel(box='nwmed',axis='latitude',edge='start'),dictionary['boxes'].sel(box='nwmed',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='nwmed',axis='longitude',edge='start'),dictionary['boxes'].sel(box='nwmed',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),       
        sst.sel(latitude = slice(dictionary['boxes'].sel(box='cnaa',axis='latitude',edge='start'),dictionary['boxes'].sel(box='cnaa',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='cnaa',axis='longitude',edge='start'),dictionary['boxes'].sel(box='cnaa',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),                                                                                                                            )
        rain.sel(latitude = slice(dictionary['boxes'].sel(box='ce_obs',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce_obs',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce_obs',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce_obs',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
                                                                            )
                                                                                
    return t2m_sa_, z_sa_, sm_all_levels_sa_, sst_nwmed_sa_, sst_cnaa_sa_, rain_sa_


# In[8]:


def select_vert_level(sm_all_levels_sa_):
    
    """
    inputs
    ------ 
    sm_all_levels_sa_                      xr.Dataset : 4 vertical levels x time series of Soil Water Volume Level in CE


    outputs
    -------
    sm_sa_                               xr.DataArray : time series of Soil Water Volume Level in CE at 0 - 28 cm depth
    
    """

    if dictionary['verbosity'] > 1: print('**************************************** Selecting vertical levels ******************************************')
    
    # Extract and combine soil moisture levels
    ## Level 1: 0 - 7 cm
    ## Level 2: 7 - 28 cm
    ## Level 3: 28 - 100 cm
    ## Level 4: 100 - 289 cm
    ### Combine the first two levels into one (sm)
    sm_ds = xr.Dataset({'sm1': sm_all_levels_sa_.swvl1, 'sm2': sm_all_levels_sa_.swvl2})
    sm_sa_ = sm_ds.to_array(dim ='new').mean('new')
    
    return sm_sa_


# In[9]:


def construct_dataset(t2m_sa_, z_sa_, sm_sa_, sea_sa_, sst_nwmed_sa_, sst_cnaa_sa_, rain_sa_):
    
    """
    inputs
    ------
    t2m_sa_                  xr.DataArray : time series of 2m air temperature in CE (target & predictor)
    z_sa_                    xr.DataArray : time series of Geopotential in CE at selected geopotential level (predictor)
    sm_sa_                   xr.DataArray : time series of Soil Water Volume Level in CE at 0 - 28 cm depth (predictor)
    sea_sa_                  xr.DataArray : time series of Summer East Atlantic (SEA) pattern (predictor)
    sst_nwmed_sa_            xr.DataArray : time series of Sea Surface Temperature in NWMED (predictor)
    sst_cnaa_sa_             xr.DataArray : time series of Sea Surface Temperature in CNAA (predictor)
    rain_sa_                 xr.DataArray : time series of total precipitation in CE (predictor) 
    

    outputs
    -------   
    dset_sa_                 xr.Dataset : dataset of spatially averaged variables 
   
    sa = spatially averaged
    
    """ 
    
    if dictionary['verbosity'] > 1: print('****************************************** Constructing dataset ********************************************')
    
    # Create dataset containing all spatially averaged monthly variables
    dset_sa_1 = xr.Dataset({ 
        'sst_nwmed': sst_nwmed_sa_,
        'sst_cnaa': sst_cnaa_sa_
                        }).resample(time = '1D')

    # Create dataset containing all spatially averaged daily variables & resample
    dset_sa_2 = xr.Dataset({       
        't2m': t2m_sa_,
        't2m_x': t2m_sa_,
        'z': z_sa_,
        'rain': rain_sa_,
        'sm': sm_sa_, 
        'sea': sea_sa_
                        }).resample(time = '1D').first()

    # Interpolate monthly values to daily ones for SST
    dset_sa_1 = dset_sa_1.interpolate('linear')

    # Merge two datasets
    dset_sa_ = xr.merge([dset_sa_2, dset_sa_1])
    ## Drop extra coordinates    
    dset_sa_ = dset_sa_.drop(['level'], errors = 'ignore')
    
    return dset_sa_


# In[11]:


def fit_lin_reg(X, Y):
    
    """
    inputs
    ------
    X           xr.DataArray : x-coordinate
    Y           xr.DataArray : function of x (y = f(x)) to be fitted    


    outputs
    -------
    a                  float : slope of linear regression
    b                  float : intercept of linear regression
    
    Returns a,b in solution to y = ax + b such that root mean square distance between trend line and original points is minimized
    
    """
    
    N = len(X)
    X = range(N)   
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx
    
    #Slope
    a = (Sxy * N - Sy * Sx)/det
    # Intercept
    b = (Sxx * Sy - Sx * Sxy)/det
    
    return a.values, b.values


# In[12]:


def detrend(dset, show_trends):
        
    """
    inputs
    ------
    dset                 xr.Dataset : dataset containing variables with long-term linear trend
    show_trends                bool : if True, shows linear trends for each variable in the dataset before and after the detrending
    

    outputs
    -------   
    dset                 xr.Dataset : dataset containing variables without long-term linear trend   
    
    """ 



    # Remove nan's
    dset = dset.dropna(dim = 'time', how = 'any', thresh = None)   
    var_names = list(dset)
    
    if show_trends == True:
        for var in var_names[1:]:    
            # Find linear trends in time series 
            a,b = fit_lin_reg(dset[var].time, dset[var])
            # Convert trend to decadal 
            a_dec = a * 365.25 * 10
            print(dictionary['long_predictor_names'][var], 'has slope: {:.2E}'.format(a_dec), dictionary['units'][var],'/decade and intercept: {:.2E}'.format(b), dictionary['units'][var])

    if dictionary['verbosity'] > 1: print('************************************* Linearly detrending the data *****************************************')
    for var in var_names: 
        # Remove linear trends (if present) from the time series
        ## Remove the effects of stochastic and deterministic trends from a data set to show only the fluctuations;
        ## Allow cyclical and other patterns to be identified     
        dset[var] = dset[var].copy(data = signal.detrend(dset[var], axis = 0)) 
        
    if show_trends == True:
        for var in var_names[1:]:    
            # Find linear trends in time series 
            a,b = fit_lin_reg(dset[var].time, dset[var])
            # Convert trend to decadal 
            a_dec = a * 365.25 * 10
            print(dictionary['long_predictor_names'][var], 'has slope: {:.2E}'.format(a_dec), dictionary['units'][var],'/decade and intercept: {:.2E}'.format(b), dictionary['units'][var])
    
    return dset


# In[13]:


def compute_std_climatology(dset):
    
    """
    inputs
    ------
    dset                 xr.Dataset : absolute values of each variable in their respective physical units
    

    outputs
    -------   
    dset_clim_std        xr.Dataset : standarized daily climatology (1 value for each day of the year x # vars)
    
    """ 

    if dictionary['verbosity'] > 1: print('*************************** Computing standarized climatology *****************************')

    # Daily standard deviations
    dset_daily_std = dset.groupby('time.dayofyear').std('time')
    
    # Daily climatology
    dset_clim = dset.groupby('time.dayofyear').mean('time')
    
    ## Smooth monthly
    dset_clim_sm = dset_clim.pad(dayofyear = 15, mode = 'wrap').rolling(dayofyear = 31, center=True).mean().dropna(dim = 'dayofyear')
    
    ## Standarized smoothed daily climatology
    dset_clim_std = dset_clim_sm/dset_daily_std
    
    ## Drop extra coordinates
    dset_clim_std = dset_clim_std.drop(['dayofyear'])    
        
    return dset_clim_std  


# In[14]:


def compute_climatology(dset):
    
    """
    inputs
    ------
    dset                 xr.Dataset : absolute values of each variable in their respective physical units
    

    outputs
    -------   
    dset_clim_sm         xr.Dataset : daily climatology (1 value for each day of the year x # vars)
    
    """ 

    if dictionary['verbosity'] > 1: print('*************************** Computing climatology *****************************')
    
    # Daily climatology
    dset_clim = dset.groupby('time.dayofyear').mean('time')
    
    ## Smooth monthly
    dset_clim_sm = dset_clim.pad(dayofyear = 15, mode = 'wrap').rolling(dayofyear = 31, center=True).mean().dropna(dim = 'dayofyear')
    
    ## Drop extra coordinates
    dset_clim_sm = dset_clim_sm.drop(['dayofyear'])    
        
    return dset_clim_sm


# In[17]:


def standarize_data(dset):

        
    """
    inputs
    ------
    dset                 xr.Dataset : original dataset
    

    outputs
    -------   
    dset_std             xr.Dataset : standarized dataset

    
    """ 

    if dictionary['verbosity'] > 1: print('*************************** Standarizing the data *****************************')

    # Daily standard deviations
    dset_daily_std = dset.groupby('time.dayofyear').std('time')
    
    # Standarization step
    dset_std = (dset.groupby('time.dayofyear'))/dset_daily_std

    ## Drop extra coordinates
    dset_std = dset_std.drop(['dayofyear'])
    
    ## Replace infinite values with 0 (i.e., no anomaly)
    dset_std = dset_std.where(abs(dset_std) != np.inf).fillna(0.)
        
    return dset_std
    


# In[18]:


def smooth(dset):
    
    """
    inputs
    ------
    dset                 xr.Dataset : dataset containing daily values for each variable
    

    outputs
    -------   
    dset_sm              xr.Dataset : dataset containing values smoothed out by the user-defined timestep    
    
    
    """ 

    if dictionary['verbosity'] > 1: print('******************************* Smoothing out the data via a centred', dictionary['timestep'], 
                                          dictionary['mean_type'] ,'mean **********************************')

    # Timestep
    ## Reduce to timestep resolution
    if dictionary['mean_type'] == 'simple':
        dset_sm = dset.resample(time = dictionary['timestep']).mean() 

    ## Keep resolution from dset and place the mean value centered at the middle of the timestep 
    ### e.g. for timestep = '7D', the mean of the week 1-7 is placed in position 4
    if dictionary['mean_type'] == 'rolling':
        dset_sm = dset.rolling(time = dictionary['timestep_num'], center = True).mean().dropna(dim = 'time')
        
    return dset_sm 


# In[24]:


def compute_hw_bin_sd(t2m_anom, sd_threshold):

        
    """
    inputs
    ------
    t2m_anom                       xr.DataArray or xr.Dataset : time series of detrended 2m air temperature anomalies w.r.t. climatology in CE


    outputs
    -------   
    hw_bin_sd                      xr.DataArray or xr.Dataset : binary single measurement (extreme temperature only) weekly heat wave index 
    
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


# In[25]:


def save_pred_and_targ(dset_anom_sa_, dset_clim_std_, t2m_, hw_bin_1SD_, hw_bin_15SD_):
    
    """
    inputs
    ------
    dset_anom_sa_              xr.Dataset : preprocessed spatially averaged predictor's anomalies (standarized and smoothed)
    dset_clim_std_             xr.Dataset : climatology for the target variable (standarized and smoothed)
    t2m_                     xr.DataArray : 2m temperature spatially averaged over CE (non-standarized, but smoothed)
    hw_bin_1SD_              xr.DataArray : binary single measurement (extreme temperature only) weekly heat wave index 
                                            (1: t2m anomalies > 1 standard deviation)
    hw_bin_15SD_             xr.DataArray : binary single measurement (extreme temperature only) weekly heat wave index 
                                            (1: t2m anomalies > 1.5 standard deviations)
    

    outputs
    -------   
    None. Saves predictors and targets to files. 
    
    """  

    if dictionary['verbosity'] > 1: print('************************************** Saving predictors and targets ******************************************')

    # Specify save names
    save_name_preprocessed_predictors = 'preprocessed_predictors' + dictionary['timestep'] + '_' + dictionary['mean_type'] + '_mean.nc'
    save_name_targets = 'targets_' + dictionary['timestep'] + '_' + dictionary['mean_type'] + '_mean.nc'   
    save_name_clim_std = 'climatology.nc'
    
    # Specify objects
    obj_sa = dset_anom_sa_[['t2m_x', 'z', 'rain', 'sm', 'sea', 'sst_nwmed', 'sst_cnaa']]
    obj_targets = xr.Dataset(data_vars={'t2m':(('time'), t2m_), 'hw_bin_1SD':(('time'), hw_bin_1SD_), 'hw_bin_15SD':(('time'), hw_bin_15SD_)}, 
                                coords={'time': (t2m_.time)})
    
    # Save
    save_to_file(obj_sa, dictionary['path_data'], save_name_preprocessed_predictors, 'nc')
    save_to_file(obj_targets, dictionary['path_data'], save_name_targets, 'nc')
    if dictionary['mean_type'] == 'rolling': save_to_file(dset_clim_std_, dictionary['path_data'], save_name_clim_std, 'nc')


# **----------------------------------------------------------------------------------------------------------------------------------------------**

# **----------------------------------------------------------------------------------------------------------------------------------------------**

# **----------------------------------------------------------------------------------------------------------------------------------------------**

# ## Main
# ### Data preprocessing (part 1)

# In[26]:


# 1. Import data
t2m_raw, z_raw, sm_raw, sst_raw, rain_raw = import_data()


# In[27]:


# 2. North Atlantic patterns
snao_sa, sea_sa = compute_north_atlantic_pca(z_raw)


# In[28]:


# 3. Predictors preprocessing
## 3.1. Space: select lat-lon boxes & take spatial averages 
(t2m_sa, z_sa, sm_all_levels_sa, sst_nwmed_sa, sst_cnaa_sa, rain_sa) = box_selection(t2m_raw, z_raw, sm_raw, sst_raw, rain_raw)


# In[29]:


# ---------------------------------------------------------------------------------------------------------------------------------------#    
## 3.2. Level: vertical level selection & combination
sm_sa = select_vert_level(sm_all_levels_sa)  


# In[30]:


# ---------------------------------------------------------------------------------------------------------------------------------------#
## 3.3. Dataset: homogenize variables in time and construct dataset
dset_sa = construct_dataset(t2m_sa, z_sa, sm_sa, sea_sa, sst_nwmed_sa, sst_cnaa_sa, rain_sa)


# In[35]:


# ---------------------------------------------------------------------------------------------------------------------------------------#
## 3.4. Detrend the data
dset_sa_det = detrend(dset_sa, False)


# In[36]:


# ---------------------------------------------------------------------------------------------------------------------------------------#
## 3.5. Standarized climatology
dset_clim_std_sa = compute_std_climatology(dset_sa_det)


# In[37]:


# ---------------------------------------------------------------------------------------------------------------------------------------#
## 3.6. Daily anomalies w.r.t. climatology
dset_anom_sa = compute_anomalies_wrt_climatology(dset_sa_det)


# In[39]:


# ---------------------------------------------------------------------------------------------------------------------------------------#        
## 3.7. Smooth out the data
dset_anom_sm_sa = smooth(dset_anom_sa)


# In[40]:


# ---------------------------------------------------------------------------------------------------------------------------------------#
## 3.8. Standarize
dset_anom_std_sa = standarize_data(dset_anom_sa)
dset_anom_sm_std_sa = standarize_data(dset_anom_sm_sa)


# **---------------------------------------------------------------------------------------------------------------------------------------**

# **---------------------------------------------------------------------------------------------------------------------------------------**

# In[44]:


# 4. Target definition
## Make 1SD index
hw_bin_1SD = compute_hw_bin_sd(dset_anom_sm_std_sa.t2m, 1)
## Make 1.5SD index
hw_bin_15SD = compute_hw_bin_sd(dset_anom_sm_std_sa.t2m, 1.5)


# **---------------------------------------------------------------------------------------------------------------------------------------**

# **---------------------------------------------------------------------------------------------------------------------------------------**

# In[45]:


# 5. Save predictors and targets
save_pred_and_targ(dset_anom_sm_std_sa, dset_clim_std_sa, dset_anom_sm_sa.t2m, hw_bin_1SD, hw_bin_15SD)


# In[ ]:




