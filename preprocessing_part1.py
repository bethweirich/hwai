#!/usr/bin/env python
# coding: utf-8

# # Preprocessing part 1

# In[3]:


# Import packages
import xarray as xr
from eofs.xarray import Eof
import numpy as np
import pandas as pd
import time
from scipy.stats import norm
from scipy import signal


# In[4]:


# Import own functions
from ipynb.fs.defs.utils import save_to_file, compute_ones_percentage, flip_longitude_360_2_180, month_to_season
from ipynb.fs.defs.plotting import show_multiple_time_series, show_individual_time_series, show_snao_sea_patterns, show_first_x_eofs


# In[5]:


# Import constants
from ipynb.fs.full.const import dictionary


# In[6]:


def import_data():
    
    """
    inputs
    ------
    None

    outputs
    -------
    z                       xr.DataArray : lat-lon grid x time series of Geopotential at 500hPa  
    sm                      xr.Dataset   : 4 vertical levels x lat-lon grid x time series of Soil Water Volume Level
    sst                     xr.DataArray : lat-lon grid x time series of Sea Surface Temperature
    rain                    xr.DataArray : lat-lon grid x time series of total precipitation
    t2m                     xr.DataArray : lat-lon grid x time series of 2m air temperature
    "rain_eraint             xr.DataArray : lat-lon grid x time series of total precipitation from ERAInterim"
    "t2m_eraint             xr.DataArray : lat-lon grid x time series of 2m air temperature from ERAInterim in K"
    
    """

    if dictionary['verbosity'] > 1: print('********************************************* Importing data ***********************************************')
     
    start = time.time()

    # Access file for each variable   
    ## ERA5
    (access_z_era5_file, access_sm_era5_file, access_sst_era5_file, access_rain_era5_file, access_t2m_era5_file) = (
                                dictionary['path_input_orig_pred'] + 'z_ERA5_1950-2021_M01-12_1d.nc',
                                dictionary['path_input_orig_pred'] + 'sm_ERA5_1950-2020_M01-12_1d.nc',
                                dictionary['path_input_orig_pred'] + 'sst_ERA5_1950-2020_M01-12_1d.nc',
                                dictionary['path_input_orig_pred'] + 'rain_ERA5_1950-2020_M01-12_1d.nc',
                                dictionary['path_input_orig_pred'] + 't2m_ERA5_1950-2021_M01-12_1d.nc'
                                                                                                                       )
    ## ERA20C
    (access_z_era20c_file, access_sm_era20c_file, access_sst_era20c_file, access_rain_era20c_file, access_t2m_era20c_file) = (
                                dictionary['path_input_orig_pred'] + 'z_ERA20C_1900-2009_M01-12_1d.nc',
                                dictionary['path_input_orig_pred'] + 'sm_ERA20C_1900-2009_M01-12_1d.nc',
                                dictionary['path_input_orig_pred'] + 'sst_ERA20C_1900-2009_M01-12_1d.nc',
                                dictionary['path_input_orig_pred'] + 'rain_ERA20C_1900-2009_M01-12_1d.nc',
                                dictionary['path_input_orig_pred'] + 't2m_ERA20C_1900-2009_M01-12_1d.nc'
                                                                                                                               )
    ## Combi: E-OBS+ERAInt+ERA5-Land+HadISST
    (access_z_eraint_file, access_sm_era5land_file, access_sst_hadisst_file, access_rain_eobs_file, access_rain_eraint_file, access_t2m_eobs_file, access_t2m_eraint_file) = (
                dictionary['path_eraint'] + 'phi/daymean/phi-eraint-37plevels-daymean-*',
                dictionary['path_era5_land'] + 'sm/VSW_l1-4_1981-2019-daily.nc',
                dictionary['path_hadisst'] + 'HadISST_monmean_1870-2019.nc',
                dictionary['path_eobs'] + 'precip/daymean/rr_ens_mean_0.25deg_reg_v22.0e.nc',
                dictionary['path_eraint'] + 'precip/daymean/precip-eraint-daymean-*',
                dictionary['path_eobs'] + 't2m_mean/daymean/tg_ens_mean_0.25deg_reg_v22.0e.nc',
                dictionary['path_eraint'] + 't2m/daymean/t2m-eraint-daymean-*'
                                                                                                                                            )

        
        
    # Load data
    ## Target
    t2m = xr.open_dataset(access_t2m_eobs_file).tg.load()
    
    ## Predictors
    ### ERA5
    if dictionary['dataset'] == 'ERA5':
        print('Using ERA5 data')
        (z, sm, sst, rain, t2m_x) = (
                                    xr.open_dataset(access_z_era5_file).z.load().sel(level = dictionary['geopotential_level']),
                                    xr.open_dataset(access_sm_era5_file).load(),
                                    xr.open_dataset(access_sst_era5_file).sst.load(),
                                    xr.open_dataset(access_rain_era5_file).tp.load(),
                                    xr.open_dataset(access_t2m_era5_file).t2m.load()
                                                             )
        # Rename lat and lon to latitude and longitude for all variables but SST
        (z, sm, rain, t2m_x) = (
                                    z.rename({'lat': 'latitude', 'lon': 'longitude'}),
                                    sm.rename({'lat': 'latitude', 'lon': 'longitude'}),
                                    rain.rename({'lat': 'latitude', 'lon': 'longitude'}),
                                    t2m_x.rename({'lat': 'latitude', 'lon': 'longitude'})
                                                            )
    ### ERA20C
    if dictionary['dataset'] == 'ERA20C':        
        print('Using ERA20C data')
        (z, sm, sst, rain, t2m_x) = (
                                    xr.open_dataset(access_z_era20c_file).z.load(),
                                    xr.open_dataset(access_sm_era20c_file).load(),
                                    xr.open_dataset(access_sst_era20c_file).sst.load(),
                                    xr.open_dataset(access_rain_era20c_file).tp.load(),
                                    xr.open_dataset(access_t2m_era20c_file).t2m.load()
                                                                         )        
    ### Combi: E-OBS+ERAInt+ERA5-Land+HadISST   
    elif dictionary['dataset'] == 'combi': 
        print('Using Combi: E-OBS+ERAInt+ERA5-Land+HadISST data')
        (z, sm, sst, rain, rain_eraint, t2m_x, t2m_x_eraint) =  (
                    xr.open_mfdataset(access_z_eraint_file,combine = 'by_coords', parallel = True).phi.load().sel(level = dictionary['geopotential_level']),
                    xr.open_dataset(access_sm_era5land_file).load(),
                    xr.open_dataset(access_sst_hadisst_file).sst.load(),
                    xr.open_dataset(access_rain_eobs_file).rr.load(),
                    xr.open_mfdataset(access_rain_eraint_file, combine = 'by_coords', parallel = True).tp.load(),
                    xr.open_dataset(access_t2m_eobs_file).tg.load(),
                    xr.open_mfdataset(access_t2m_eraint_file, combine = 'by_coords', parallel = True).t2m.load()
                                                             )


    end = time.time()
    if dictionary['verbosity'] > 2: print('Time needed: ', (end - start)/60, ' minutes.')
    
    if dictionary['dataset'] == 'ERA5' or dictionary['dataset'] == 'ERA20C': 
        return t2m, z, sm, sst, rain, t2m_x, None, None
    elif dictionary['dataset'] == 'combi': 
        return t2m, z, sm, sst, rain, t2m_x, rain_eraint, t2m_x_eraint 


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
    ## Plot patter of first 10 EOFs
    if dictionary['verbosity'] > 4: show_first_x_eofs(eofs, num_eofs)
    ## Attribute EOFs to Atlantic Patterns: 1st (mode = 0) to SNAO, 2nd (mode = 1) to SEA
    snao_eof = eofs.sel(mode = 0)
    if dictionary['dataset'] != 'ERA20C': sea_eof = eofs.sel(mode = 1)
    else: sea_eof = eofs.sel(mode = 3)

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


# In[7]:


def box_selection(t2m, z, sm, sst,  rain, t2m_x, rain_eraint, t2m_x_eraint):
    
    """
    inputs
    ------
    t2m                      xr.DataArray : lat-lon grid x time series of 2m air temperature (target)
    z                        xr.DataArray : lat-lon grid x time series of Geopotential at 500hPa (predictor)
    sm                       xr.Dataset   : 4 vertical levels x lat-lon grid x time series of Soil Water Volume Level (predictor)
    sst                      xr.DataArray : lat-lon grid x time series of Sea Surface Temperature (predictor)
    rain                     xr.DataArray : lat-lon grid x time series of total precipitation (predictor)
    t2m_x                    xr.DataArray : lat-lon grid x time series of 2m air temperature (predictor)
    "rain_eraint             xr.DataArray : lat-lon grid x time series of total precipitation from ERAInt" (predictor)
    "t2m_x_eraint            xr.DataArray : lat-lon grid x time series of 2m air temperature from ERAInt" (predictor)

    outputs
    -------
    t2m_sa_                  xr.DataArray : time series of 2m air temperature in CE (target)   
    z_sa_                    xr.DataArray : time series of Geopotential in CE at 500hPa (predictor)
    sm_all_levels_sa_        xr.Dataset   : 4 vertical levels x time series of Soil Water Volume Level in CE (predictor)
    sst_med_sa_              xr.DataArray : time series of Sea Surface Temperature in MED (predictor)
    sst_nwmed_sa_            xr.DataArray : time series of Sea Surface Temperature in NWMED (predictor)
    sst_semed_sa_            xr.DataArray : time series of Sea Surface Temperature in SEMED (predictor)
    sst_baltic_sa_           xr.DataArray : time series of Sea Surface Temperature in Baltic (predictor)
    sst_cbna_sa_             xr.DataArray : time series of Sea Surface Temperature in CBNA (predictor)
    sst_nwna_sa_             xr.DataArray : time series of Sea Surface Temperature in NWNA (predictor)
    sst_sena_sa_             xr.DataArray : time series of Sea Surface Temperature in SENA (predictor)
    sst_enso_sa_             xr.DataArray : time series of Sea Surface Temperature in ENSO (predictor)
    rain_sa_                 xr.DataArray : time series of total precipitation in CE (predictor)
    rain_pac_sa_             xr.DataArray : time series of total precipitation in the Pacific (predictor)
    rain_carib_sa_           xr.DataArray : time series of total precipitation in the Caribbean (predictor)    
    t2m_x_sa_                xr.DataArray : time series of 2m air temperature in CE (predictor)  
    t2m_x_nsa_               xr.DataArray : CE lat-lon grid x time series of 2m air temperature (predictor)
    z_nsa_                   xr.DataArray : CE lat-lon grid x time series of Geopotential at 500hPa (predictor)
    
    sa = spatially averaged
    nsa = not spatially averaged
    
    """ 
    
    if dictionary['verbosity'] > 1: print('******************************* Selecting boxes and spatially averaging data *******************************')
    
    if dictionary['dataset'] == 'ERA5' or dictionary['dataset'] == 'ERA20C':
        # Flip latitude
        (z, sm, rain, t2m_x) = (
                z.sortby('latitude', ascending = False),
                sm.sortby('latitude', ascending = False),
                rain.sortby('latitude', ascending = False),
                t2m_x.sortby('latitude', ascending = False)
                                )
        # Flip longitude
        sst =  flip_longitude_360_2_180(sst, sst.longitude)

    # Select data in grid point boxes & average over each box
    ## CE for 2m Temperature, Geopotential, E-Obs Precipitation and Soil Moisture 
    ## MED (NWMED and SEMED), NA (NWNA, SENA, BALTIC, CBNA) and Pacific (ENSO) for Sea Surface Temperature
    ## Pacific and Caribbean Boxes for ERA-Int Precipitation
    
    ## Target
    t2m_sa_ = t2m.sel(latitude = slice(dictionary['boxes'].sel(box='ce_obs',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce_obs',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce_obs',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce_obs',axis='longitude',edge='end'))).mean('latitude').mean('longitude')                   
    
    ## Predictors
    (z_sa_, sm_all_levels_sa_, sst_med_sa_, sst_nwmed_sa_, sst_semed_sa_, sst_baltic_sa_, sst_cbna_sa_, sst_nwna_sa_, sst_sena_sa_, sst_enso_sa_, z_nsa_) =  (
                z.sel(latitude = slice(dictionary['boxes'].sel(box='ce',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
                sm.sel(latitude = slice(dictionary['boxes'].sel(box='ce',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
                sst.sel(latitude = slice(dictionary['boxes'].sel(box='med',axis='latitude',edge='start'),dictionary['boxes'].sel(box='med',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='med',axis='longitude',edge='start'),dictionary['boxes'].sel(box='med',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),       
                sst.sel(latitude = slice(dictionary['boxes'].sel(box='nwmed',axis='latitude',edge='start'),dictionary['boxes'].sel(box='nwmed',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='nwmed',axis='longitude',edge='start'),dictionary['boxes'].sel(box='nwmed',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),       
                sst.sel(latitude = slice(dictionary['boxes'].sel(box='semed',axis='latitude',edge='start'),dictionary['boxes'].sel(box='semed',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='semed',axis='longitude',edge='start'),dictionary['boxes'].sel(box='semed',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
                sst.sel(latitude = slice(dictionary['boxes'].sel(box='baltic',axis='latitude',edge='start'),dictionary['boxes'].sel(box='baltic',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='baltic',axis='longitude',edge='start'),dictionary['boxes'].sel(box='baltic',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
                sst.sel(latitude = slice(dictionary['boxes'].sel(box='cbna',axis='latitude',edge='start'),dictionary['boxes'].sel(box='cbna',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='cbna',axis='longitude',edge='start'),dictionary['boxes'].sel(box='cbna',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
                sst.sel(latitude = slice(dictionary['boxes'].sel(box='nwna',axis='latitude',edge='start'),dictionary['boxes'].sel(box='nwna',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='nwna',axis='longitude',edge='start'),dictionary['boxes'].sel(box='nwna',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
                sst.sel(latitude = slice(dictionary['boxes'].sel(box='sena',axis='latitude',edge='start'),dictionary['boxes'].sel(box='sena',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='sena',axis='longitude',edge='start'),dictionary['boxes'].sel(box='sena',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
                sst.sel(latitude = slice(dictionary['boxes'].sel(box='enso',axis='latitude',edge='start'),dictionary['boxes'].sel(box='enso',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='enso',axis='longitude',edge='start'),dictionary['boxes'].sel(box='enso',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),          
                xr.concat([z.sel(latitude = slice(dictionary['boxes'].sel(box='ce_large_left',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce_large_left',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce_large_left',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce_large_left',axis='longitude',edge='end'))),
                z.sel(latitude = slice(dictionary['boxes'].sel(box='ce_large_right',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce_large_right',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce_large_right',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce_large_right',axis='longitude',edge='end')))], dim = 'longitude')
                                                                                                                                                                )
    if dictionary['dataset'] == 'ERA5' or dictionary['dataset'] == 'ERA20C':
        (rain_sa_, rain_pac_sa_, rain_carib_sa_, t2m_x_sa_, t2m_x_nsa_) = (
                    rain.sel(latitude = slice(dictionary['boxes'].sel(box='ce',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
                    rain.sel(latitude = slice(dictionary['boxes'].sel(box='pac',axis='latitude',edge='start'),dictionary['boxes'].sel(box='pac',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='pac',axis='longitude',edge='start'),dictionary['boxes'].sel(box='pac',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
                    rain.sel(latitude = slice(dictionary['boxes'].sel(box='carib',axis='latitude',edge='start'),dictionary['boxes'].sel(box='carib',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='carib',axis='longitude',edge='start'),dictionary['boxes'].sel(box='carib',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
                    t2m_x.sel(latitude = slice(dictionary['boxes'].sel(box='ce',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce',axis='longitude',edge='end'))).mean('latitude').mean('longitude'), 
                    xr.concat([t2m_x.sel(latitude = slice(dictionary['boxes'].sel(box='ce_large_left',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce_large_left',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce_large_left',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce_large_left',axis='longitude',edge='end'))),
                    t2m_x.sel(latitude = slice(dictionary['boxes'].sel(box='ce_large_right',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce_large_right',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce_large_right',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce_large_right',axis='longitude',edge='end')))], dim = 'longitude')
                                                                        )

    elif dictionary['dataset'] == 'combi': 
        (rain_sa_, rain_pac_sa_, rain_carib_sa_, t2m_x_sa_, t2m_x_nsa_) =  (
                    rain.sel(latitude = slice(dictionary['boxes'].sel(box='ce_obs',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce_obs',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce_obs',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce_obs',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
                    rain_eraint.sel(latitude = slice(dictionary['boxes'].sel(box='pac',axis='latitude',edge='start'),dictionary['boxes'].sel(box='pac',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='pac',axis='longitude',edge='start'),dictionary['boxes'].sel(box='pac',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
                    rain_eraint.sel(latitude = slice(dictionary['boxes'].sel(box='carib',axis='latitude',edge='start'),dictionary['boxes'].sel(box='carib',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='carib',axis='longitude',edge='start'),dictionary['boxes'].sel(box='carib',axis='longitude',edge='end'))).mean('latitude').mean('longitude'),
                    t2m_x.sel(latitude = slice(dictionary['boxes'].sel(box='ce_obs',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce_obs',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce_obs',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce_obs',axis='longitude',edge='end'))).mean('latitude').mean('longitude'), 
                    xr.concat([t2m_x_eraint.sel(latitude = slice(dictionary['boxes'].sel(box='ce_large_left',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce_large_left',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce_large_left',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce_large_left',axis='longitude',edge='end'))),
                    t2m_x_eraint.sel(latitude = slice(dictionary['boxes'].sel(box='ce_large_right',axis='latitude',edge='start'),dictionary['boxes'].sel(box='ce_large_right',axis='latitude',edge='end')), longitude = slice(dictionary['boxes'].sel(box='ce_large_right',axis='longitude',edge='start'),dictionary['boxes'].sel(box='ce_large_right',axis='longitude',edge='end')))], dim = 'longitude')
                                                                        )
    
    return t2m_sa_, z_sa_, sm_all_levels_sa_, sst_med_sa_, sst_nwmed_sa_, sst_semed_sa_, sst_baltic_sa_, sst_cbna_sa_, sst_nwna_sa_, sst_sena_sa_, sst_enso_sa_, rain_sa_, rain_pac_sa_, rain_carib_sa_, t2m_x_sa_, t2m_x_nsa_, z_nsa_


# In[8]:


def select_vert_level(sm_all_levels_sa_):
    
    """
    inputs
    ------ 
    sm_all_levels_sa_                    xr.Dataset   : 4 vertical levels x time series of Soil Water Volume Level in CE


    outputs
    -------
    sm_sa_                               xr.DataArray : time series of Soil Water Volume Level in CE at 0 - 28 cm depth
    sm_deeper_sa_                        xr.DataArray : time series of Soil Water Volume Level in CE at 0 - 100 cm depth
    sm_deepest_sa_                       xr.DataArray : time series of Soil Water Volume Level in CE at 100 - 289 cm depth
    
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
    ### Combine the first three levels into one (sm_deeper)
    sm_deeper_ds = xr.Dataset({'sm1': sm_all_levels_sa_.swvl1, 'sm2': sm_all_levels_sa_.swvl2, 'sm3': sm_all_levels_sa_.swvl3})
    sm_deeper_sa_ = sm_deeper_ds.to_array(dim='new').mean('new')
    ### Extract the 4th level (sm_deepest)
    sm_deepest_sa_ = sm_all_levels_sa_.swvl4
    
    return sm_sa_, sm_deeper_sa_, sm_deepest_sa_


# In[9]:


def construct_dataset(t2m_sa_, z_sa_, sm_sa_, sm_deeper_sa_, sm_deepest_sa_, snao_sa_, sea_sa_, sst_med_sa_, sst_nwmed_sa_, sst_semed_sa_, sst_baltic_sa_, sst_cbna_sa_,
                sst_nwna_sa_, sst_sena_sa_, sst_enso_sa_, rain_sa_, rain_pac_sa_, rain_carib_sa_, t2m_x_sa_, t2m_x_nsa_, z_nsa_):
    
    """
    inputs
    ------
    t2m_sa_                  xr.DataArray : time series of 2m air temperature in CE (target)
    z_sa_                    xr.DataArray : time series of Geopotential in CE at selected geopotential level (predictor)
    sm_sa_                   xr.DataArray : time series of Soil Water Volume Level in CE at 0 - 28 cm depth (predictor)
    sm_deeper_sa_            xr.DataArray : time series of Soil Water Volume Level in CE at 0 - 100 cm depth (predictor)
    sm_deepest_sa_           xr.DataArray : time series of Soil Water Volume Level in CE at 100 - 289 cm depth (predictor)
    snao_sa_                 xr.DataArray : time series of Summer North Atlantic Oscillation (SNAO) (predictor)
    sea_sa_                  xr.DataArray : time series of Summer East Atlantic (SEA) pattern (predictor)
    sst_med_sa_              xr.DataArray : time series of Sea Surface Temperature in MED (predictor)
    sst_nwmed_sa_            xr.DataArray : time series of Sea Surface Temperature in NWMED (predictor)
    sst_semed_sa_            xr.DataArray : time series of Sea Surface Temperature in SEMED (predictor)
    sst_baltic_sa_           xr.DataArray : time series of Sea Surface Temperature in Baltic (predictor)
    sst_cbna_sa_             xr.DataArray : time series of Sea Surface Temperature in CBNA (predictor)
    sst_nwna_sa_             xr.DataArray : time series of Sea Surface Temperature in NWNA (predictor)
    sst_sena_sa_             xr.DataArray : time series of Sea Surface Temperature in SENA (predictor)
    sst_enso_sa_             xr.DataArray : time series of Sea Surface Temperature in ENSO (predictor)
    rain_sa_                 xr.DataArray : time series of total precipitation in CE (predictor)
    rain_pac_sa_             xr.DataArray : time series of total precipitation in the Pacific (predictor)
    rain_carib_sa_           xr.DataArray : time series of total precipitation in the Caribbean (predictor)
    t2m_x_sa_                xr.DataArray : time series of 2m air temperature in CE (predictor)   
    t2m_x_nsa_               xr.DataArray : CE lat-lon grid x time series of 2m air temperature (predictor)
    z_nsa_                   xr.DataArray : CE lat-lon grid x time series of Geopotential at selected geopotential level (predictor)  
    

    outputs
    -------   
    dset_sa_                 xr.Dataset : dataset of spatially averaged variables 
    dset_nsa_                xr.Dataset : dataset of not spatially averaged variables 

    
    sa = spatially averaged
    nsa = not spatially averaged
    
    """ 
    
    if dictionary['verbosity'] > 1: print('****************************************** Constructing dataset ********************************************')
    
    # Create dataset (matrix of xarrays) containing all spatially averaged monthly variables
    dset_sa_1 = xr.Dataset({
        'sst_med': sst_med_sa_,
        'sst_nwmed': sst_nwmed_sa_, 
        'sst_semed': sst_semed_sa_, 
        'sst_baltic': sst_baltic_sa_, 
        'sst_cbna': sst_cbna_sa_, 
        'sst_nwna': sst_nwna_sa_, 
        'sst_sena': sst_sena_sa_,
        'sst_enso': sst_enso_sa_
                        }).resample(time = '1D')

    # Create dataset (matrix of xarrays) containing all spatially averaged daily variables & resample
    dset_sa_2 = xr.Dataset({       
        't2m': t2m_sa_,
        't2m_x': t2m_x_sa_,
        'z': z_sa_,
        'rain': rain_sa_,
        'rain_pac': rain_pac_sa_,
        'rain_carib': rain_carib_sa_,
        'sm': sm_sa_, 
        'sm_deeper': sm_deeper_sa_, 
        'sm_deepest': sm_deepest_sa_, 
        'snao': snao_sa_,
        'sea': sea_sa_
                        }).resample(time = '1D').first()

    # Create dataset with not spatially averaged variables
    t2m_x_nsa_ = t2m_x_nsa_.resample(time = '1D').first()
    dset_nsa_ = xr.Dataset({
        't2m_x_nsa': t2m_x_nsa_,
        'z_nsa': z_nsa_
                        }).resample(time = '1D').first()

    # Interpolate monthly values to daily ones for SST
    ## Interpolate to time 00:00:00 to be consistent (resample) 
    if dictionary['dataset'] == 'ERA5' or dictionary['dataset'] == 'ERA20C':
        dset_sa_1 = dset_sa_1.first()
    elif dictionary['dataset'] == 'combi': 
        dset_sa_1 = dset_sa_1.interpolate('linear')

    # Merge two datasets
    dset_sa_ = xr.merge([dset_sa_2, dset_sa_1])
    ## Drop times with Nan's
    dset_nsa_ = dset_nsa_.dropna(dim = 'time', how = 'any', thresh = None)
    ## Drop extra coordinates    
    dset_sa_ = dset_sa_.drop(['level'], errors = 'ignore')
    dset_nsa_ = dset_nsa_.drop(['level'], errors = 'ignore')
    
    return dset_sa_, dset_nsa_


# In[10]:


def compute_pcd_index(dset):
    
    
    """
    inputs
    ------
    dset                 xr.Dataset : dataset containing rain_pac and rain_carib
    

    outputs
    -------   
    dset                 xr.Dataset : dataset containing pcd_index
    
    """ 

    if dictionary['verbosity'] > 1: print('****************************************** Computing PCD index *********************************************')

    #### Compute PCD Index (as defined in Wulff et al. "Tropical Forcing of the Summer East Atlantic Pattern"):
    #### Precipitation in Pacific box minus precipitation in Caribbean box -> afterwards normalized in step 2.5.
    pcd_index_data = dset.rain_pac - dset.rain_carib
    ## Add data to dset
    dset = dset.assign(pcd_index = pcd_index_data)
    # Drop the Precipitation in the Pacific and the Caribbean
    dset = dset.drop('rain_pac').drop('rain_carib')
    
    return dset


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
    dset_clim_std        xr.Dataset : daily climatology (1 value for each day of the year x # vars)
    
    """ 

    if dictionary['verbosity'] > 1: print('*************************** Computing standarized climatology *****************************')
    
    # Daily climatology
    dset_clim = dset.groupby('time.dayofyear').mean('time')
    
    ## Smooth monthly
    dset_clim_sm = dset_clim.pad(dayofyear = 15, mode = 'wrap').rolling(dayofyear = 31, center=True).mean().dropna(dim = 'dayofyear')
    
    ## Drop extra coordinates
    dset_clim_sm = dset_clim_sm.drop(['dayofyear'])    
        
    return dset_clim_sm


# In[15]:


def correct_sm_data(dset, method):

    """
    inputs
    ------ 
    dset                   xr.Dataset : dataset containing jump in soil moisture
    method                        str : what to do to correct weird jumps in data? 'interpolate' or 'exclude'

    outputs
    -------
    dset_c                 xr.Dataset : corrected dataset 
    
    """

    if dictionary['verbosity'] > 1: print('**************************************** Correcting soil moisture data ******************************************')

    # Min and max dates for which the data is available
    min_date = pd.DatetimeIndex(dset.to_dataframe().index)[0]
    max_date = pd.DatetimeIndex(dset.to_dataframe().index)[-1]
    
    if dictionary['dataset'] == 'ERA5' and pd.DatetimeIndex(dset.to_dataframe().index).year[0] <= 1953 and pd.DatetimeIndex(dset.to_dataframe().index).year[-1] >= 1953:     
        if method == 'interpolate':
            # Correct jump in soil moisture in year 1953 for ERA5
            ## Extract sm only
            sm = dset.drop('t2m').drop('t2m_x').drop('z').drop('snao').drop('sea').drop('rain').drop('pcd_index').drop('sst_med').drop('sst_nwmed').drop('sst_semed').drop('sst_baltic').drop('sst_cbna').drop('sst_nwna').drop('sst_sena').drop('sst_enso')
            ## Replace error period with Nan
            sm.sm.loc['1953'] = np.nan
            sm.sm_deeper.loc['1953'] = np.nan
            sm.sm_deepest.loc['1953'] = np.nan
            ## Interpolate linearly
            #print('Length before interpolation: ', len(sm_cut.time))
            sm_interp = sm.interpolate_na(dim = 'time', method = 'linear')
            ## Merge with remaining variables again
            dset_c = xr.merge([
                dset.drop('sm').drop('sm_deeper').drop('sm_deepest'),
                sm_interp])
            
        elif method == 'exclude':
            # Exclude data from 1953 completely for ERA5
            #dset_c = xr.concat([dset.sel(time = slice(str(dictionary['start_year']), '1952-12-31')), dset.sel(time = slice('1954-01-01', str(dictionary['end_year'])))], dim = 'time')
            dset_c = xr.concat([dset.sel(time = slice(min_date, '1952-12-31')), dset.sel(time = slice('1954-01-01', max_date))], dim = 'time')
    
        return dset_c
    
    else:
        return dset


# In[16]:


def correct_sst_data(dset_anom, method):

    """
    inputs
    ------ 
    dset_anom                   xr.Dataset : dataset of anomalies containing jump in SSTs
    method                             str : what to do to correct weird jumps in data? 'interpolate' or 'exclude'
    
    outputs
    -------
    dset_anom_c                 xr.Dataset : corrected dataset of anomalies 
    
    """

    if dictionary['verbosity'] > 1: print('**************************************** Correcting SST data ******************************************')

    # Min and max dates for which the data is available
    min_date = pd.DatetimeIndex(dset_anom.to_dataframe().index)[0]
    max_date = pd.DatetimeIndex(dset_anom.to_dataframe().index)[-1]
    
    if dictionary['dataset'] == 'ERA5': 
        if method == 'interpolate':
            # Correct jump in SSTs in year 1956 for ERA5
            ## Extract SSTs only
            sst_anom = dset_anom.drop('t2m').drop('t2m_x').drop('z').drop('sm').drop('sm_deeper').drop('sm_deepest').drop('snao').drop('sea').drop('rain').drop('pcd_index')
            ## Replace error period with Nan
            sst_anom_c = sst_anom.where(abs(sst_anom.sst_enso)<6)
            ## Add correction for jumps arising after standarization with simple mean
            sst_anom_cc = sst_anom.where(abs(sst_anom_c.sst_nwmed)<5)
            ## Interpolate linearly
            sst_anom_interp = sst_anom_cc.interpolate_na(dim = 'time', method = 'linear')
            ## Merge with remaining variables again
            dset_anom_c = xr.merge([
                dset_anom.drop('sst_med').drop('sst_nwmed').drop('sst_semed').drop('sst_baltic').drop('sst_cbna').drop('sst_nwna').drop('sst_sena').drop('sst_enso'),
                sst_anom_interp])
        if method == 'exclude' and min_date.year <= 1956 and max_date.year >= 1956:
            # Exclude data from 1956 completely for ERA5
            #dset_anom_c = xr.concat([dset_anom.sel(time = slice(str(dictionary['start_year']), '1955-12-31')), dset_anom.sel(time = slice('1957-01-01', str(dictionary['end_year'])))], dim = 'time')
            dset_anom_c = xr.concat([dset_anom.sel(time = slice(min_date, '1955-12-31')), dset_anom.sel(time = slice('1957-01-01', max_date))], dim = 'time')
    

            
        return dset_anom_c
    
    else:
        return dset_anom


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
    if dictionary['mean_type'] == 'simple':
        # OLD APPROACH: SIMPLE MEAN (timestep resolution)
        dset_sm = dset.resample(time = dictionary['timestep']).mean() 

    if dictionary['mean_type'] == 'rolling':
        # NEW APPROACH: ROLLING MEAN (daily resolution) -> mean value always centered at the middle of the timestep 
        # e.g. at position 4 for a week numbered as 1 2 3 4 5 6 7 and timestep = '7D'
        dset_sm = dset.rolling(time = dictionary['timestep_num'], center = True).mean().dropna(dim = 'time')
        
    return dset_sm 


# In[19]:


def compute_na_index(dset_anom_):
    
    """
    inputs
    ------
    dset_anom_                 xr.Dataset : dataset containing sst_nwna and sst_sena anomalies
    

    outputs
    -------   
    dset_anom_                 xr.Dataset : dataset containing na_index, sst_nwna and sst_sena anomalies
    
    """     
    
    if dictionary['verbosity'] > 1: print('****************************************** Computing NA index **********************************************')
    
    # NA Index 
    ## (as defined in Ossó et al. "Develpment, Amplification, and Decay of
    ## Atlantic/European Summer Weather Patterns linked to Springh North Atlantic SST"):
    ## Anomalies in NW box minus anomalies in SE box 
    na_index_data = dset_anom_.sst_nwna - dset_anom_.sst_sena
    ## Add data to dset_anom
    dset_anom_ = dset_anom_.assign(na_index = na_index_data)
    
    return dset_anom_


# In[20]:


def find_extreme_days(t):
    
    """
    inputs
    ------
    t                       xr.DataArray : data array containing the detrended standarized 2m air temperature anomalies w.r.t. climatology


    outputs
    -------   
    t_1d_above_             xr.DataArray : data array with binary index (1: extremely hot day, 0: not extremely hot day). An extremely hot day is defined
                                           as being above a certain percentile, which defined by the user.
    
    """  

    if dictionary['verbosity'] > 1: print('**************************************** Finding extremely hot days *******************************************')

    # Percentile treshold
    ## Percentile threshold := matrix with one value per day of year (time = 366) per grid point
    t_per = t.groupby('time.dayofyear').quantile(dictionary['percentile'], dim = 'time').load() 
    ## Smooth out percentile threshold by averaging over 1 month via a cyclic running mean
    t_per_sm = t_per.pad(dayofyear = 15, mode = 'wrap').rolling(dayofyear = 31, center = True).mean().dropna(dim = 'dayofyear')

    # Identification of extreme days (above percentile) in full time period 
    ## Output: binary array (1/0)
    t_1d_above_ = xr.where(t.groupby('time.dayofyear') > t_per_sm, 1, 0).load() 
    ## Drop extra coordinates
    t_1d_above_ = t_1d_above_.drop(['quantile'])  
    
    if dictionary['verbosity'] > 2: compute_ones_percentage(t_1d_above_, 't_1d_above_', False)
    
    return t_1d_above_   


# In[21]:


def cal_onset_date(x):
    
    
    """
    inputs
    ------
    x                       xr.DataArray : binary index (1: extremely hot day, 0: not extremely hot day). An extremely hot day is defined
                                           as being above a certain percentile, which defined by the user.


    outputs
    -------   
    onset                   xr.DataArray : binary index (1: extremely hot days sequence starts on that day, 0: no extremely hot days 
                                           sequence starts on that day)      
    end                     xr.DataArray : binary index (1: extremely hot days sequence ends on that day, 0: no extremely hot days 
                                           sequence ends on that day) 
    duration                xr.DataArray : duration (in days) of the extremely hot days sequence that starts on that day
    
    """  

    if dictionary['verbosity'] > 1: print('*************************************** Computing heat wave duration ******************************************')


    # Compute difference of shifted arrays
    ## Shift array by 1 position and substract the orinigal one from it (1: enter extreme weather period,
    ## -1: exit extreme weather period, 0: inside of a period -> can be both extreme or not)
    shift = np.array(x[1:]) - np.array(x[0:len(x) - 1]).astype(int)
    # Detect transition form 0 to 1 
    onset = np.array(np.where(shift == 1)).astype(int).flatten() 
    # Detect transition form 1 to 0 
    end = np.array(np.where(shift == -1)).astype(int).flatten()
    
    # Compute the duration of each event
    ## Check the first and last event
    ### Make sure that there are changes and not all > 95th percentile nor all < 95th percentile, if not: trivial duration 
    if (len(end) == 0 or len(onset) == 0):
        duration = end
    else:  
        #### Make sure that the first event does not end before it starts, otherwise cut off first end
        if (end[0] <= onset[0]):
            end = end[1:len(onset) + 1]
        #### Make sure that the last event that starts also ends, otherwise cut off last start
        if (onset[len(onset) - 1] >= end[len(end) - 1]):
            onset = onset[0:len(onset) - 1]    
        duration = np.array(end - onset)
        
    return onset, end, duration


# In[22]:


def filter_min_duration(x):

    """
    inputs
    ------
    x                       xr.DataArray : binary index (1: extremely hot day, 0: not extremely hot day). An extremely hot day is defined
                                           as being above a certain percentile, which defined by the user.


    outputs
    -------   
    x                       xr.DataArray : binary index (1: daily heat wave, 0: no daily heat wave). A daily heat wave is defined as an 
                                           extremely hot day belonigng to a sequnce of 3 or more extremely hot days
    
    """  

    on,en,dur = cal_onset_date(x)
    
    if dictionary['verbosity'] > 1: print('**************************************** Filtering heat wave events *******************************************')

    if(len(on) >= 1):
        on_d = on[np.logical_not(np.isnan(xr.where(dur >= dictionary['min_duration'],np.nan, on)))].astype(int)
        en_d = en[np.logical_not(np.isnan(xr.where(dur >= dictionary['min_duration'],np.nan, en)))].astype(int)
        for k, onset in enumerate(on_d):
            x[onset:en_d[k] + 1] = 0
            
    if dictionary['verbosity'] > 2: compute_ones_percentage(x, 'x', False)
    
    del(on,en,dur)
    return(x)   


# In[23]:


def compute_hw_bin_2in7(t2m_anom):
    
    """
    inputs
    ------
    t2m_anom                  xr.DataArray : time series of detrended 2m air temperature anomalies w.r.t climatology in CE


    outputs
    -------   
    hw_bin_2in7_              xr.DataArray : binary multi-measurement (extreme temperature + persistence) weekly heat wave index 
    
    """  

    if dictionary['verbosity'] > 1: print('************************************** Computing the hw_bin_2in7 index ****************************************')


    # Find extremely hot days
    t2m_1d_above = find_extreme_days(t2m_anom)

    # Apply persistence condition: remove hw's with a duration of less than min_duration days
    ## New array with the same dimensionality and coordinates
    t2m_above = t2m_1d_above.copy(data = np.zeros(t2m_1d_above.shape))
    ## Filter
    t2m_above[:] = filter_min_duration(t2m_1d_above[:])

    # Smooth out the data
    hw_con = smooth(t2m_above)
    
    # Make the heat wave index binary again (0: if none of the days has a HW, 1: if at least 2/7 of the days have a HW)
    hw_bin_2in7_ = (hw_con > 0.28).astype(int)
    
    if dictionary['verbosity'] > 2: compute_ones_percentage(hw_bin_2in7_, 'hw_bin_2in7_', False)
        
    return hw_bin_2in7_


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


def save_pred_and_targ(dset_anom_sa_, dset_anom_nsa_, dset_clim_std_, t2m_, hw_bin_2in7_, hw_bin_1SD_, hw_bin_15SD_):
    
    """
    inputs
    ------
    dset_anom_sa_              xr.Dataset : preprocessed spatially averaged predictor's anomalies (standarized and smoothed)
    dset_anom_nsa_             xr.Dataset : preprocessed not spatially averaged predictor's anomalies (standarized and smoothed)
    dset_clim_std_             xr.Dataset : climatology for the target variable (standarized and smoothed)
    t2m_                     xr.DataArray : 2m temperature spatially averaged over CE (non-standarized, but smoothed)
    hw_bin_2in7_             xr.DataArray : binary multi-measurement (extreme temperature + persistence) weekly heat wave index 
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
    save_name_sa = 'input_' + dictionary['timestep'] +  '_' + dictionary['mean_type'] + '_mean_sa.nc'
    save_name_nsa = 'input_' + dictionary['timestep'] +  '_' + dictionary['mean_type'] + '_mean_nsa.nc'
    save_name_clim_std = 'clim_' + dictionary['timestep'] + '_' + dictionary['mean_type'] + '_mean.nc'
    save_name_targets = 'targets_' + dictionary['timestep'] +  '_' + dictionary['mean_type'] + '_mean.nc'
    
    # Specify objects
    obj_sa = dset_anom_sa_[['t2m_x', 'z', 'rain', 'sm', 'sm_deeper', 'sm_deepest', 'snao', 'sea', 'sst_med', 'sst_nwmed', 'sst_semed',
                            'sst_baltic', 'sst_cbna', 'sst_nwna', 'sst_sena', 'na_index', 'sst_enso', 'pcd_index']]
    #.rename({'t2m': 't2m_x'})
    obj_nsa = dset_anom_nsa_[['t2m_x_nsa', 'z_nsa']]
    obj_clim_std = dset_clim_std_
    obj_targets = xr.Dataset(data_vars={'t2m':(('time'), t2m_), 'hw_bin_2in7':(('time'), hw_bin_2in7_), 'hw_bin_1SD':(('time'), hw_bin_1SD_),
                                           'hw_bin_15SD':(('time'), hw_bin_15SD_)}, 
                                coords={'time': (t2m_.time)})
    
    #Save
    save_to_file(obj_sa, dictionary['path_input_prepro_pred'], save_name_sa, 'nc')
    save_to_file(obj_nsa, dictionary['path_input_prepro_pred'], save_name_nsa, 'nc')
    save_to_file(obj_clim_std, dictionary['path_input_prepro_pred'], save_name_clim_std, 'nc')
    save_to_file(obj_targets, dictionary['path_input_targ'], save_name_targets, 'nc')


# **----------------------------------------------------------------------------------------------------------------------------------------------**

# **----------------------------------------------------------------------------------------------------------------------------------------------**

# **----------------------------------------------------------------------------------------------------------------------------------------------**

# ## Main
# ### Data preprocessing (part 1)

# In[7]:


# 1. Import data
t2m_raw, z_raw, sm_raw, sst_raw, rain_raw, t2m_x_raw, rain_eraint_raw, t2m_x_eraint_raw = import_data()


# In[ ]:


# 2. North Atlantic patterns
snao_sa, sea_sa = compute_north_atlantic_pca(z_raw)


# In[ ]:


# 3. Predictors preprocessing
## 3.1. Space: select lat-lon boxes & take spatial averages 
(t2m_sa, z_sa, sm_all_levels_sa, sst_med_sa, sst_nwmed_sa, sst_semed_sa, sst_baltic_sa, sst_cbna_sa, sst_nwna_sa, 
sst_sena_sa, sst_enso_sa, rain_sa, rain_pac_sa, rain_carib_sa, t2m_x_sa, t2m_x_nsa, 
z_nsa) = box_selection(t2m_raw, z_raw, sm_raw, sst_raw, rain_raw, t2m_x_raw, rain_eraint_raw, t2m_x_eraint_raw)


# In[ ]:


# ---------------------------------------------------------------------------------------------------------------------------------------#    
## 3.2. Level: vertical level selection & combination
sm_sa, sm_deeper_sa, sm_deepest_sa = select_vert_level(sm_all_levels_sa)  


# In[ ]:


# ---------------------------------------------------------------------------------------------------------------------------------------#
## 3.3. Dataset: homogenize variables in time and construct datasets
dset_sa, dset_nsa = construct_dataset(t2m_sa, z_sa, sm_sa, sm_deeper_sa, sm_deepest_sa, snao_sa, sea_sa, sst_med_sa, sst_nwmed_sa, sst_semed_sa, 
                sst_baltic_sa, sst_cbna_sa, sst_nwna_sa, sst_sena_sa, sst_enso_sa, rain_sa, rain_pac_sa, rain_carib_sa, 
                t2m_x_sa, t2m_x_nsa, z_nsa)


# In[ ]:


### Save in-between results to save time
save_to_file(dset_sa, dictionary['path_input_prepro_pred'], 'dset_sa_temp.nc', 'nc')
save_to_file(dset_nsa, dictionary['path_input_prepro_pred'], 'dset_nsa_temp.nc', 'nc')


# **---------------------------------------------------------------------------------------------------------------------------------------**

# In[26]:


### Open in-between results
print('Using ', dictionary['dataset'], ' data')
dset_sa = xr.open_dataset(dictionary['path_input_prepro_pred'] + 'dset_sa_temp.nc').load()
dset_nsa = xr.open_dataset(dictionary['path_input_prepro_pred'] + 'dset_nsa_temp.nc').load()


# In[27]:


# ---------------------------------------------------------------------------------------------------------------------------------------#
## 3.4. Pacific-Caribbean Dipole (PCD)
### Compute PCD index
dset_sa = compute_pcd_index(dset_sa)


# In[28]:


# ---------------------------------------------------------------------------------------------------------------------------------------#
## 3.5. Correct data jumps in soil moisture
dset_sa_c = correct_sm_data(dset_sa, method = 'interpolate')


# In[29]:


# ---------------------------------------------------------------------------------------------------------------------------------------#
## 3.6. Detrend the data
dset_sa_c_det = detrend(dset_sa_c, False)
dset_nsa_det = detrend(dset_nsa, False)


# In[30]:


# ---------------------------------------------------------------------------------------------------------------------------------------#
## 3.7. Standarized climatology
dset_clim_std_sa = compute_std_climatology(dset_sa_c_det)
dset_clim_std_nsa = compute_std_climatology(dset_nsa_det)


# In[31]:


# ---------------------------------------------------------------------------------------------------------------------------------------#
## 3.8. Daily anomalies w.r.t. climatology
dset_anom_sa = compute_anomalies_wrt_climatology(dset_sa_c_det)
dset_anom_nsa = compute_anomalies_wrt_climatology(dset_nsa_det)


# In[32]:


# ---------------------------------------------------------------------------------------------------------------------------------------#
## 3.9. Correct data jumps in SSTs
dset_anom_sa_c = correct_sst_data(dset_anom_sa, method = 'interpolate')


# In[33]:


# ---------------------------------------------------------------------------------------------------------------------------------------#        
## 3.10. Smooth out the data
dset_anom_sm_sa = smooth(dset_anom_sa_c)
dset_anom_sm_nsa = smooth(dset_anom_nsa)


# In[34]:


# ---------------------------------------------------------------------------------------------------------------------------------------#
## 3.11. Standarize
dset_anom_std_sa = standarize_data(dset_anom_sa_c)
dset_anom_sm_std_sa = standarize_data(dset_anom_sm_sa)
dset_anom_sm_std_nsa = standarize_data(dset_anom_sm_nsa)


# In[35]:


# ---------------------------------------------------------------------------------------------------------------------------------------#
## 3.12. Correct data jumps in SSTs that appeared after standarization
dset_anom_sm_std_sa_c = correct_sst_data(dset_anom_sm_std_sa, method = 'interpolate')


# In[36]:


# ---------------------------------------------------------------------------------------------------------------------------------------#        
## 3.13. North Atlantic Index
dset_anom_sm_std_sa = compute_na_index(dset_anom_sm_std_sa_c)


# In[37]:


if dictionary['verbosity'] > 2: show_multiple_time_series(dset_anom_sm_std_sa)


# **---------------------------------------------------------------------------------------------------------------------------------------**

# **---------------------------------------------------------------------------------------------------------------------------------------**

# In[38]:


# 4. Target definition
## Make hw_bin_2in7 index
hw_bin_2in7 = compute_hw_bin_2in7(dset_anom_std_sa.t2m)
## Make 1SD index
hw_bin_1SD = compute_hw_bin_sd(dset_anom_sm_std_sa.t2m, 1)
## Make 1.5SD index
hw_bin_15SD = compute_hw_bin_sd(dset_anom_sm_std_sa.t2m, 1.5)


# **---------------------------------------------------------------------------------------------------------------------------------------**

# **---------------------------------------------------------------------------------------------------------------------------------------**

# In[39]:


# 5. Save predictors and targets
save_pred_and_targ(dset_anom_sm_std_sa, dset_anom_sm_std_nsa, dset_clim_std_sa, dset_anom_sm_sa.t2m, hw_bin_2in7, hw_bin_1SD, hw_bin_15SD)

