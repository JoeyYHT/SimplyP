# ----------------------------------------------------------------------- 
# Name:     inputs.py
# Purpose:  Read in input parameters to set up and run the model.
#           Read in meteorological data, and if desired calculate derived
#           data to run the model.

# Author:   Leah Jackson-Blake
# Created:  06/11/2018
# Copyright:(c) Leah Jackson-Blake and NIVA, 2018
# Licence:
# -----------------------------------------------------------------------
""" Read in and process input parameters and data
"""

import pandas as pd, numpy as np
import calendar
import math

def read_input_data(params_fpath):
    """ Read SimplyP setup data from Excel template.
    
    Args:
        params_fpath: Raw str. Path to completed Excel input template.
        
    Returns:
        Tuple (p_SU, dynamic_options, p, p_LU, p_SC, p_struc, met_df, obs_dict).
        
        Parameter values. Indices are the parameter names (which match the input parameter sheet).
        Values are the parameters. 
        p_SU:            Series. Setup parameters
        dynamic_options: Series. Subset of dynamic setup parameters
        p:               Series. Parameters which are constant over land use and sub-catchment/reach
        p_LU:            Dataframe. Land use parameters. One column per land use type ('A','S','IG','NC')
        p_SC:            Dataframe. Sub-catchment and reach parameters. One column per sub-catchment/reach
        p_struc:         Dataframe. 
        met_df:          Dataframe. Meteorological data and data derived from it
                         (if desired, including results from snow accumulation & melt module, PET)
        obs_dict:        Dict. Observed discharge and chemistry data.
                         Keys: reach number. Values: Dataframe with datetime index and columns for water quality
                         variables. Columns are all 
    """
    # ----------------------------------------------------------------------------------------
    # USER SET-UP PARAMETERS
    p_SU = pd.read_excel(params_fpath, sheet_name='Setup', index_col=0, usecols="A,C")
    p_SU = p_SU['Value'] # Convert to a series

    # Extract user set-up parameters for dynamic dict
    # **dynamic terrestrial P inputs and effluent inputs not yet implemented**
    dynamic_options = p_SU[['Dynamic_EPC0', 'Dynamic_effluent_inputs',
                            'Dynamic_terrestrialP_inputs','Dynamic_erodibility']]

    # ----------------------------------------------------------------------------------------
    # MODEL PARAMETERS

    # CONSTANT PARAMS: Parameters that're constant over land use, sub-catchment or reach.
    # Values in col 'Value'
    p = pd.read_excel(params_fpath, sheet_name='Constant', index_col=0, usecols="B,E")
    p = p['Value'] # Convert to a series

    # LAND USE PARAMETERS. Values in cols A,S,IG,NC
    p_LU = pd.read_excel(params_fpath, sheet_name='LU', index_col=0, usecols="B,E,F,G,H")

    # SUB-CATCHMENT & REACH PARAMETERS: Values in cols '1', '2',..
    # Some fiddling required to parse the right number of columns, according to the number of SCs
    p['SC_list'] = np.arange(1,p_SU.n_SC+1)
    lastCol = chr(ord('E')+p_SU.n_SC-1) # Last column in excel sheet to be parsed
    if p_SU.n_SC ==1:
        usecols_str = "B,E"
    else:
        usecols_str = "B,E:%s" %lastCol
    p_SC = pd.read_excel(params_fpath, sheet_name='SC_reach', index_col=0, usecols=usecols_str)
    
    # REACH STRUCTURE PARAMETERS
    # Describe which reaches flow into each other, and whether to sum fluxes to produce an input
    # to a water body (e.g. a lake or coastal zone)
    p_struc = pd.read_excel(params_fpath, sheet_name='Reach_structure', index_col=0, usecols="A,B,C")
    p_struc.columns = ['Upstream_SCs','In_final_flux?']  # Shorten column names
    
    # Easy to make some mistakes with the reach parameters, so add a couple of checks
    if p_SU.n_SC != len(p_struc['Upstream_SCs']):
        raise ValueError("The number of sub-catchments specified in your 'Setup' parameter sheet doesn't \nmatch the number of rows in your 'Reach_structure' sheet")
    if p_SU.n_SC != len(p_SC.columns):
        raise ValueError("The number of columns in your 'SC_reach' sheet should match the number of sub-catchments specified in your 'Setup' parameter sheet")
    
    # Print some output
    print ('Parameter values successfully read in')

    # -----------------------------------------------------------------------------------------
    # MET DATA
    # Assume constant met data over the catchment. This could be amended in the future.
    met_df = pd.read_csv(p_SU.metdata_fpath, parse_dates=True, dayfirst=True, index_col=0)
    met_df = met_df.truncate(before=p_SU.st_dt, after=p_SU.end_dt)  # Truncate to the desired period
    
    print ('Input meteorological data read in')
    
    # If desired, run SNOW MODULE
    if p_SU.inc_snowmelt == 'y':
        met_df = snow_hydrol_inputs(p['D_snow_0'], p['f_DDSM'], met_df)
        print ('Snow accumulation and melt module run to estimate snowmelt inputs to the soil')
    else:
        met_df.rename(columns={'Precipitation':'P'}, inplace=True)
    
    # If PET isn't in the input met data, calculate it using Thornthwaite's 1948 equation
    if 'PET' not in met_df.columns:
        met_df = daily_PET(latitude=p['latitude'], met_df=met_df)
        print ('PET estimated using the Thornthwaite method')
 
    # -----------------------------------------------------------------------------------------
    # OBSERVATIONS
    # If file paths provided, read in observed data
    
    # Read from excel files (one for each of Q and chem). Excel files should have one sheet per
    # sub-catchment/reach, numbered 1, 2, etc. Obs for each reach are read into a dataframe.
    # Each reach is stored as a separate df in obs_dict (key is the reach number, as an integer).
    # Units of Q: m3/s, Units of chemistry: mg/l
    
    # If a string has been provided for the Q observations, try reading in
    if isinstance(p_SU.Qobsdata_fpath, str):
        Qobs_xl = pd.ExcelFile(p_SU.Qobsdata_fpath)
        SC_with_Qobs = [int(x) for x in Qobs_xl.sheet_names]  # List of sub-catchments with Q data
        print ('Observed discharge data read in')
    else:
        SC_with_Qobs = []
        
    # If a string has been provided for water chem obs, try reading in    
    if isinstance(p_SU.chemObsData_fpath, str):
        chemObs_xl = pd.ExcelFile(p_SU.chemObsData_fpath)
        SC_with_chemObs = [int(x) for x in chemObs_xl.sheet_names]  # List of sub-catchments with chemistry data
        print ('Observed water chemistry data read in')
    else:
        SC_with_chemObs = []

    obs_dict = {}   # Key: sub-catchment number (1,2,...); only SCs with obs are included
                    # Returns dataframe of observed data (if any)
                    
    for SC in p['SC_list']:  # Loop through all sub-catchments being simulated
        df_li = []  # List of Q and chem dataframes for the reach; may be empty or have up to 2 dfs
                    
        if SC in SC_with_Qobs:
            Qobs_df = pd.read_excel(p_SU.Qobsdata_fpath, sheet_name=str(SC), index_col=0)
            Qobs_df = Qobs_df.truncate(before=p_SU.st_dt, after=p_SU.end_dt)
            df_li.append(Qobs_df)
                    
        if SC in SC_with_chemObs:
            chemObs_df = pd.read_excel(p_SU.chemObsData_fpath, sheet_name=str(SC), index_col=0)
            chemObs_df = chemObs_df.truncate(before=p_SU.st_dt, after=p_SU.end_dt)
            df_li.append(chemObs_df)
        
        # If this SC has observations, add it to the dictionary of observations (obs_dict)
        if len(df_li)>0:
            obs_df = pd.concat(df_li, axis=1)  # If have both Q & chem data, combine into one df
            obs_dict[SC] = obs_df  # Add to dictionary
    
    # -----------------------------------------------------------------------------------------   
    return (p_SU, dynamic_options, p, p_LU, p_SC, p_struc, met_df, obs_dict)

#########################################################################################
 
def snow_hydrol_inputs(D_snow_0, f_DDSM, met_df):
    """ Calculate snow accumulation and melt i.e. estimates total hydrological input to soil box
        as (rain + snowmelt). Source for priors for DDF:
        
            http://directives.sc.egov.usda.gov/OpenNonWebContent.aspx?content=17753.wba

        Future potential extensions:
        
            (1) Add options for how temperature is assumed to vary through the day, e.g. triangular or
                sinuosoidal variations to get a more accurate portrayal of the degree-days above the
                threshold
                
            (2) Consider setting ET to 0 when D_snow > 0

    Args:
        D_snow_0: Float. Initial snow depth (mm)
        f_DDSM:   Float. Degree-day factor for snow melt (mm/degree-day deg C)
        met_df:   Dataframe. Met data with cols T_air, PET, Precipitation
    
    Returns:
        met_df with additional columns [P_snow, P_rain, P_melt, D_snow_start, D_snow_end, P].
        Of these, P is the hydrological input to the soil store (mm/d)
    """    
    # Precipitation falling as snow (mm/d, as water equivalents)
    met_df.loc[:,'P_snow'] = met_df['Precipitation'].ix[met_df['T_air']<0]  # = total pptn if air T<0
    met_df['P_snow'].fillna(0, inplace=True)  # otherwise, =0
    
    # Precipitation falling as rain (mm/d)
    met_df['P_rain'] = met_df['Precipitation'] - met_df['P_snow']

    # Potential daily snow melt (unlimited by snow pack depth) (mm/day)
    met_df['P_melt'] = f_DDSM*(met_df['T_air']-0)
    met_df['P_melt'][met_df['P_melt']<0]=0  # Set negative values to 0 (i.e. only melt when T_air>0)

    # Snow pack depth (mm), as end of day depth = start of day depth + inputs - melt, where melt is
    # limited by the depth wherever necessary.
    met_df['D_snow_start'], met_df['D_snow_end'] = np.nan, np.nan # Set-up
    # First time-step manually, to take initial condition into account
    met_df.ix[0,'D_snow_start'] = D_snow_0 # Assign user-supplied starting depth to first row
    met_df.ix[0,'P_melt'] = np.minimum(met_df.ix[0,'P_melt'],met_df.ix[0,'D_snow_start']) # Melt limited by depth
    met_df.ix[0,'D_snow_end'] = (met_df.ix[0,'D_snow_start']+
                                met_df.ix[0,'P_snow']-met_df.ix[0,'P_melt']) # Change over day
    # Calculate for subsequent days
    for idx in range (1,len(met_df)):
        met_df.ix[idx,'D_snow_start'] = met_df.ix[idx-1,'D_snow_end']
        met_df.ix[idx,'P_melt'] = np.minimum(met_df.ix[idx,'P_melt'],met_df.ix[idx,'D_snow_start'])
        met_df.ix[idx,'D_snow_end'] = met_df.ix[idx,'D_snow_start']+met_df.ix[idx,'P_snow']-met_df.ix[idx,'P_melt']

    # Hydrological input to soil box
    met_df.loc[:,'P'] = met_df['P_rain'] + met_df['P_melt']
    
    return met_df


# ###############################################################################################################

# Functions for estimating daily Thornthwaite potential evapotranspiration (PET) from monthly estimates
# Calculate potential evapotranspiration using the Thornthwaite (1948 method)
# Many functions copied from https://github.com/woodcrafty/PyETo/blob/master/pyeto
# :copyright: (c) 2015 by Mark Richards.
# :license: BSD 3-Clause 

# Nice comparison of some different PET methods, inc. Thornthwaite:
# Xu and Singh (2001), Hydrol. Proc.
# http://folk.uio.no/chongyux/papers_SCI/HYP_5.pdf
# ----------
# Thornthwaite CW (1948) An approach toward a rational classification of
# climate. Geographical Review, 38, 55-94.

# Set up
_MONTHDAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_LEAP_MONTHDAYS = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

def daily_PET(latitude, met_df):
    """
    Calculate daily PET values by linearly interpolating monthly PET
    values (see documentation above and below), centred on the middle
    of the month

    Input:
    latitude: User-supplied parameter (float), in degrees
    met_df: Dataframe containing datetime index and column 'T_air'

    Returns:
    met_df: Dataframe with extra column 'PET' (units mm/day)
    """

    # Convert latitude from deg to radians
    latitude = deg2rad(latitude)

    # Calculate mean number of daylight hours for a leap and a non-leap year
    monthly_mean_dlh = monthly_mean_daylight_hours(latitude, year=1983)
    monthly_mean_dlh_leap = monthly_mean_daylight_hours(latitude, year=1984)

    # Resample daily temperature to monthly means for the whole series
    T_m = met_df['T_air'].resample('M').mean()

    # Empty PET list to be populated with monthly values for all years
    PET_m = []

    # Group T_m by year then start looping through the years
    yr_grped = T_m.groupby(T_m.index.year)

    for group in yr_grped:

        year = group[0]  # Current year

        # Current monthly mean temperatures (numpy array length 12)
        T_m = group[1].values
        if len(T_m)<12: # Check have 12 values, raise error if not
            raise ValueError('PET calc requires input met data for whole calendar years.'
                             'Year {0!r} does not contain 12 months. Check input met data,'
                             'or change the start/end dates in the parameter file'.format(year))

        # Pick appropriate mean daylight hour series to use according to year
        if year is None or not calendar.isleap(year):
            monthly_mean_dlh = monthly_mean_dlh
        else:
            monthly_mean_dlh = monthly_mean_dlh_leap

        # Calculate PET for this year and append to list for all years (units mm/month)
        PET_m_i = annual_thornthwaite(T_m, monthly_mean_dlh, year=year)
        PET_m.extend(PET_m_i) # Add elements from list 2 to list 1

    # Resample monthly PET to daily

    # Create monthly datetime index, with an offset so it's the middle of the month
    # (then we'll interpolate between monthly values, to give a smoother curve than
    # just forward filling)
    start_date = met_df.index[0].date()
    end_date = met_df.index[-1].date()
    dt_index_m = pd.DatetimeIndex(freq='MS', start=start_date,
                                  end=end_date, dayfirst=True)
    dt_index_m_offset = dt_index_m + pd.DateOffset(days=15)

    # Make dataframe using monthly PET data and datetime index just created
    PET_m_df = pd.DataFrame(data=PET_m, index=dt_index_m_offset, columns=['PET'])

    # Divide the monthly PET values by number of days in the month to get units mm/day
    PET_m_df['PET'] = PET_m_df['PET'] / PET_m_df.index.daysinmonth

    # Recast to daily (by joining to met_df)
    # If already have PET in met_df, replace it with the new calculation
    if 'PET' not in met_df.columns:
        met_df = met_df.join(PET_m_df)
    else:
        met_df.drop(['PET'], axis=1, inplace=True)
        met_df = met_df.join(PET_m_df)

    # Linearly interpolate to fill NaNs
    met_df['PET'] = met_df['PET'].interpolate(method='linear', limit=32,
                                                       limit_direction='both')
    
    return met_df


def deg2rad(degrees):
    """
    Convert angular degrees to radians
    :param degrees: Value in degrees to be converted.
    :return: Value in radians
    :rtype: float
    """
    return degrees * (math.pi / 180.0)

def check_latitude_rad(latitude):
    # Lat range in radians
    _MINLAT_RADIANS = deg2rad(-90.0)
    _MAXLAT_RADIANS = deg2rad(90.0)
    # Check
    if not _MINLAT_RADIANS <= latitude <= _MAXLAT_RADIANS:
        raise ValueError(
            'latitude outside valid range {0!r} to {1!r} rad: {2!r}'
            .format(_MINLAT_RADIANS, _MAXLAT_RADIANS, latitude))
        
def check_doy(doy):
    """
    Check day of the year is valid.
    """
    if not 1 <= doy <= 366:
        raise ValueError(
            'Day of the year (doy) must be in range 1-366: {0!r}'.format(doy))

def check_sunset_hour_angle_rad(sha):
    """
    Sunset hour angle has the range 0 to 180 degrees.
    See http://mypages.iit.edu/~maslanka/SolarGeo.pdf
    """
    # Sunset hour angle
    _MINSHA_RADIANS = 0.0
    _MAXSHA_RADIANS = deg2rad(180)
    # Check
    if not _MINSHA_RADIANS <= sha <= _MAXSHA_RADIANS:
        raise ValueError(
            'sunset hour angle outside valid range {0!r} to {1!r} rad: {2!r}'
            .format(_MINSHA_RADIANS, _MAXSHA_RADIANS, sha))
        
def check_sol_dec_rad(sd):
    """
    Solar declination can vary between -23.5 and +23.5 degrees.
    See http://mypages.iit.edu/~maslanka/SolarGeo.pdf
    """
    # Solar declination
    _MINSOLDEC_RADIANS = deg2rad(-23.5)
    _MAXSOLDEC_RADIANS = deg2rad(23.5)
    # Check
    if not _MINSOLDEC_RADIANS <= sd <= _MAXSOLDEC_RADIANS:
        raise ValueError(
            'solar declination outside valid range {0!r} to {1!r} rad: {2!r}'
            .format(_MINSOLDEC_RADIANS, _MAXSOLDEC_RADIANS, sd))

def sol_dec(day_of_year):
    """
    Calculate solar declination from day of the year.
    Based on FAO equation 24 in Allen et al (1998).
    :param day_of_year: Day of year integer between 1 and 365 or 366).
    :return: solar declination [radians]
    :rtype: float
    """
    check_doy(day_of_year)
    return 0.409 * math.sin(((2.0 * math.pi / 365.0) * day_of_year - 1.39))

def sunset_hour_angle(latitude, sol_dec):
    """
    Calculate sunset hour angle (*Ws*) from latitude and solar
    declination.
    Based on FAO equation 25 in Allen et al (1998).
    :param latitude: Latitude [radians]. Note: *latitude* should be negative
        if it in the southern hemisphere, positive if in the northern
        hemisphere.
    :param sol_dec: Solar declination [radians]. Can be calculated using
        ``sol_dec()``.
    :return: Sunset hour angle [radians].
    :rtype: float
    """
    check_latitude_rad(latitude)
    check_sol_dec_rad(sol_dec)
    cos_sha = -math.tan(latitude) * math.tan(sol_dec)
    # If tmp is >= 1 there is no sunset, i.e. 24 hours of daylight
    # If tmp is <= 1 there is no sunrise, i.e. 24 hours of darkness
    # See http://www.itacanet.org/the-sun-as-a-source-of-energy/
    # part-3-calculating-solar-angles/
    # Domain of acos is -1 <= x <= 1 radians (this is not mentioned in FAO-56!)
    return math.acos(min(max(cos_sha, -1.0), 1.0))

def daylight_hours(sha):
    """
    Calculate daylight hours from sunset hour angle.
    Based on FAO equation 34 in Allen et al (1998).
    :param sha: Sunset hour angle [rad]. Can be calculated using
        ``sunset_hour_angle()``.
    :return: Daylight hours.
    :rtype: float
    """
    check_sunset_hour_angle_rad(sha)
    return (24.0 / math.pi) * sha

def monthly_mean_daylight_hours(latitude, year=None):
    """
    Calculate mean daylight hours for each month of the year for a given
    latitude.
    :param latitude: Latitude [radians]
    :param year: Year for the daylight hours are required. The only effect of
        *year* is to change the number of days in Feb to 29 if it is a leap
        year. If left as the default, None, then a normal (non-leap) year is
        assumed.
    :return: Mean daily daylight hours of each month of a year [hours]
    :rtype: List of floats.
    """
    check_latitude_rad(latitude)

    if year is None or not calendar.isleap(year):
        month_days = _MONTHDAYS
    else:
        month_days = _LEAP_MONTHDAYS
    monthly_mean_dlh = []
    doy = 1         # Day of the year
    for mdays in month_days:
        dlh = 0.0   # Cumulative daylight hours for the month
        for daynum in range(1, mdays + 1):
            sd = sol_dec(doy)
            sha = sunset_hour_angle(latitude, sd)
            dlh += daylight_hours(sha)
            doy += 1
        # Calc mean daylight hours of the month
        monthly_mean_dlh.append(dlh / mdays)
    return monthly_mean_dlh

def annual_thornthwaite(monthly_t, monthly_mean_dlh, year=None):
    """
    Estimate monthly potential evapotranspiration (PET) using the
    Thornthwaite (1948) method for 12 months in a single year
    
    Thornthwaite equation:
        *PET* = 1.6 (L/12) (N/30) (10 Ta / I)**a
    where:
    * *Ta* is the mean daily air temperature [deg C, if negative use 0] of the
      month being calculated
    * *N* is the number of days in the month being calculated
    * *L* is the mean day length [hours] of the month being calculated
    * *a* = (6.75 x 10-7)*I***3 - (7.71 x 10-5)*I***2 + (1.792 x 10-2)*I* + 0.49239
    * *I* is a heat index which depends on the 12 monthly mean temperatures and
      is calculated as the sum of (*Tai* / 5)**1.514 for each month, where
      Tai is the air temperature for each month in the year
    :param monthly_t: Iterable containing mean daily air temperature for each
        month of the year [deg C].
    :param monthly_mean_dlh: Iterable containing mean daily daylight
        hours for each month of the year (hours]. These can be calculated
        using ``monthly_mean_daylight_hours()``.
    :param year: Year for which PET is required. The only effect of year is
        to change the number of days in February to 29 if it is a leap year.
        If it is left as the default (None), then the year is assumed not to
        be a leap year.
    :return: Estimated monthly potential evaporation of each month of the year
        [mm/month]
    :rtype: List of floats
    """
    # Checks
    if len(monthly_t) != 12:
        raise ValueError(
            'monthly_t should be length 12 but is length {0}.'
            .format(len(monthly_t)))
    if len(monthly_mean_dlh) != 12:
        raise ValueError(
            'monthly_mean_dlh should be length 12 but is length {0}.'
            .format(len(monthly_mean_dlh)))

    if year is None or not calendar.isleap(year):
        month_days = _MONTHDAYS
    else:
        month_days = _LEAP_MONTHDAYS

    # Negative temperatures should be set to zero
    adj_monthly_t = [t * (t >= 0) for t in monthly_t]

    # Calculate the heat index (I)
    I = 0.0
    for Tai in adj_monthly_t:
        if Tai / 5.0 > 0.0:
            I += (Tai / 5.0) ** 1.514

    a = (6.75e-07 * I ** 3) - (7.71e-05 * I ** 2) + (1.792e-02 * I) + 0.49239

    pet = []
    for Ta, L, N in zip(adj_monthly_t, monthly_mean_dlh, month_days):
        # Multiply by 10 to convert cm/month --> mm/month
        pet.append(
            1.6 * (L / 12.0) * (N / 30.0) * ((10.0 * Ta / I) ** a) * 10.0)

    return pet
