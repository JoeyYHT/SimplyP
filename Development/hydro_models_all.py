"""
Created on Fri Nov 06 09:56:26 2015
@author: lj40184

List of hydrology model functions tried and tested as part of water quality
model development
"""

# Import modules
import numpy as np, pandas as pd
from scipy.integrate import odeint

#%% (standard cell separator)

# VERSION 1: two linear reservoirs (soil and groudnwater). Soil water flow only
# when soil water level is above field capacity

# Result: Pretty good, but not good at simultaing the recession or low flows.
# Groundwater 'drying out' too quickly.

def hydro_model_1(met_df, ics, p, period, step_len=1):
    """ The hydrological model

            met_df         Dataframe containing columns 'Rainfall_mm' and 'PET_mm', with datetime index
            ics            Vector of initial conditions [Vs0, Vg0]
            p              Series of parameter values (index = param name)
            period         Vector of [start, end] dates [yyyy-mm-dd, yyyy-mm-dd]
            step_len       Length of each step in the input dataset (days)

        Returns a dataframe with column headings
        [Vs, Qs, Qg, Ds, Dg, Sim_Runoff, Obs_Runoff]
    """
    # ------------------------------------------------------------------------
    # Define the ODE system
    def f(y, t, ode_params):
        """ Define ODE system.
                y is list [Vs, Qs, Qg, Ds, Dg]
                t is an array of time points of interest
                params is a tuple of input values & model params
                (P, E, f_IExcess, alpha, beta, T_s, T_g, fc)
        """
        # Unpack incremental values for Qs and Qg
        Vs_i = y[0]
        Qs_i = y[1]
        Qg_i = y[2]
        
        # Unpack params
        P, E, f_IExcess, alpha, beta, T_s, T_g, fc = ode_params
    
        # Model equations
        dQs_dV = (((Vs_i - fc)*np.exp(fc - Vs_i))/(T_s*((np.exp(fc-Vs_i) + 1)**2)))
        +(1/(T_s*(np.exp(fc-Vs_i) + 1)))
        dVs_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*Vs_i)) - Qs_i
        dQs_dt = dQs_dV*dVs_dt
        dQg_dt = (beta*Qs_i - Qg_i)/T_g
        dDs_dt = (1 - beta)*Qs_i
        dDg_dt = Qg_i
        
        # Add results of equations to an array
        res = np.array([dVs_dt, dQs_dt, dQg_dt, dDs_dt, dDg_dt])
        
        return res
    # -------------------------------------------------------------------------

    # Unpack initial conditions
    Vs0, Vg0 = ics

    # Time points to evaluate ODEs at. We're only interested in the start and
    # the end of each step
    ti = [0, step_len]

    # Lists to store output
    output_ODEs = []
    output_rest = []

    # Loop over met data
    for idx in range(len(met_df)):

        # Get P and E for this day
        P = met_df.ix[idx, 'P']
        E = met_df.ix[idx, 'PET']

        # Calculate infiltration excess and add to results
        Qq = p['f_IExcess']*P
        output_rest.append(Qq)

        # Calculate Qs0 and Qg0 from Vs0 and Vg0
        Qs0 = (Vs0 - p['fc'])/(p['T_s']*(1 + np.exp(p['fc'] - Vs0)))
        Qg0 = Vg0/p['T_g']

        # Vector of initial conditions
        y0 = [Vs0, Qs0, Qg0, 0., 0.]

        # Model parameters plus rainfall and ET, for input to solver
        ode_params = np.array([P, E, p['f_IExcess'], p['alpha'], p['beta'], p['T_s'],
                               p['T_g'], p['fc']])

        # Solve
        y = odeint(f, y0, ti, args=(ode_params,))

        # Extract values for end of step
        res = y[1]

        # Numerical errors may result in very tiny values <0
        # set these back to 0
        res[res<0] = 0
        output_ODEs.append(res)

        # Update initial conditions for next step
        Vs0 = res[0]
        Vg0 = res[2]*p['T_g']

    # Build a dataframe of ODE results
    df1 = pd.DataFrame(data=np.vstack(output_ODEs),
                      columns=['Vs', 'Qs', 'Qg', 'Ds', 'Dg'],
                      index=met_df.index)

    # Dataframe of non ODE results
    df2 = pd.DataFrame(data=np.vstack(output_rest), columns=['Qq'],
                     index=met_df.index)

    # Concatenate results dataframes
    df = pd.concat([df1,df2], axis=1)

    # Estimate runoff as Ds + Dg
    df['Sim_Runoff_mm_IE'] = df['Ds'] + df['Dg'] + df['Qq']
    df['Sim_Runoff_mm'] = df['Ds'] + df['Dg']

    return df

#%% (standard cell separator)

# VERSION 2: Same as version 1, but with a non-linear groundwater reservoir
# (Vg = T*Qg^b, as described by e.g. Wittenberg '99)

# Result: if set the exponent > 1, then the groundwater is better simulated,
# but still not good. Need it to better linked to soil water level (even when
# soil water is less than field capacity). Also, the values of b that help the
# simulation look more realistic are nowhere near what the literature says they
# should be (<1, around 0.5). Though makes me wonder if I'm getting confused 
# about the constants, therefore look into this more if decide to adopt this.

def hydro_model_2(met_df, ics, p, period, step_len=1):
    """ The hydrological model
            met_df         Dataframe containing columns 'Rainfall_mm' and 'PET_mm',
                           with datetime index
            ics            Vector of initial conditions [Vs0, Vg0]
            p              Series of parameter values (index = param name)
                           Has additional parameter k_g (exponent in v=aQ^k_g)
            period         Vector of [start, end] dates [yyyy-mm-dd, yyyy-mm-dd]
            step_len       Length of each step in the input dataset (days)

        Returns a dataframe with column headings
        [Vs, Qs, Vg, Qg, Ds, Dg, Sim_Runoff, Obs_Runoff]
    """
    
    # Function defining the ODE system, to be solved by SCIPY.INTEGRATE.ODEINT
    # at each time step
    def f(y, t, ode_params):
        """ Define ODE system.
                y is list [Vs, Qs, Vg, Qg, Ds, Dg]
                t is an array of time points of interest
                params is a tuple of input values & model params
                (P, E, f_IExcess, alpha, beta, T_s, T_g, fc, k_g)
        """
        # Unpack incremental values for Qs and Qg
        Vs_i = y[0]
        Qs_i = y[1]
        Vg_i = y[2]
        Qg_i = y[3]

        # Unpack params
        P, E, f_IExcess, alpha, beta, T_s, T_g, fc, k_g = ode_params

        # Soil equations
        dQs_dVs = (((Vs_i - fc)*np.exp(fc - Vs_i))/(T_s*((np.exp(fc-Vs_i) + 1)**2))) + (1/(T_s*(np.exp(fc-Vs_i) + 1)))
        dVs_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*Vs_i)) - Qs_i
        dQs_dt = dQs_dVs*dVs_dt

        # Groundwater equations
        dQg_dVg = (Vg_i**((1-k_g)/k_g))/(k_g*(T_g**(1/k_g)))
        dVg_dt = beta*Qs_i - (Vg_i/T_g)**1/k_g
        dQg_dt = dQg_dVg*dVg_dt

        # Total drainage volumes over the timestep
        dDs_dt = (1 - beta)*Qs_i
        dDg_dt = Qg_i

        # Add results of equations to an array
        res = np.array([dVs_dt, dQs_dt, dVg_dt, dQg_dt, dDs_dt, dDg_dt])

        return res
    # -------------------------------------------------------------------------

    # Unpack initial conditions
    Vs0, Vg0 = ics

    # Time points to evaluate ODEs at. We're only interested in the start and the end of each step
    ti = [0, step_len]

    # Lists to store output
    output_ODEs = []
    output_rest = []

    # Loop over met data
    for idx in range(len(met_df)):

        # Get P and E for this day
        P = met_df.ix[idx, 'P']
        E = met_df.ix[idx, 'PET']

        # Calculate infiltration excess and add to results
        Qq = p['f_IExcess']*P
        output_rest.append(Qq)

        # Calculate Qs0 and Qg0 from Vs0 and Vg0
        Qs0 = (Vs0 - p['fc'])/(p['T_s']*(1 + np.exp(p['fc'] - Vs0)))
        Qg0 = (Vg0/p['T_g'])**(1/p['k_g'])

        # Vector of initial conditions (Ds and Dg always 0 at start of time step)
        y0 = [Vs0, Qs0, Vg0, Qg0, 0., 0.]

        # Model parameters plus rainfall and ET, for input to solver
        ode_params = np.array([P, E, p['f_IExcess'], p['alpha'], p['beta'], p['T_s'],
                               p['T_g'], p['fc'], p['k_g']])

        # Solve
        y = odeint(f, y0, ti, args=(ode_params,))

        # Extract values for end of step
        res = y[1]

        # Numerical errors may result in very tiny values <0
        # set these back to 0
        res[res<0] = 0
        output_ODEs.append(res)

        # Update initial conditions for next step
        Vs0 = res[0]
        Vg0 = res[2]

    # Build a dataframe of ODE results
    df1 = pd.DataFrame(data=np.vstack(output_ODEs),
                      columns=['Vs', 'Qs', 'Vg', 'Qg', 'Ds', 'Dg'],
                      index=met_df.index)

    # Dataframe of non ODE results
    df2 = pd.DataFrame(data=np.vstack(output_rest), columns=['Qq'],
                     index=met_df.index)

    # Concatenate results dataframes
    df = pd.concat([df1,df2], axis=1)

    # Estimate runoff as Ds + Dg
    df['Sim_Runoff_mm_IE'] = df['Ds'] + df['Dg'] + df['Qq']
    df['Sim_Runoff_mm'] = df['Ds'] + df['Dg']
    
    return df

#%% (standard cell separator)

# VERSION 3: Allow runoff from the soil box when the soil water level is below
# field capacity, using the same function to limit its value as is used for the
# AET calculation. Allows runoff to the stream and percolation to groundwater
# when the soil water is below field capacity, but at a reduced rate.

# Result: Too much lost from the soil water, and therefore groundwater and soil
# water flows are too high, resulting in discharge which is also too high. The
# only way to make discharge around the right level is to reduce field capacity
# to values which essentially stop it physically representing field capacity
# any more (e.g. 10mm), but even then the fit isn't very good.

# Conclude: Need soil water flow to be very limited below field capacity.

def hydro_model_3(met_df, ics, p, period, step_len=1):
    """ The hydrological model

            met_df         Dataframe containing columns 'Rainfall_mm' and 'PET_mm', with datetime index
            ics            Vector of initial conditions [Vs0, Vg0]
            p              Series of parameter values (index = param name)
            period         Vector of [start, end] dates [yyyy-mm-dd, yyyy-mm-dd]
            step_len       Length of each step in the input dataset (days)

        Returns a dataframe with column headings
        [Vs, Qs, Qg, Ds, Dg, Sim_Runoff, Obs_Runoff]
    """
    # ------------------------------------------------------------------------
    # Define the ODE system
    def f(y, t, ode_params):
        """ Define ODE system.
                y is list [Vs, Qs, Qg, Ds, Dg]
                t is an array of time points of interest
                params is a tuple of input values & model params
                (P, E, f_IExcess, alpha, beta, T_s, T_g, fc)
        """
        # Unpack incremental values for Qs and Qg
        Vs_i = y[0]
        Qs_i = y[1]
        Qg_i = y[2]
        
        # Unpack params
        P, E, f_IExcess, alpha, beta, T_s, T_g, fc, k = ode_params
           
        # Soil water equations
        dVs_dt = P*(1-f_IExcess) - alpha*E*(1-np.exp(-k*Vs_i)) - (Vs_i/T_s)*(1-np.exp(-k*Vs_i))       
        dQs_dV = np.exp(-k*(Vs_i**2))*(2*k*(Vs_i**2)+np.exp(k*(Vs_i**2))-1)*(1/T_s)
        dQs_dt = dQs_dV*dVs_dt
        
        # Groundwater equations
        dQg_dt = (beta*Qs_i - Qg_i)/T_g
        
        # Total drainage volumes over the timestep
        dDs_dt = (1 - beta)*Qs_i
        dDg_dt = Qg_i
        
        # Add results of equations to an array
        res = np.array([dVs_dt, dQs_dt, dQg_dt, dDs_dt, dDg_dt])
        
        return res
    # -------------------------------------------------------------------------

    # Unpack initial conditions
    Vs0, Vg0 = ics

    # Time points to evaluate ODEs at. We're only interested in the start and
    # the end of each step
    ti = [0, step_len]

    # Lists to store output
    output_ODEs = []
    output_rest = []
    
    # Calculate the value of the exponential shape parameter, k
    k = np.log(0.1)/-p['fc']

    # Loop over met data
    for idx in range(len(met_df)):

        # Get P and E for this day
        P = met_df.ix[idx, 'P']
        E = met_df.ix[idx, 'PET']

        # Calculate infiltration excess and add to results
        Qq = p['f_IExcess']*P
        output_rest.append(Qq)

        # Calculate Qs0 and Qg0 from Vs0 and Vg0
        Qs0 = Vs0*(1/p['T_s'])*(1-np.exp(-k*Vs0))
        Qg0 = Vg0/p['T_g']

        # Vector of initial conditions (Ds and Dg always 0 at start of time step)
        y0 = [Vs0, Qs0, Qg0, 0., 0.]

        # Model parameters plus rainfall and ET, for input to solver
        ode_params = np.array([P, E, p['f_IExcess'], p['alpha'], p['beta'], p['T_s'],
                               p['T_g'], p['fc'], k])

        # Solve
        y = odeint(f, y0, ti, args=(ode_params,))

        # Extract values for end of step
        res = y[1]

        # Numerical errors may result in very tiny values <0
        # set these back to 0
        res[res<0] = 0
        output_ODEs.append(res)

        # Update initial conditions for next step
        Vs0 = res[0]
        Vg0 = res[2]*p['T_g']  # Qg * T_g

    # Build a dataframe of ODE results
    df1 = pd.DataFrame(data=np.vstack(output_ODEs),
                      columns=['Vs', 'Qs', 'Qg', 'Ds', 'Dg'],
                      index=met_df.index)

    # Dataframe of non ODE results
    df2 = pd.DataFrame(data=np.vstack(output_rest), columns=['Qq'],
                     index=met_df.index)

    # Concatenate results dataframes
    df = pd.concat([df1,df2], axis=1)

    # Estimate runoff as Ds + Dg
    df['Sim_Runoff_mm_IE'] = df['Ds'] + df['Dg'] + df['Qq']
    df['Sim_Runoff_mm'] = df['Ds'] + df['Dg']

    return df
    
#%% (standard cell separator)
    
# VERSION 4: Exactly the same as Version 1 (two linear reservoirs), but with a
# fudge to prevent groundwater flow dropping below a user-specified threshold.

# NOTE: This version does not satisfy the water balance!!! It is just included
# as a quick fix to provide a model that can be compared with INCA hydrology
# simulations, and to allow chemical equations to be developed.

# A good parameter set for model 4 (just from rough playing) is:
param_dict = {'fc':290, 'beta':0.6, 'f_IExcess':0.015, 'alpha':0.90,
              'T_s':6.,'T_g':60., 'Qg_min':0.4}

def hydro_model_4(met_df, ics, p, period, step_len=1):
    """ The hydrological model

            met_df         Dataframe containing columns 'Rainfall_mm' and 'PET_mm', with datetime index
            ics            Vector of initial conditions [Vs0, Vg0]
            p              Series of parameter values (index = param name)
                           Includes the extra param q_gw_min
            period         Vector of [start, end] dates [yyyy-mm-dd, yyyy-mm-dd]
            step_len       Length of each step in the input dataset (days)

        Returns a dataframe with column headings
        [Vs, Qs, Qg, Ds, Dg, Sim_Runoff, Obs_Runoff]
    """
    # ------------------------------------------------------------------------
    # Define the ODE system
    def f(y, t, ode_params):
        """ Define ODE system.
                y is list [Vs, Qs, Qg, Ds, Dg]
                t is an array of time points of interest
                params is a tuple of input values & model params
                (P, E, f_IExcess, alpha, beta, T_s, T_g, fc)
        """
        # Unpack incremental values for Qs and Qg
        Vs_i = y[0]
        Qs_i = y[1]
        Qg_i = y[2]
        
        # Unpack params
        P, E, f_IExcess, alpha, beta, T_s, T_g, fc = ode_params
    
        # Model equations
        dQs_dV = (((Vs_i - fc)*np.exp(fc - Vs_i))/(T_s*((np.exp(fc-Vs_i) + 1)**2)))
        +(1/(T_s*(np.exp(fc-Vs_i) + 1)))
        dVs_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*Vs_i)) - Qs_i
        dQs_dt = dQs_dV*dVs_dt
        dQg_dt = (beta*Qs_i - Qg_i)/T_g
        dDs_dt = (1 - beta)*Qs_i
        dDg_dt = Qg_i
        
        # Add results of equations to an array
        res = np.array([dVs_dt, dQs_dt, dQg_dt, dDs_dt, dDg_dt])
        
        return res
    # -------------------------------------------------------------------------

    # Unpack initial conditions
    Vs0, Vg0 = ics

    # Time points to evaluate ODEs at. We're only interested in the start and
    # the end of each step
    ti = [0, step_len]

    # Lists to store output
    output_ODEs = []
    output_rest = []

    # Loop over met data
    for idx in range(len(met_df)):

        # Get P and E for this day
        P = met_df.ix[idx, 'P']
        E = met_df.ix[idx, 'PET']

        # Calculate infiltration excess and add to results
        Qq = p['f_IExcess']*P
        output_rest.append(Qq)

        # Calculate Qs0 and Qg0 from Vs0 and Vg0
        Qs0 = (Vs0 - p['fc'])/(p['T_s']*(1 + np.exp(p['fc'] - Vs0)))
        Qg0 = Vg0/p['T_g']

        # Vector of initial conditions
        y0 = [Vs0, Qs0, Qg0, 0., 0.]

        # Model parameters plus rainfall and ET, for input to solver
        ode_params = np.array([P, E, p['f_IExcess'], p['alpha'], p['beta'], p['T_s'],
                               p['T_g'], p['fc']])

        # Solve
        y = odeint(f, y0, ti, args=(ode_params,))

        # Extract values for end of step
        res = y[1]

        # Numerical errors may result in very tiny values <0
        # set these back to 0
        res[res<0] = 0
        output_ODEs.append(res)

        # Update initial conditions for next step
        Vs0 = res[0]
        Vg0 = res[2]*p['T_g']

    # Build a dataframe of ODE results
    df1 = pd.DataFrame(data=np.vstack(output_ODEs),
                      columns=['Vs', 'Qs', 'Qg', 'Ds', 'Dg'],
                      index=met_df.index)

    # Dataframe of non ODE results
    df2 = pd.DataFrame(data=np.vstack(output_rest), columns=['Qq'],
                     index=met_df.index)

    # Concatenate results dataframes
    df = pd.concat([df1,df2], axis=1)
    
    # SEEM TO HAVE CRUCIAL DIFFERENCE TO V1 MISSING!! HERE, SHOULD HAVE SOMETHING
    # WHICH SETS GROUNDWATER DRAINAGE TO THE MINIMUM GW Q IF IT'S BELOW IT!!

    # Estimate runoff as Ds + Dg
    df['Sim_Runoff_mm_IE'] = df['Ds'] + df['Dg'] + df['Qq']
    df['Sim_Runoff_mm'] = df['Ds'] + df['Dg']

    return df

#%% (standard cell separator)

# VERSION 5: Almost the same as Version 4, but with the addition of a single
# in-stream reach.
# Includes a fudge to prevent groundwater flow from dropping below a user-specified
# threshold (implemented in a slightly different way to Version 4)

# Result: Good. Without much effort to manually calibrate any of the parmaeters,
# have a NSE of 0.67 for 2004-2005 (old Coull rating), log-NSE of 0.77. i.e.
# comparable to manual calibration using INCA.

def hydro_model_5(met_df, ics, p, period, step_len=1):
    """ The hydrological model

            met_df         Dataframe containing columns 'Rainfall_mm' and 'PET_mm', with datetime index
            ics            Vector of initial conditions [Vs0, Qg0, Qr0]
            p              Series of parameter values (index = param name)
                           Includes the extra param q_gw_min
            period         Vector of [start, end] dates [yyyy-mm-dd, yyyy-mm-dd]
            step_len       Length of each step in the input dataset (days)

        Returns a dataframe with column headings
        ['Vs', 'Qs', 'Vg', 'Qg', 'Vr', 'Qr', 'Dr','Qq']
        (soil water volume and flow, groundwater volume and flow, reach volume
        and flow, mean average daily flow in reach, quick flow)
    """
    # ------------------------------------------------------------------------
    # Define the ODE system
    def f(y, t, ode_params):
        """ Define ODE system.
                y is list or variables for which we want to determine their value at the end of
                    the time step
                    [Vs, Qs, Vg, Qg, Vr, Qr, Dr]
                t is an array of time points of interest
                params is a tuple of input values & model params:
                    (P, E, Qq_i, f_IExcess, alpha, beta, T_s, T_g, fc, L_reach, a_Q, b_Q)
        """
        # Unpack incremental values for Qs and Qg from 
        Vs_i = y[0]
        Qs_i = y[1]
        Qg_i = y[3]
        Qr_i = y[5]
        
        # Unpack params
        P, E, Qq_i, f_IExcess, alpha, beta, T_s, T_g, fc, L_reach, a_Q, b_Q = ode_params
    
        # Soil equations
        dQs_dV = (((Vs_i - fc)*np.exp(fc - Vs_i))/(T_s*((np.exp(fc-Vs_i) + 1)**2)))
        +(1/(T_s*(np.exp(fc-Vs_i) + 1)))
        dVs_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*Vs_i)) - Qs_i
        dQs_dt = dQs_dV*dVs_dt
        
        # Groundwater equations
        dQg_dt = (beta*Qs_i - Qg_i)/T_g
        dVg_dt = beta*Qs_i - Qg_i
        
        # Instream equations       
        # NB factor in dQr_dt converts units of instream velocity (aQ^b) from m/s to mm/day; L_reach is in mm
        dQr_dt = ((Qq_i + (1-beta)*Qs_i + Qg_i) - Qr_i)* a_Q*(Qr_i**b_Q)*(8.64*10**7)/((1-b_Q)*L_reach)
        dVr_dt = (Qq_i + (1-beta)*Qs_i + Qg_i) - Qr_i
        dDr_dt = Qr_i
        
        # Add results of equations to an array
        res = np.array([dVs_dt, dQs_dt, dVg_dt, dQg_dt, dVr_dt, dQr_dt, dDr_dt])
        
        return res
    # -------------------------------------------------------------------------

    # Unpack initial conditions (initial soil water volume, groundwater flow, instream flow)
    Vs0, Qg0, Qr0 = ics

    # Time points to evaluate ODEs at. We're only interested in the start and
    # the end of each step
    ti = [0, step_len]

    # Lists to store output
    output_ODEs = []
    output_rest = []

    # Loop over met data
    for idx in range(len(met_df)):

        # Get P and E for this day
        P = met_df.ix[idx, 'P']
        E = met_df.ix[idx, 'PET']

        # Calculate infiltration excess and add to results
        Qq_i = p['f_IExcess']*P
        output_rest.append(Qq_i)

        # Calculate additional initial conditions from user-input initial conditions
        Qs0 = (Vs0 - p['fc'])/(p['T_s']*(1 + np.exp(p['fc'] - Vs0)))
        Vg0 = Qg0 *p['T_g']
        Vr0 = Qr0 * (p['L_reach']/(p['a_Q'])*Qr0**p['b_Q']) # i.e. V=QT, where T=L/aQ^b

        # Vector of initial conditions (adding 0 for Dr0, daily mean instream Q)
        y0 = [Vs0, Qs0, Vg0, Qg0, Vr0, Qr0, 0.0]

        # Model parameters plus rainfall and ET, for input to solver
        ode_params = np.array([P, E, Qq_i, p['f_IExcess'],p['alpha'], p['beta'],
                               p['T_s'], p['T_g'], p['fc'], p['L_reach'], p['a_Q'], p['b_Q']])

        # Solve
        y = odeint(f, y0, ti, args=(ode_params,))

        # Extract values for end of step
        res = y[1]

        # Numerical errors may result in very tiny values <0
        # set these back to 0
        res[res<0] = 0
        output_ODEs.append(res)

        # Update initial conditions for next step
        Vs0 = res[0]
        # FUDGE to re-set groundwater to user-supplied min flow at start of each time step!!!
        if param_dict['Qg_min'] > res[3]:
            Qg0 = p['Qg_min']
        else:
            Qg0 = res[3]
        Qr0 = res[5]

    # Build a dataframe of ODE results
    df1 = pd.DataFrame(data=np.vstack(output_ODEs),
                      columns=['Vs', 'Qs', 'Vg', 'Qg', 'Vr', 'Qr', 'Dr'],
                      index=met_df.index)
    
    # Dataframe of non ODE results
    df2 = pd.DataFrame(data=np.vstack(output_rest), columns=['Qq'],
                     index=met_df.index)

    # Concatenate results dataframes
    df = pd.concat([df1,df2], axis=1)

    return df