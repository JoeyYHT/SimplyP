# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:50:24 2015
@author: LJ40184

List of sediment model functions tried as part of water quality
model development
"""

# Import modules
import numpy as np, pandas as pd
from scipy.integrate import odeint

#%% (standard cell separator)

def sed_model_1(met_df, ics, p, period, step_len=1):
    """ Firs hydrology & sediment model

            met_df         Dataframe containing columns 'Rainfall_mm' and 'PET_mm', with datetime index
            ics            Vector of initial conditions [Vs0, Qg0, Qr0]
            p              Series of parameter values (index = param name)
                           Includes the extra param q_gw_min
            period         Vector of [start, end] dates [yyyy-mm-dd, yyyy-mm-dd]
            step_len       Length of each step in the input dataset (days)

        Returns a dataframe with column headings
        ['Vs', 'Qs', 'Vg', 'Qg', 'Vr', 'Qr', 'Dr','Qq', 'Mland']
        (soil water volume and flow, groundwater volume and flow, reach volume
        and flow, mean average daily flow in reach, quick flow)
        
        Future needs: Add in saturation excess to help simulate winter peaks
    """
    # ------------------------------------------------------------------------
    # Define the ODE system
    def ode_f(y, t, ode_params):
        """ Define ODE system.
                y is list or variables for which we want to determine their value at the end of
                    the time step
                    [Vs, Qs, Vg, Qg, Vr, Qr, Dr, Msus]
                t is an array of time points of interest
                params is a tuple of input values & model params:
                    (P, E, Qq_i, Mland_i, f_IExcess, alpha, beta, T_s, T_g, fc, L_reach,
                    a_Q, b_Q, E_Q, k_EQ)
        """
        # Unpack incremental values for initial conditions for this time step 
        Vs_i = y[0]
        Qs_i = y[1]
        Vg_i = y[2]
        Qg_i = y[3]
        Vr_i = y[4]
        Qr_i = y[5]
        Msus_i = y[7]
        
        # Unpack params
        (P, E, Qq_i, Mland_i, f_IExcess, alpha, beta, T_s,
         T_g, fc, L_reach, S_reach, a_Q, b_Q, E_Q, k_EQ) = ode_params
    
        # Soil equations (units mm or mm/day)
        dQs_dV = (((Vs_i - fc)*np.exp(fc - Vs_i))/(T_s*((np.exp(fc-Vs_i) + 1)**2)))
        +(1/(T_s*(np.exp(fc-Vs_i) + 1)))
        dVs_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*Vs_i)) - Qs_i
        dQs_dt = dQs_dV*dVs_dt
        
        # Groundwater equations (units mm or mm/day)
        dQg_dt = (beta*Qs_i - Qg_i)/T_g
        dVg_dt = beta*Qs_i - Qg_i
        
        # Instream equations (units m3 or m3/s)
        # Units: factors in dQr_dt convert units of instream velocity (aQ^b) from m/s to mm/s, the
        # time constant from s to days & L_reach from m to mm
        dQr_dt = ((Qq_i + (1-beta)*Qs_i + Qg_i) - Qr_i)* a_Q*(Qr_i**b_Q)*(8.64*10**7)/((1-b_Q)*(L_reach*1000))
        dVr_dt = (Qq_i + (1-beta)*Qs_i + Qg_i) - Qr_i
        dDr_dt = Qr_i
        
        # Instream suspended sediment (kg; change in kg/day)
        dMsus_dt = Mland_i + E_Q*S_reach*(Qr_i**k_EQ) - (Msus_i/Vr_i)*Qr_i #Units: (kg/day) - (kg/mm)*(mm/day)
        
        # Add results of equations to an array
        res = np.array([dVs_dt, dQs_dt, dVg_dt, dQg_dt, dVr_dt, dQr_dt, dDr_dt, dMsus_dt])
        
        return res
    # -------------------------------------------------------------------------

    # Unpack user-supplied initial conditions
    # (initial soil water volume, groundwater flow, instream flow)
    Vs0, Qg0, Qr0_m3s = ics
    # Convert units of Qr0 to mm/day
    Qr0 = UC_Qinv(Qr0_m3s, p['A_catch'])
    # Assume initial suspended sediment mass is 0 kg, and have short burn-in period
    Msus0 = 0.0

    # Time points to evaluate ODEs at. We're only interested in the start and
    # the end of each step
    ti = [0, step_len]

    # Lists to store output
    output_ODEs = []  # From ode_f function
    output_nonODE = []  # Will include: Qq, Mland

    # Loop over met data
    for idx in range(len(met_df)):

        # Get P and E for this day
        P = met_df.ix[idx, 'P']
        E = met_df.ix[idx, 'PET']

        # Calculate infiltration excess (mm/(day * catchment area))
        Qq_i = p['f_IExcess']*P
        
        # Calculate terrestrial erosion and delivery to the stream, Mland (kg/day)
        Mland_i = p['E_land']*(Qq_i**p['k_Eland']) #Units: (kg/mm)*(mm/day)
        
        # Append to results
        output_nonODE_i = [Qq_i, Mland_i]
        output_nonODE.append(output_nonODE_i)

        # Calculate additional initial conditions from user-input initial conditions
        # Soil and groundwater
        Qs0 = (Vs0 - p['fc'])/(p['T_s']*(1 + np.exp(p['fc'] - Vs0)))
        Vg0 = Qg0 *p['T_g']
        # Instream hydrol
        Tr0 = (1000*p['L_reach'])/(p['a_Q']*(Qr0**p['b_Q'])*(8.64*10**7)) #Reach time constant (days), where T=L/aQ^b
        Vr0 = Qr0*Tr0 # i.e. V=QT (mm)

        # Vector of initial conditions for start of time step
        # Assume 0 for Dr0 (daily mean instream Q)
        y0 = [Vs0, Qs0, Vg0, Qg0, Vr0, Qr0, 0.0, Msus0]

        # Model parameters plus rainfall and ET, for input to solver
        ode_params = np.array([P, E, Qq_i, Mland_i, p['f_IExcess'],p['alpha'], p['beta'],
                              p['T_s'], p['T_g'], p['fc'], p['L_reach'], p['S_reach'], p['a_Q'], p['b_Q'],
                              p['E_Q'], p['k_EQ']])

        # Solve
        y = odeint(ode_f, y0, ti, args=(ode_params,))

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
        Msus0 = res[7]

    # Build a dataframe of ODE results
    df1 = pd.DataFrame(data=np.vstack(output_ODEs),
                      columns=['Vs', 'Qs', 'Vg', 'Qg', 'Vr', 'Qr', 'Dr', 'Msus'],
                      index=met_df.index)
    
    # Dataframe of non ODE results
    df2 = pd.DataFrame(data=np.vstack(output_nonODE), columns=['Qq', 'Mland'],
                     index=met_df.index)

    # Concatenate results dataframes
    df = pd.concat([df1,df2], axis=1)

    return df

#%% (standard cell separator)

# Code snippets for analysing output from sediment model

# 1) Plot normalised sediment dynamics along with instream discharge and Qquick

# Add some suspended sed to Mland, as a simple linear function of instream discharge
Mland2 = df['Mland'] + 1000*df['Dr']
fQ = df['Dr']

# Normalise M_land and suspended sediment concs and plot on same figure
log_Mland = np.sqrt(df['Msus_mg-l'])
log_Mland2 = np.sqrt(Mland2)
log_fQ = np.sqrt(fQ)
df[df['Obs_SS']==0] = 0.5
log_SSobs = np.sqrt(df['Obs_SS'])

Mland_norm = (log_Mland - np.mean(log_Mland))/np.std(log_Mland)
Mland_norm.name = 'Msus_mg-l'
Mland2_norm = (log_Mland2 - np.mean(log_Mland2))/np.std(log_Mland2)
Mland2_norm.name = 'Mland_fQ'
fQ_norm = (log_fQ - np.mean(log_fQ))/np.std(log_fQ)
fQ_norm.name = ('Normalised_Q')
SSobs_norm = (log_SSobs - np.mean(log_SSobs))/np.std(log_SSobs)
norm_df = pd.concat([Mland_norm, Mland2_norm, fQ_norm, SSobs_norm],axis=1)

fig4 = norm_df[['Obs_SS', 'Mland_fQ']].plot(subplots=False,figsize=(15, 5))

# plt.savefig(r'M:\Working\NewModel\ModelOutputs\sed_tests5.png')
# fig5 = norm_df.boxplot(return_type ='dict', figsize=(1,3))