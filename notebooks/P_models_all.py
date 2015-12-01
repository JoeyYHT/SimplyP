# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:50:50 2015
@author: lj40184

List of phosphorus model functions tried as part of water quality
model development
"""

# Import modules
import numpy as np, pandas as pd
from scipy.integrate import odeint

#%%

# Unit conversions
def UC_Q(Q_mmd, A_catch):
    """Convert discharge from units of mm/day to m3/day"""
    Q_m3d = Q_mmd*1000*A_catch
    return Q_m3d

def UC_Qinv(Q_m3s, A_catch):
    """Convert discharge from units of m3/s to mm/day"""
    Q_mmd = Q_m3s * 86400/(1000*A_catch)
    return Q_mmd

def UC_C(C_kgmm, A_catch):
    """Convert concentration from units of kg/mm to mg/l
    Divide answer by 10**6 to convert from mg/mm to mg/l"""
    C_mgl = C_kgmm/A_catch
    return C_mgl

def UC_Cinv(C_mgl, A_catch):
    """Convert concentration from units of mg/l to kg/mm"""
    C_kgmm = C_mgl*A_catch
    return C_kgmm
    
def UC_V(V_mm, A_catch, outUnits):
    """Convert volume from mm to m^3 or to litres. outUnits 'm3' or 'l'"""
    factorDict = {'m3':10**3, 'l':10**6}
    V = V_mm * factorDict[outUnits] * A_catch
    return V
    
#%% (standard cell separator)

def P_model_1(met_df, params, period, dynamic_dict, run_mode, step_len=1):
    """
    First version of hydrology, sediment and phosphorus model. Non-distributed.    
    
    Inputs:
        met_df         Dataframe containing columns 'P', the precipitation+snowmelt input to
                       the soil box, and 'PET' (units mm)

        params         Series of parameter values (index = param name)

        period         Vector of [start, end] dates [yyyy-mm-dd, yyyy-mm-dd]

        dynamic_dict   Dictionary of options controlling whether inputs/variables are calculated
                       dynamically or kept constant.
                       Dictionary keys: 'Dynamic_EPC0', 'Dynamic_effluent_inputs',
                       'Dynamic_terrestrialP_inputs', 'Dynamic_GW_TDP'. Set to 'y' or 'n'.
                       **NB only the first is implemented in the model at the moment***

        run_mode       'cal' or 'val'. Determines whether the soil sorption coefficient, Kf,
                       is calculated (calibration period) or read in as a parameter (validation period)

        step_len       Length of each step in the input dataset (days). Default=1

    Returns a two-element tuple (df, Kf):
    1) A dataframe with column headings:
        Vs: Soil water volume (mm), Qs: Soil water flow (mm/d)
        Vg: Groundwater volume (mm), Qg: Groundwater flow (mm/d)
        Qq: Quick flow (mm/d)
        Vr: Reach volume (mm), Qr: Reach discharge (mm/d), Dr: Daily average reach discharge (mm/d)
        Mland: Sediment mass delivered ot the stream (kg/d), Msus: Reach suspended sediment mass
        P_labile: labile soil P mass (kg), EPC0_kgmm: Soil EPC0 (kg/mm), TDPs: Soil water TDP mass (kg)
        TDPr: Instream TDP mass (kg), PPr: Instream PP mass (kg)
    2) Kf, the soil adsorption coefficient (units mm/kg Soil)
    
    Potential improvements needed:
    * Add in saturation excess
    * Low flow simulation - replace sustainable flow parameter with percolation
      from soil to groundwater box when soil water level below field capacity
    * Sed equations: see notes in Word doc on to do list specifically relating
      to suspended sediment
    * Look into reducing the number of parameters required to simulate in-stream
      flow velocity (currently use a and b, but seems excessive)
    * PP: in-stream source? E.g. in summer. Sewage source? E.g. constant 10%?
    * Minor: can probably delete dVg/dt equation and remove Vg altogether
    
    Results: good
    """
    
    # #########################################################################################
    # Define the ODE system
    def ode_f(y, t, ode_params):
        """
        Define ODE system
        Inputs:
            y: list of variables expressed as dy/dx. The value of y is determined for the end of the time step
            t: array of time points of interest
            params: tuple of input values & model parameter values
        """
        
        # Unpack initial conditions for this time step 
        # Hydrology (Dr_i would be y[6], but it's 0 at the start of every time step)
        Vs_i = y[0] # Soil water volume (mm)
        Qs_i = y[1] # Soil water flow (mm/day)
        Vg_i = y[2] # Groundwater volume (mm)
        Qg_i = y[3] # Groundwater discharge (mm/day)
        Vr_i = y[4] # Reach volume (mm)
        Qr_i = y[5] # Reach discharge (mm/day)
        # Sediment
        Msus_i = y[7] # Mass of suspended sediment in the stream reach (kg)
        # Phosphorus
        Plab_i = y[8] # Mass of labile P in the soil (kg)
        TDPs_i = y[9] # Mass of total dissolved P in soil water (kg)
        TDPr_i = y[10] # Mass of total dissolved P in stream reach (kg)
        PPr_i = y[11]  # Mass of particulate P in stream reach (kg)
        
        # Unpack params
        (P, E, Qq_i, Mland_i, f_IExcess, alpha, beta, T_s,
        T_g, fc, L_reach, S_reach, A_catch, a_Q, b_Q, E_Q, k_EQ,
        P_netInput, EPC0, Kf, Msoil, TDPeff, TDPg, E_PP, PPeff, P_inactive) = ode_params
    
        # Soil equations (units mm or mm/day)
        dQs_dV = (((Vs_i - fc)*np.exp(fc - Vs_i))/(T_s*((np.exp(fc-Vs_i) + 1)**2)))
        +(1/(T_s*(np.exp(fc-Vs_i) + 1)))
        dVs_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*Vs_i)) - Qs_i
        dQs_dt = dQs_dV*dVs_dt
        
        # Groundwater equations (units mm or mm/day)
        dQg_dt = (beta*Qs_i - Qg_i)/T_g
        dVg_dt = beta*Qs_i - Qg_i
        
        # Instream equations (units m3 or m3/s)
        # Factors in dQr_dt: instream velocity (aQ^b) from m to mm; time constant from s to d; L_reach from m to mm
        dQr_dt = ((Qq_i + (1-beta)*Qs_i + Qg_i) - Qr_i)* a_Q*(Qr_i**b_Q)*(8.64*10**7)/((1-b_Q)*(L_reach*1000))
        dVr_dt = (Qq_i + (1-beta)*Qs_i + Qg_i) - Qr_i
        dDr_dt = Qr_i
        
        # Instream suspended sediment (kg; change in kg/day)
        dMsus_dt = (Mland_i # Delivery from the land (kg/day)
                   + E_Q*S_reach*(Qr_i**k_EQ)  # Entrainment from the stream bed (kg/d)
                   - (Msus_i/Vr_i)*Qr_i) # Outflow from the reach;(kg/mm)*(mm/day)
        
        # Soil labile phosphorus mass (kg) (two alternative formulations; give same results)
        dPlab_dt = Kf*Msoil*((TDPs_i/Vs_i)-EPC0)  # Net sorption
#         dPlab_dt = (Plab_i/EPC0)*((TDPs_i/Vs_i)-EPC0)
        
        # Change in dissolved P mass in soil water (kg/day)
        dTDPs_dt = ((P_netInput*100*A_catch/365)     # Net inputs (fert + manure - uptake) (kg/ha/yr)
                    -(Plab_i/EPC0)*((TDPs_i/Vs_i)-EPC0)#(Kf*Msoil*((TDPs_i/Vs_i)-EPC0)) # Net sorption (kg/day)
                   - (Qs_i*TDPs_i/Vs_i)              # Outflow via soil water flow (kg/day)
                   - (Qq_i*TDPs_i/Vs_i))             # Outflow via quick flow (kg/day)
        
        # Change in in-stream TDP mass (kg/d)
        dTDPr_dt = ((1-beta)*Qs_i*(TDPs_i/Vs_i)  # Soil water input. Units: (mm/d)(kg/mm)
                   + Qq_i*(TDPs_i/Vs_i)          # Quick flow input. Units: (mm/d)(kg/mm)
                   + Qg_i*UC_Cinv(TDPg,A_catch)  # Groundwater input. Units: (mm/d)(kg/mm)
                   + TDPeff                      # Effluent input (kg/day)
                   - Qr_i*(TDPr_i/Vr_i))         # Outflow from the reach. Units: (mm/d)(kg/mm)
        
        # Change in in-stream PP mass (kg/d)
        dPPr_dt = (E_PP*Mland_i*(Plab_i+P_inactive)/Msoil  # Delivery from the land
                   + (E_Q*S_reach*(Qr_i**k_EQ))*E_PP*(Plab_i+P_inactive)/Msoil  # M_ent*P content of sed
                   + PPeff                      # Effluent input (kg/day)
                   - Qr_i*(PPr_i/Vr_i))  # Outflow from reach. Units: (mm/d)(kg/mm)
        
        # Add results of equations to an array
        res = np.array([dVs_dt, dQs_dt, dVg_dt, dQg_dt, dVr_dt, dQr_dt, dDr_dt, dMsus_dt,
                       dPlab_dt, dTDPs_dt, dTDPr_dt, dPPr_dt])
        
        return res
    
    # ###########################################################################################
    # --------------------------------------------------------------------------------------------

    # INITIAL CONDITIONS AND DERIVED PARAMETERS
    # Unpack user-supplied initial conditions and calculate other initial conditions which are not modified
    # during looping over met data
    
    # 1) Hydrology
    Vs0 = p['fc']   # Initial soil volume (mm). Assume it's at field capacity.
    Qg0 = p['Qg0_init']      # Initial groundwater flow (mm/d)
    Qr0 = UC_Qinv(p['Qr0_init'], p['A_catch'])  # Convert units of initial reach Q from m3/s to mm/day
    
    # 2) Initial instream sediment mass (kg). Assume initial suspended sediment mass is 0 kg
    Msus0 = 0.0
    
    # 3) Terrestrial P
    # Set up for calculating initial conditions (and calculate useful things)
    p['Msoil'] = p['Msoil_m2'] * 10**6 * p['A_catch'] # Soil mass (kg). Units: kgSoil/m2 * m2/km2 * km2
    EPC0_0 = UC_Cinv(p['EPC0_init_mgl'], p['A_catch'])  # Convert units of EPC0 from mg/l to kg/mm
    # Initial conditions 
    # Soil labile P mass (kg). NB just for arable at the mo!!!! For SN: Plab0 = 0
    Plab0 = 10**-6*(p['SoilPconc_A']-p['SoilPconc_SN']) * p['Msoil'] # Units: kgP/mgP * mgP/kgSoil * kgSoil
    # Inactive soil P mass (kg). Assume = semi-natural total soil P mass for all LU classes
    p['P_inactive'] = 10**-6*p['SoilPconc_SN']*p['Msoil']
    TDPs0 = EPC0_0 * Vs0 # Initial soil water TDP mass (kg); Units: (kg/mm)*mm
    # Set the value for Kf, the adsorption coefficient (mm/kg soil)
    if run_mode == 'cal':
        # If the calibration period, calculate. Assume SN has EPC0=0, PlabConc =0
        Kf = 10**-6 * (p['SoilPconc_A']-p['SoilPconc_SN'])/EPC0_0  # (kgP/mgP * mgP/kgSoil * mm/kgP)
    else:
        # If not the calibration period, read Kf in from the series of param values
        Kf = p['Kf']

    # 4) Initial instream P mass (assume 0.0)
    TDPr0 = 0.0
    PPr0 = 0.0
    
    # --------------------------------------------------------------------------------------------    
    # SETUP ADMIN FOR LOOPING OVER MET DATA
    # Time points to evaluate ODEs at (we're only interested in the start and end of each step)
    ti = [0, step_len]
    # Lists to store output
    output_ODEs = []  # From ode_f function
    output_nonODE = []  # Will include: Qq, Mland

    # --------------------------------------------------------------------------------------------
    # START LOOP OVER MET DATA
    for idx in range(len(met_df)):

        # Get precipitation and evapotranspiration for this day
        P = met_df.ix[idx, 'P']
        E = met_df.ix[idx, 'PET']

        # Calculate infiltration excess (mm/(day * catchment area))
        Qq_i = p['f_IExcess']*P
        
        # Calculate terrestrial erosion and delivery to the stream, Mland (kg/day)
        Mland_i = p['E_land']*(Qq_i**p['k_Eland']) #Units: (kg/mm)*(mm/day)
        
        # Calculate dynamic EPC0 as a function of labile P mass
        if dynamic_dict['Dynamic_EPC0'] == 'y':
            EPC0_i = Plab0/(Kf*p['Msoil']) # First timestep this equals EPC0_0
        else:
            EPC0_i = EPC0_0
        
        # Append to non ODE results
        output_nonODE_i = [Qq_i, Mland_i, EPC0_i]
        output_nonODE.append(output_nonODE_i)

        # Calculate additional initial conditions from user-input initial conditions/ODE solver output
        Qs0 = (Vs0 - p['fc'])/(p['T_s']*(1 + np.exp(p['fc'] - Vs0))) # Soil flow (mm/d)
        Vg0 = Qg0 *p['T_g'] # groundwater vol (mm)
        Tr0 = (1000*p['L_reach'])/(p['a_Q']*(Qr0**p['b_Q'])*(8.64*10**7)) #Reach time constant (days); T=L/aQ^b
        Vr0 = Qr0*Tr0 # Reach volume (V=QT) (mm)

        # Vector of initial conditions for start of time step (assume 0 for Dr0, daily mean instream Q)
        y0 = [Vs0, Qs0, Vg0, Qg0, Vr0, Qr0, 0.0, Msus0, Plab0, TDPs0, TDPr0, PPr0]

        # Today's rainfall, ET & model parameters for input to solver
        ode_params = np.array([P, E, Qq_i, Mland_i,
                              p['f_IExcess'],p['alpha'], p['beta'], p['T_s'], p['T_g'], p['fc'],
                              p['L_reach'], p['S_reach'], p['A_catch'],
                              p['a_Q'], p['b_Q'],
                              p['E_Q'], p['k_EQ'],
                              p['P_netInput'], EPC0_i, Kf, p['Msoil'],
                              p['TDPeff'], p['TDPg'],
                              p['E_PP'], p['PPeff'], p['P_inactive']])

        # Solve ODEs
        y = odeint(ode_f, y0, ti, args=(ode_params,))

        # Extract values for the end of the step
        res = y[1]
        res[res<0] = 0 # Numerical errors may result in very tiny values <0; set these back to 0
        output_ODEs.append(res)

        # Update initial conditions for next step (for Vs0, Qg0, Qr0, Msus0, Plab0, TDPs0)
        Vs0 = res[0]
        # FUDGE to re-set groundwater to user-supplied min flow at start of each time step!!!
        if p['Qg_min'] > res[3]:
            Qg0 = p['Qg_min']
        else:
            Qg0 = res[3]
        Qr0 = res[5]
        Msus0 = res[7]
        Plab0 = res[8]
        TDPs0 = res[9]
        TDPr0 = res[10]
        PPr0 = res[11]
        
        # END LOOP OVER MET DATA
    # --------------------------------------------------------------------------------------------
    
    # Build a dataframe of ODE results
    df1 = pd.DataFrame(data=np.vstack(output_ODEs),
                      columns=['Vs', 'Qs', 'Vg', 'Qg', 'Vr', 'Qr', 'Dr', 'Msus', 'P_labile',
                              'TDPs', 'TDPr', 'PPr'],
                      index=met_df.index)
    
    # Dataframe of non ODE results
    df2 = pd.DataFrame(data=np.vstack(output_nonODE), columns=['Qq', 'Mland', 'EPC0_kgmm'],
                      index=met_df.index)

    # Concatenate results dataframes
    df = pd.concat([df1,df2], axis=1)

    return (df, Kf)

# Associated parameter dict:
# Model parameters, including starting guesses for those being calibrated
# Units: L_reach in mm for now. a_Q and b_Q are for m/s vs m3/s, therefore convert in script for now.
p = {'A_catch': 51.7, 'L_reach':10000. , 'S_reach':0.8, 'fc':290.,
              'alpha':0.95, 'beta':0.6,'f_IExcess':0.015, 'T_s':3.,'T_g':60., 'Qg_min':0.4,
              'a_Q':0.5, 'b_Q':0.5,'Qg0_init':1.0, 'Qr0_init': 1.0,
              'E_land':5000., 'k_Eland':1.2, 'E_Q': 300., 'k_EQ':1.7,
              'P_netInput': 5.0, 'EPC0_init_mgl':0.07, 'Kf': 5.5263885e-05,
              'TDPeff':TDPeff, 'TDPg':0.01,
              'PPeff': 0.1*TDPeff/0.9, 'E_PP':2.0,
              'SoilPconc_SN':900., 'SoilPconc_A':1100., 'Msoil_m2':200.}

#%%