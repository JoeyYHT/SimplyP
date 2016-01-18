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
def P_model_LU_v1(met_df, p, p_LU, dynamic_dict, run_mode, step_len=1):
    """
    Second version of hydrology, sediment and P model.
    * Includes simple hard-coded variation of processes by land use (arable,
    * improved grassland and semi-natural). Does not include the ability to
      simulate a new land use class with a mixture of properties of the others.
    * Does not include sub-catchments or reaches.
    
    Inputs:
        met_df         Dataframe containing columns 'P', the precipitation+snowmelt input to
                       the soil box, and 'PET' (units mm)

        p              Series of parameter values which don't vary by land use (index: param name)
        
        p_LU           Dataframe of parameter values which vary by land use
                       Index: parameter name
                       Columns: 'A', 'S', 'IG' (agricultural, semi-natural, improved grassland)
                       IG only has values for parameter 'E_land' (soil erodibility), in which case
                       value in 'A' column is for arable land

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
    2) Kf, the soil adsorption coefficient (units mm/kg Soil). #Units (mgP/kgSoil)(mm/kgP).
       Multiply by: 10^-6 for (mm/kgSoil), by A_catch*10^6 for (l/kgSoil)
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
        # Hydrology
        VsA_i = y[0] # Agricultural soil water volume (mm)
        QsA_i = y[1] # Agricultural soil water flow (mm/day)
        VsS_i = y[2] # Semi-natural soil water volume (mm)
        QsS_i = y[3] # Semi-natural soil water flow (mm/day)
        Vg_i = y[4]  # Groundwater volume (mm)
        Qg_i = y[5]  # Groundwater discharge (mm/day)
        Vr_i = y[6]  # Reach volume (mm)
        Qr_i = y[7]  # Reach discharge (mm/day)
        #(Dr_i would be y[i] here, but it's 0 at the start of every time step)
        # Sediment
        Msus_i = y[9]  # Mass of suspended sediment in the stream reach (kg)
        # Phosphorus
        PlabA_i = y[10] # Mass of labile P in agricultural soil (kg)
        TDPsA_i = y[11] # Mass of TDP in agricultural soil water (kg)
        TDPr_i = y[12] # Mass of total dissolved P in stream reach (kg)
        PPr_i = y[13]  # Mass of particulate P in stream reach (kg)
        
        # Unpack params. Params that vary by LU are series, with indices ['A','S','IG'],
        # LU-varying params: T_s, P_netInput, EPC0, Mland_i
        (P, E, Qq_i, Mland_i, f_A, f_Ar, f_S, f_IG, f_IExcess, alpha, beta, T_s,
        T_g, fc, L_reach, S_reach, A_catch, a_Q, b_Q, E_Q, k_EQ,
        P_netInput_A, EPC0_A, Kf, Msoil, TDPeff, TDPg, E_PP, PPeff, P_inactive) = ode_params
    
        # HYDROLOGY
        
        # Soil hydrology equations (units mm or mm/day): Agricultural land
        dQsA_dV = ((((VsA_i - fc)*np.exp(fc - VsA_i))/(T_s['A']*((np.exp(fc-VsA_i) + 1)**2)))
                    +(1/(T_s['A']*(np.exp(fc-VsA_i) + 1))))
        dVsA_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*VsA_i)) - QsA_i
        dQsA_dt = dQsA_dV*dVsA_dt
        
        # Soil hydrology equations (units mm or mm/day): Semi-natural/other land
        dQsS_dV = ((((VsS_i - fc)*np.exp(fc - VsS_i))/(T_s['S']*((np.exp(fc-VsS_i) + 1)**2)))
                    +(1/(T_s['S']*(np.exp(fc-VsS_i) + 1))))
        dVsS_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*VsS_i)) - QsS_i
        dQsS_dt = dQsS_dV*dVsS_dt
        
        # Groundwater equations (units mm or mm/day)
        dQg_dt = (beta*(f_A*QsA_i + f_S*QsS_i) - Qg_i)/T_g
        dVg_dt = beta*(f_A*QsA_i + f_S*QsS_i) - Qg_i
        
        # Instream equations (units mm or mm/day)
        dQr_dt = (((Qq_i + (1-beta)*(f_A*QsA_i + f_S*QsS_i) + Qg_i) - Qr_i)
                  *a_Q*(Qr_i**b_Q)*(8.64*10**7)/((1-b_Q)*(L_reach*1000))) #U/L=1/T. Units (m/s)(s/d)(mm/m)/m(mm/m) 
        dVr_dt = Qq_i + (1-beta)*(f_A*QsA_i + f_S*QsS_i) + Qg_i - Qr_i
        dDr_dt = Qr_i
        
        # SEDIMENT
        # Instream suspended sediment (kg; change in kg/day)
        dMsus_dt = (f_Ar*Mland_i['A'] + f_IG*Mland_i['IG'] + f_S* Mland_i['S']     # Delivery from the land (kg/day)
                   + E_Q*S_reach*(Qr_i**k_EQ)                        # Entrainment from the stream bed (kg/d)
                   - (Msus_i/Vr_i)*Qr_i)                             # Outflow from the reach;(kg/mm)*(mm/day)
        
        # PHOSPHORUS
        
        # Agricultural soil labile phosphorus mass (kg) (two alternative formulations; give same results)
        # Assume semi-natural land has no labile soil P
        dPlabA_dt = Kf*Msoil*((TDPsA_i/VsA_i)-EPC0_A)  # Net sorption
        # dPlabA_dt = (PlabA_i/EPC0_A)*((TDPsA_i/VsA_i)-EPC0_A)
        
        # Change in dissolved P mass in agricultural soil water (kg/day)
        # Assume semi-natural land has no dissolved soil water P
        dTDPsA_dt = ((P_netInput_A*100*A_catch/365)         # Net inputs (fert + manure - uptake) (kg/ha/yr)
                   - Kf*Msoil*((TDPsA_i/VsA_i)-EPC0_A)      # Net sorpn (kg/day)
                                                            # Alternative: (PlabA_i/EPC0_A)*((TDPsA_i/VsA_i)-EPC0_A) 
                   - (QsA_i*TDPsA_i/VsA_i)                  # Outflow via soil water flow (kg/day)
                   - (Qq_i*TDPsA_i/VsA_i))                  # Outflow via quick flow (kg/day)
        
        # Change in in-stream TDP mass (kg/d)   
        dTDPr_dt = ((1-beta)*f_A*QsA_i*(TDPsA_i/VsA_i)  # Soil water input from agri land. Units: (mm/d)(kg/mm)
                   + f_A*Qq_i*(TDPsA_i/VsA_i)           # Quick flow input from agri land. Units: (mm/d)(kg/mm)
                   + Qg_i*UC_Cinv(TDPg,A_catch)         # Groundwater input. Units: (mm/d)(kg/mm)
                   + TDPeff                             # Effluent input (kg/day)
                   - Qr_i*(TDPr_i/Vr_i))                # Outflow from the reach. Units: (mm/d)(kg/mm)
        
        # Change in in-stream PP mass (kg/d)
        dPPr_dt = (E_PP * (f_Ar*Mland_i['A']*(PlabA_i+P_inactive)/Msoil  # From arable land
                   + f_IG*Mland_i['IG']*(PlabA_i+P_inactive)/Msoil        # From improved grassland
                   + f_S*Mland_i['S']*P_inactive/Msoil)                  # From semi-natural land
                   + (E_Q*S_reach*(Qr_i**k_EQ))*E_PP*(f_A*(PlabA_i+P_inactive)+f_S*P_inactive)/Msoil  # Entrained                + PPeff                      # Effluent input (kg/day)
                   - Qr_i*(PPr_i/Vr_i))                                   # Outflow from reach. Units: (mm/d)(kg/mm)
        
        # Add results of equations to an array
        res = np.array([dVsA_dt, dQsA_dt, dVsS_dt, dQsS_dt,dVg_dt, dQg_dt, dVr_dt, dQr_dt, dDr_dt, dMsus_dt,
                       dPlabA_dt, dTDPsA_dt, dTDPr_dt, dPPr_dt])
        
        return res
    
    # ###########################################################################################
    # --------------------------------------------------------------------------------------------

    # INITIAL CONDITIONS AND DERIVED PARAMETERS
    # Unpack user-supplied initial conditions and calculate other initial conditions which are not modified
    # during looping over met data
    
    # General
    p['f_A'] = p['f_IG']+p['f_Ar'] # Area of intensive agricultural land (as a fraction of total area)
    
    # 1) Hydrology
    VsA0 = p['fc']   # Initial soil volume (mm). Assume it's at field capacity.
    VsS0 = VsA0      # Initial soil vol, semi-natural land (mm). Assume same as agricultural for now!!
    Qg0 = p['Qg0_init']      # Initial groundwater flow (mm/d)
    Qr0 = UC_Qinv(p['Qr0_init'], p['A_catch'])  # Convert units of initial reach Q from m3/s to mm/day
    
    # 2) Initial instream suspended sediment mass (kg). Assume it's 0 kg
    Msus0 = 0.0
    
    # 3) Terrestrial P
    # Calculate initial conditions and convert input param units where necessary
    p['Msoil'] = p['Msoil_m2'] * 10**6 * p['A_catch'] # Soil mass (kg). Units: kgSoil/m2 * m2/km2 * km2
    # Inactive soil P mass (kg). Assume equals semi-natural total soil P mass for all LU classes
    p['P_inactive'] = 10**-6*p_LU['S']['SoilPconc']*p['Msoil']
    
    # Varying by LU (intensive agricultural or semi-natural)
    for LU in ['A','S']:
        # Convert units of EPC0 from mg/l to kg/mm to give initial EPC0
        p_LU.ix['EPC0_0',LU] = UC_Cinv(p_LU[LU]['EPC0_init_mgl'], p['A_catch'])
        # Initial labile P. Units: kgP/mgP * mgP/kgSoil * kgSoil. Assume Plab0 = 0 for semi-natural
        p_LU.ix['Plab0',LU] = 10**-6*(p_LU[LU]['SoilPconc']-p_LU['S']['SoilPconc']) * p['Msoil']  
        # Initial soil water TDP mass (kg); Units: (kg/mm)*mm
        p_LU.ix['TDPs0',LU] = p_LU[LU]['EPC0_0'] * (p['f_A']*VsA0+p['f_S']*VsS0)
    
    # Set initial agricultural labile P and soil TDP masses as variables to be updated during looping
    # (assume semi-natural remain at 0 for both)
    Plab0_A, TDPs0_A = p_LU.ix['Plab0','A'], p_LU.ix['TDPs0','A']
    
    # Set the value for Kf, the adsorption coefficient (mm/kg soil)
    if run_mode == 'cal':
        # If the calibration period, calculate. Assume SN has EPC0=0, PlabConc =0
        Kf = 10**-6*(p_LU['A']['SoilPconc']-p_LU['S']['SoilPconc'])/p_LU['A']['EPC0_0']  # (kg/mg)(mg/kgSoil)(mm/kg)
    else:
        # If not the calibration period, read Kf in from the series of param values
        Kf = p['Kf']

    # 4) Initial instream TDP and PP masses (kg; assume both equal 0.0)
    TDPr0 = 0.0
    PPr0 = 0.0
    
    # --------------------------------------------------------------------------------------------    
    # SETUP ADMIN FOR LOOPING OVER TIME STEPS
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
        # NB this is the flux assuming the land use covered the whole catchment. Divide by catchment
        # area to get areal flux to e.g. compare to literature values
        Mland_i = pd.Series(3*[np.NaN],['A','S','IG'])
        for LU in ['A','S','IG']:
            Mland_i[LU] = p_LU[LU]['E_land']*(Qq_i**p['k_Eland']) #Units: (kg/mm)*(mm/day)
        
        # Calculate dynamic EPC0 as a function of labile P mass
        if dynamic_dict['Dynamic_EPC0'] == 'y':
            EPC0_A_i = p_LU['A']['Plab0']/(Kf*p['Msoil']) # First timestep this equals EPC0_0
        else:
            EPC0_A_i = p_LU['A']['EPC0_0']
        
        # Append to non ODE results
        output_nonODE_i = [Qq_i, Mland_i['A'],Mland_i['IG'],Mland_i['S'], EPC0_A_i]
        output_nonODE.append(output_nonODE_i)

        # Calculate additional initial conditions from user-input initial conditions/ODE solver output
        QsA0 = (VsA0 - p['fc'])/(p_LU['A']['T_s']*(1 + np.exp(p['fc'] - VsA0))) # Soil flow (mm/d)
        QsS0 = (VsS0 - p['fc'])/(p_LU['S']['T_s']*(1 + np.exp(p['fc'] - VsS0)))
        Vg0 = Qg0 *p['T_g'] # groundwater vol (mm)
        Tr0 = (1000*p['L_reach'])/(p['a_Q']*(Qr0**p['b_Q'])*(8.64*10**7)) #Reach time constant (days); T=L/aQ^b
        Vr0 = Qr0*Tr0 # Reach volume (V=QT) (mm)

        # Vector of initial conditions for start of time step (assume 0 for Dr0, daily mean instream Q)
        y0 = [VsA0, QsA0, VsS0, QsS0, Vg0, Qg0, Vr0, Qr0, 0.0, Msus0, Plab0_A,
              TDPs0_A, TDPr0, PPr0]

        # Today's rainfall, ET & model parameters for input to solver
        ode_params = [P, E, Qq_i, Mland_i, p['f_A'], p['f_Ar'], p['f_S'], p['f_IG'],
                        p['f_IExcess'],p['alpha'], p['beta'],
                        p_LU.ix['T_s'], p['T_g'], p['fc'],
                        p['L_reach'], p['S_reach'], p['A_catch'],
                        p['a_Q'], p['b_Q'],
                        p['E_Q'], p['k_EQ'],
                        p_LU['A']['P_netInput'], EPC0_A_i, Kf, p['Msoil'],
                        p['TDPeff'], p['TDPg'],
                        p['E_PP'], p['PPeff'], p['P_inactive']]

        # Solve ODEs
        y = odeint(ode_f, y0, ti, args=(ode_params,))

        # Extract values for the end of the step
        res = y[1]
        res[res<0] = 0 # Numerical errors may result in very tiny values <0; set these back to 0
        output_ODEs.append(res)

        # Update initial conditions for next step (for Vs, Qg0, Qr0, Msus0, Plab0, TDPs0)
        VsA0 = res[0]
        VsS0 = res[2]
        # FUDGE to re-set groundwater to user-supplied min flow at start of each time step!!!
        if p['Qg_min'] > res[5]:
            Qg0 = p['Qg_min']
        else:
            Qg0 = res[5]
        Qr0 = res[7]
        Msus0 = res[9]
        Plab0_A = res[10]
        TDPs0_A = res[11]
        TDPr0 = res[12]
        PPr0 = res[13]
        
        # END LOOP OVER MET DATA
    # --------------------------------------------------------------------------------------------
    
    # Build a dataframe of ODE results
    df1 = pd.DataFrame(data=np.vstack(output_ODEs),
                      columns=['VsA', 'QsA','VsS', 'QsS','Vg', 'Qg', 'Vr', 'Qr', 'Dr',
                               'Msus', 'P_labile','TDPs_A', 'TDPr', 'PPr'],
                      index=met_df.index)
    
    # Dataframe of non ODE results
    df2 = pd.DataFrame(data=np.vstack(output_nonODE), columns=['Qq','Mland_A','Mland_IG','Mland_S','EPC0_A_kgmm'],
                      index=met_df.index)

    # Concatenate results dataframes
    df = pd.concat([df1,df2], axis=1)

    return (df, Kf)
    


#%%

# READ IN INPUT DATA
# Met data and observations
met_df = pd.read_csv(metdata_fpath, parse_dates=True, dayfirst=True, index_col=0)
Qobs_df = pd.read_csv(Qobsdata_fpath, parse_dates=True, dayfirst=True, index_col=0)
chemObs_df = pd.read_csv(chemObsData_fpath, parse_dates=True, dayfirst=True, index_col=0)

# Read in model parameters from file
# Land use parameters
p_LU = pd.read_excel(params_fpath, sheetname='LU',index_col=0,parse_cols="B,E,F,G,H") # Values in cols A,S,IG,NC
# Sub-catchment and reach parameters
p_SC_R = pd.read_excel(params_fpath, sheetname='SC_reach',index_col=0, parse_cols="B,E") # Values in cols '1', '2',..
# Parameters that're constant over land use, sub-catchment or reach
p_constant_df = pd.read_excel(params_fpath,sheetname='Constant',index_col=0,parse_cols="B,E") # Values in col 'Value'

# PROCESS INPUT DATA & SET UP

# MET AND OBSERVED DATA
# Truncate the met and observed data to the desired period
met_df = met_df.truncate(before=st_dt, after=end_dt)
Qobs_df = Qobs_df.truncate(before=st_dt, after=end_dt)
chemObs_df = chemObs_df.truncate(before=st_dt, after=end_dt)
# Combine Q and chem data into one df
obs_df = pd.concat([Qobs_df, chemObs_df], axis=1)

# Pre-processing of parameters:
# PARAMETERS: read into model as a series or a dataframe (if vary by land use or sub-catchment/reach)

# If reading parameter values from dictionary in iPython notebook,  convert from dictionary to series
# p = pd.Series(param_dict)  # All params; cols = param names, one row with values

# Otherwise, if reading in from a file, do some processing of parameters that are general and
# constant in any e.g. auto-calibration/ncertainty analysis

p_constant = p_constant_df['Value'] #convert to a series
p = pd.concat([p_SC_R[1],p_constant],axis=0) #TEMPORARY, UNTIL VARY EQNS BY SUB-CATCHMENT/REACH

# Add empty rows to land use param dataframe to be populated later in model (4 LU classes: 'A','S','NC','IG')
p_LU.ix['EPC0_0',:] = 4*[np.NaN]  # Initial EPC0 (kg/mm)
p_LU.ix['Plab0',:] = 4*[np.NaN]   # Initial labile P (kg)
p_LU.ix['TDPs0',:] = 4*[np.NaN]   # Initial soil water TDP mass (kg)
p_LU.ix['f_TDPr',:] = 4*[np.NaN]  # fraction of land use classes contributing to reach TDP mass

# Sort out land use fractions
p['f_A'] = p['f_IG']+p['f_Ar'] # Area of intensive agricultural land (as a fraction of total area)
# Series giving fraction of land covered by arable and semi-natural land, after subtracting any newly-converted
# land. This fraction is only used for the reach TDP calculation, hence the variable name
if p['NC_type'] == 'A': # f_NC is fraction of agricultural land that's newly converted from SN
    p_LU.ix['f_TDPr','A'] = p['f_A'] - p['f_NC']*p['f_A']
    p_LU.ix['f_TDPr','S'] = p['f_S']
    p_LU.ix['f_TDPr','NC'] = p['f_NC']*p['f_A']
else:
    p_LU.ix['f_TDPr','A'] = p['f_A'] # f_NC is fraction of semi-natural land converted from agricultural
    p_LU.ix['f_TDPr','S'] = p['f_S'] - p['f_NC']*p['f_S']
    p_LU.ix['f_TDPr','NC'] = p['f_NC']*p['f_S']


def P_model_LU_v2(met_df, p, p_LU, dynamic_dict, run_mode, step_len=1):
    """
    Intermediate version of hydrology, sediment and phosphorus model, with processes varying by land use,
    including the ability to simulate a newly converted land use class (semi-natural to agricultural,
    or agricultural to semi-natural).
    
    In this version, the newly-converted land use class only affects TDP, not PP
    
    Inputs:
        met_df         Dataframe containing columns 'P', the precipitation+snowmelt input to
                       the soil box, and 'PET' (units mm)

        p              Series of parameter values which don't vary by land use (index: param name)
        
        p_LU           Dataframe of parameter values which vary by land use
                       Index: parameter name
                       Columns: 'A', 'S', 'IG' (agricultural, semi-natural, improved grassland)
                       IG only has values for parameter 'E_land' (soil erodibility), in which case
                       value in 'A'  is for arable land
column
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
    2) Kf, the soil adsorption coefficient (units mm/kg Soil). #Units (mgP/kgSoil)(mm/kgP).
       Multiply by: 10^-6 for (mm/kgSoil), by A_catch*10^6 for (l/kgSoil)
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
        
        # Unpack params. Params that vary by LU are series, with indices ['A','S','IG'],
        # LU-varying params: f_TDPr (fraction of land use for TDP input to reach calc), T_s,P_netInput,EPC0,Mland_i
        (P, E, Qq_i, Mland_i, f_A, f_Ar, f_S, f_IG, f_TDPr, NC_type, f_IExcess, alpha, beta, T_s,
        T_g, fc, L_reach, S_reach, A_catch, a_Q, b_Q, E_Q, k_EQ,
        P_netInput, EPC0_A, EPC0_NC, Kf, Msoil, TDPeff, TDPg, E_PP, PPeff, P_inactive) = ode_params
        
        # Unpack initial conditions for this time step 
        # Hydrology
        VsA_i = y[0] # Agricultural soil water volume (mm)
        QsA_i = y[1] # Agricultural soil water flow (mm/day)
        VsS_i = y[2] # Semi-natural soil water volume (mm)
        QsS_i = y[3] # Semi-natural soil water flow (mm/day)
        Vg_i = y[4]  # Groundwater volume (mm)
        Qg_i = y[5]  # Groundwater discharge (mm/day)
        Vr_i = y[6]  # Reach volume (mm)
        Qr_i = y[7]  # Reach discharge (mm/day)
        #(Dr_i would be y[i] here, but it's 0 at the start of every time step)
        # Sediment
        Msus_i = y[9]  # Mass of suspended sediment in the stream reach (kg)
        # Phosphorus
        PlabA_i = y[10] # Mass of labile P in agricultural soil (kg)
        PlabNC_i = y[11] # Mass of labile P in newly converted land class (kg)
        TDPsA_i = y[12] # Mass of TDP in agricultural soil water (kg)
        TDPsNC_i = y[13] # Mass of TDP in newly converted land class (kg)
        TDPr_i = y[14] # Mass of total dissolved P in stream reach (kg)
        PPr_i = y[15]  # Mass of particulate P in stream reach (kg)
        # Soil water vol & flow for newly converted land class
        if NC_type == 'A':  # If semi-natural converted to arable, assume has arable hydrol
            VsNC_i = VsA_i
            QsNC_i = QsA_i
        else:
            VsNC_i = VsS_i  # If arable converted to SN, assume has semi-natural hydrol
            QsNC_i = QsS_i
    
        # HYDROLOGY
        
        # Soil hydrology equations (units mm or mm/day): Agricultural land
        dQsA_dV = ((((VsA_i - fc)*np.exp(fc - VsA_i))/(T_s['A']*((np.exp(fc-VsA_i) + 1)**2)))
                    +(1/(T_s['A']*(np.exp(fc-VsA_i) + 1))))
        dVsA_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*VsA_i)) - QsA_i
        dQsA_dt = dQsA_dV*dVsA_dt
        
        # Soil hydrology equations (units mm or mm/day): Semi-natural/other land
        dQsS_dV = ((((VsS_i - fc)*np.exp(fc - VsS_i))/(T_s['S']*((np.exp(fc-VsS_i) + 1)**2)))
                    +(1/(T_s['S']*(np.exp(fc-VsS_i) + 1))))
        dVsS_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*VsS_i)) - QsS_i
        dQsS_dt = dQsS_dV*dVsS_dt
        
        # Groundwater equations (units mm or mm/day)
        dQg_dt = (beta*(f_A*QsA_i + f_S*QsS_i) - Qg_i)/T_g
        dVg_dt = beta*(f_A*QsA_i + f_S*QsS_i) - Qg_i
        
        # Instream equations (units mm or mm/day)
        dQr_dt = (((Qq_i + (1-beta)*(f_A*QsA_i + f_S*QsS_i) + Qg_i) - Qr_i)
                  *a_Q*(Qr_i**b_Q)*(8.64*10**7)/((1-b_Q)*(L_reach*1000))) #U/L=1/T. Units (m/s)(s/d)(mm/m)/m(mm/m) 
        dVr_dt = Qq_i + (1-beta)*(f_A*QsA_i + f_S*QsS_i) + Qg_i - Qr_i
        dDr_dt = Qr_i
        
        # SEDIMENT
        # Instream suspended sediment (kg; change in kg/day)
        dMsus_dt = (f_Ar*Mland_i['A'] + f_IG*Mland_i['IG'] + f_S* Mland_i['S']     # Delivery from the land (kg/day)
                   + E_Q*S_reach*(Qr_i**k_EQ)                        # Entrainment from the stream bed (kg/d)
                   - (Msus_i/Vr_i)*Qr_i)                             # Outflow from the reach;(kg/mm)*(mm/day)
        
        # PHOSPHORUS
        
        # Agricultural soil labile phosphorus mass (kg) (two alternative formulations; give same results)
        # Assume semi-natural land has no labile soil P
        dPlabA_dt = Kf*Msoil*((TDPsA_i/VsA_i)-EPC0_A)  # Net sorption
        # dPlabA_dt = (PlabA_i/EPC0_A)*((TDPsA_i/VsA_i)-EPC0_A)
        
        # Newly-conveted soil labile P mass (kg)
        dPlabNC_dt = Kf*Msoil*((TDPsNC_i/VsNC_i)-EPC0_NC)
        
        # Change in dissolved P mass in agricultural soil water (kg/day)
        # Assume semi-natural land has no dissolved soil water P
        dTDPsA_dt = ((P_netInput['A']*100*A_catch/365)      # Net inputs (fert + manure - uptake) (kg/ha/yr)
                   - Kf*Msoil*((TDPsA_i/VsA_i)-EPC0_A)      # Net sorpn (kg/day)
                                                            # Alternative: (PlabA_i/EPC0_A)*((TDPsA_i/VsA_i)-EPC0_A) 
                   - (QsA_i*TDPsA_i/VsA_i)                  # Outflow via soil water flow (kg/day)
                   - (Qq_i*TDPsA_i/VsA_i))                  # Outflow via quick flow (kg/day)
        
        # And in newly converted land class soil water
        dTDPsNC_dt = ((P_netInput['NC']*100*A_catch/365)    # Net inputs (fert + manure - uptake) (kg/ha/yr)
                   - Kf*Msoil*((TDPsNC_i/VsNC_i)-EPC0_NC)   # Net sorpn (kg/day)
                   - (QsNC_i*TDPsNC_i/VsNC_i)               # Outflow via soil water flow (kg/day)
                   - (Qq_i*TDPsNC_i/VsNC_i))                # Outflow via quick flow (kg/day)
        
        # Change in in-stream TDP mass (kg/d).
        # Semi-natural inputs not specified as assume 0 for soil water & quick flow
        dTDPr_dt = ((1-beta)*f_TDPr['A']*QsA_i*(TDPsA_i/VsA_i)  # Soil water input, old agri land. Units: (mm/d)(kg/mm)
                   + (1-beta)*f_TDPr['NC']*QsNC_i*(TDPsNC_i/VsNC_i)  # Soil water input, newly converted land
                   + f_TDPr['A']*Qq_i*(TDPsA_i/VsA_i)       # Quick flow input from old agri land. Units: (mm/d)(kg/mm)
                   + f_TDPr['NC']*Qq_i*(TDPsNC_i/VsNC_i)    # Quick flow inputs from newly converted land
                   + Qg_i*UC_Cinv(TDPg,A_catch)             # Groundwater input. Units: (mm/d)(kg/mm)
                   + TDPeff                                 # Effluent input (kg/day)
                   - Qr_i*(TDPr_i/Vr_i))                    # Outflow from the reach. Units: (mm/d)(kg/mm)
        
        # Change in in-stream PP mass (kg/d)
        dPPr_dt = (E_PP * (f_Ar*Mland_i['A']*(PlabA_i+P_inactive)/Msoil  # From arable land
                   + f_IG*Mland_i['IG']*(PlabA_i+P_inactive)/Msoil       # From improved grassland
                   + f_S*Mland_i['S']*P_inactive/Msoil)                  # From semi-natural land
                   + (E_Q*S_reach*(Qr_i**k_EQ))*E_PP*(f_A*(PlabA_i+P_inactive)+f_S*P_inactive)/Msoil  # Entrained                + PPeff                      # Effluent input (kg/day)
                   - Qr_i*(PPr_i/Vr_i))                                   # Outflow from reach. Units: (mm/d)(kg/mm)
        
        # Add results of equations to an array
        res = np.array([dVsA_dt, dQsA_dt, dVsS_dt, dQsS_dt,dVg_dt, dQg_dt, dVr_dt, dQr_dt, dDr_dt, dMsus_dt,
                       dPlabA_dt, dPlabNC_dt, dTDPsA_dt, dTDPsNC_dt, dTDPr_dt, dPPr_dt])          
        return res
    
    # ###########################################################################################
    # --------------------------------------------------------------------------------------------

    # INITIAL CONDITIONS AND DERIVED PARAMETERS
    # Unpack user-supplied initial conditions and calculate other initial conditions which are not modified
    # during looping over met data
    
    # 1) Hydrology
    VsA0 = p['fc']   # Initial soil volume (mm). Assume it's at field capacity.
    VsS0 = VsA0      # Initial soil vol, semi-natural land (mm). Assume same as agricultural for now!!
    Qg0 = p['Qg0_init']      # Initial groundwater flow (mm/d)
    Qr0 = UC_Qinv(p['Qr0_init'], p['A_catch'])  # Convert units of initial reach Q from m3/s to mm/day
    
    # 2) Initial instream suspended sediment mass (kg). Assume it's 0 kg
    Msus0 = 0.0
    
    # 3) Terrestrial P
    # Calculate initial conditions and convert input param units where necessary
    p['Msoil'] = p['Msoil_m2'] * 10**6 * p['A_catch'] # Soil mass (kg). Units: kgSoil/m2 * m2/km2 * km2
    # Inactive soil P mass (kg). Assume equals semi-natural total soil P mass for all LU classes
    p['P_inactive'] = 10**-6*p_LU['S']['SoilPconc']*p['Msoil']
    
    # Varying by LU (intensive agricultural or semi-natural)
    for LU in ['A','S']:
        # Convert units of EPC0 from mg/l to kg/mm to give initial EPC0
        p_LU.ix['EPC0_0',LU] = UC_Cinv(p_LU[LU]['EPC0_init_mgl'], p['A_catch'])
        # Initial labile P. Units: kgP/mgP * mgP/kgSoil * kgSoil. Assume Plab0 = 0 for semi-natural
        p_LU.ix['Plab0',LU] = 10**-6*(p_LU[LU]['SoilPconc']-p_LU['S']['SoilPconc']) * p['Msoil']  
        # Initial soil water TDP mass (kg); Units: (kg/mm)*mm
        p_LU.ix['TDPs0',LU] = p_LU[LU]['EPC0_0'] * (p['f_A']*VsA0+p['f_S']*VsS0)
    
    # Set initial agricultural labile P and soil TDP masses as variables to be updated during looping
    # (assume semi-natural remain at 0 for both)
    Plab0_A, TDPs0_A = p_LU.ix['Plab0','A'], p_LU.ix['TDPs0','A']
    # Initial labile P and soil TDP mass on newly converted land use class
    if p['NC_type']=='S':
        Plab0_NC = Plab0_A  # New class is semi-natural, from arable, therefore start with arable labile P
        TDPs0_NC = TDPs0_A
    else:
        Plab0_NC = 0.0        # New class is arable, converted from SN, therefore start with no labile P
        TDPs0_NC = p_LU.ix['TDPs0','S']
    
    # Set the value for Kf, the adsorption coefficient (mm/kg soil)
    if run_mode == 'cal':
        # If the calibration period, calculate. Assume SN has EPC0=0, PlabConc =0
        Kf = 10**-6*(p_LU['A']['SoilPconc']-p_LU['S']['SoilPconc'])/p_LU['A']['EPC0_0']  # (kg/mg)(mg/kgSoil)(mm/kg)
    else:
        # If not the calibration period, read Kf in from the series of param values
        Kf = p['Kf']

    # 4) Initial instream TDP and PP masses (kg; assume both equal 0.0)
    TDPr0 = 0.0
    PPr0 = 0.0
    
    # --------------------------------------------------------------------------------------------    
    # SETUP ADMIN FOR LOOPING OVER TIME STEPS
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
        # NB this is the flux assuming the land use covered the whole catchment. Divide by catchment
        # area to get areal flux to e.g. compare to literature values
        Mland_i = pd.Series(3*[np.NaN],['A','S','IG'])
        for LU in ['A','S','IG']:
            Mland_i[LU] = p_LU[LU]['E_land']*(Qq_i**p['k_Eland']) #Units: (kg/mm)*(mm/day)
        
        # Calculate dynamic EPC0 as a function of labile P mass
        if dynamic_dict['Dynamic_EPC0'] == 'y':
            # Agricultural EPC0
            EPC0_A_i = Plab0_A/(Kf*p['Msoil']) # First timestep this equals EPC0_0
            # EPC0 on newly-converted land
            EPC0_NC_i = Plab0_NC/(Kf*p['Msoil'])
        # Or, have a constant EPC0 throughout the model run
        else:
            EPC0_A_i = p_LU['A']['EPC0_0']
            # New semi-natural land has agricultural EPC0; new agricultural has SN EPC0
            # NB, there's no point in having a new land class with constant EPC0...
            if p['NC_type'] == 'S':
                EPC0_NC_i = p_LU['A']['EPC0_0']
            else:
                EPC0_NC_i = p_LU['S']['EPC0_0']
        
        # Append to non ODE results
        output_nonODE_i = [Qq_i, Mland_i['A'],Mland_i['IG'],Mland_i['S'], EPC0_A_i, EPC0_NC_i]
        output_nonODE.append(output_nonODE_i)

        # Calculate additional initial conditions from user-input initial conditions/ODE solver output
        QsA0 = (VsA0 - p['fc'])/(p_LU['A']['T_s']*(1 + np.exp(p['fc'] - VsA0))) # Soil flow (mm/d)
        QsS0 = (VsS0 - p['fc'])/(p_LU['S']['T_s']*(1 + np.exp(p['fc'] - VsS0)))
        Vg0 = Qg0 *p['T_g'] # groundwater vol (mm)
        Tr0 = (1000*p['L_reach'])/(p['a_Q']*(Qr0**p['b_Q'])*(8.64*10**7)) #Reach time constant (days); T=L/aQ^b
        Vr0 = Qr0*Tr0 # Reach volume (V=QT) (mm)

        # Vector of initial conditions for start of time step (assume 0 for Dr0, daily mean instream Q)
        y0 = [VsA0, QsA0, VsS0, QsS0, Vg0, Qg0, Vr0, Qr0, 0.0, Msus0, Plab0_A, Plab0_NC,
              TDPs0_A, TDPs0_NC, TDPr0, PPr0]

        # Today's rainfall, ET & model parameters for input to solver. NB the order must be the same as the
        # order in which they are unpacked within the odeint function
        ode_params = [P, E, Qq_i, Mland_i, p['f_A'],
                      p['f_Ar'], p['f_S'], p['f_IG'], p_LU.ix['f_TDPr'], p['NC_type'],
                      p['f_IExcess'],p['alpha'], p['beta'],
                      p_LU.ix['T_s'], p['T_g'], p['fc'],
                      p['L_reach'], p['S_reach'], p['A_catch'],
                      p['a_Q'], p['b_Q'],
                      p['E_Q'], p['k_EQ'],
                      p_LU.ix['P_netInput'], EPC0_A_i, EPC0_NC_i, Kf, p['Msoil'],
                      p['TDPeff'], p['TDPg'],
                      p['E_PP'], p['PPeff'], p['P_inactive']]
        
        # Solve ODEs
        y = odeint(ode_f, y0, ti, args=(ode_params,))
        
        # Extract values for the end of the step
        res = y[1]
        res[res<0] = 0 # Numerical errors may result in very tiny values <0; set these back to 0
        output_ODEs.append(res)

        # Update initial conditions for next step (for Vs, Qg0, Qr0, Msus0, Plab0, TDPs0)
        VsA0 = res[0]
        VsS0 = res[2]
        # FUDGE to re-set groundwater to user-supplied min flow at start of each time step!!!
        if p['Qg_min'] > res[5]:
            Qg0 = p['Qg_min']
        else:
            Qg0 = res[5]
        Qr0 = res[7]
        Msus0 = res[9]
        Plab0_A = res[10]
        Plab0_NC = res[11]
        TDPs0_A = res[12]
        TDPs0_NC = res[13]
        TDPr0 = res[14]
        PPr0 = res[15]
        
        # END LOOP OVER MET DATA
    # --------------------------------------------------------------------------------------------
    
    # Build a dataframe of ODE results
    df1 = pd.DataFrame(data=np.vstack(output_ODEs),
                      columns=['VsA', 'QsA','VsS', 'QsS','Vg', 'Qg', 'Vr', 'Qr', 'Dr',
                               'Msus', 'P_labile_A', 'P_labile_NC', 'TDPs_A', 'TDPs_NC', 'TDPr', 'PPr'],
                      index=met_df.index)
    
    # Dataframe of non ODE results
    df2 = pd.DataFrame(data=np.vstack(output_nonODE), columns=['Qq','Mland_A','Mland_IG','Mland_S',
                       'EPC0_A_kgmm', 'EPC0_NC_kgmm'], index=met_df.index)

    # Concatenate results dataframes
    df = pd.concat([df1,df2], axis=1)

    return (df, Kf)

#%% (standard cell separator)
def P_model_LU_v3(met_df, p, p_LU, dynamic_dict, run_mode, step_len=1):
    """
    Third version of hydrology, sediment and phosphorus model, with processes varying by land use,
    including the ability to simulate a newly converted land use class (semi-natural to agricultural,
    or agricultural to semi-natural).
	
	Includes ODEs for total mass output from stream reach per time step.
	
	Last version of model before adding in sub-catchments and reaches.
	
    Inputs:
        met_df         Dataframe containing columns 'P', the precipitation+snowmelt input to
                       the soil box, and 'PET' (units mm)

        p              Series of parameter values which don't vary by land use (index: param name)
        
        p_LU           Dataframe of parameter values which vary by land use
                       Index: parameter name
                       Columns: 'A', 'S', 'IG' (agricultural, semi-natural, improved grassland)
                       IG only has values for parameter 'E_land' (soil erodibility), in which case
                       value in 'A' column is for arable land

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
    2) Kf, the soil adsorption coefficient (units mm/kg Soil). #Units (mgP/kgSoil)(mm/kgP).
       Multiply by: 10^-6 for (mm/kgSoil), by A_catch*10^6 for (l/kgSoil)
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
        
        # Unpack params. Params that vary by LU are series, with indices ['A','S','IG'],
        # LU-varying params: f_TDPr (fraction of land use for TDP input to reach calc), T_s,P_netInput,EPC0,Mland_i
        (P, E, Qq_i, Mland_i, f_A, f_Ar, f_IG, f_S, f_NC_A, f_NC_Ar, f_NC_IG, f_NC_S, NC_type,
         f_IExcess, alpha, beta, T_s, T_g, fc, L_reach, S_reach, A_catch, a_Q, b_Q, E_Q, k_EQ,
        P_netInput, EPC0_A, EPC0_NC, Kf, Msoil, TDPeff, TDPg, E_PP, PPeff, P_inactive) = ode_params
        
        # Unpack initial conditions for this time step 
        # Hydrology
        VsA_i = y[0] # Agricultural soil water volume (mm)
        QsA_i = y[1] # Agricultural soil water flow (mm/day)
        VsS_i = y[2] # Semi-natural soil water volume (mm)
        QsS_i = y[3] # Semi-natural soil water flow (mm/day)
        Vg_i = y[4]  # Groundwater volume (mm)
        Qg_i = y[5]  # Groundwater discharge (mm/day)
        Vr_i = y[6]  # Reach volume (mm)
        Qr_i = y[7]  # Reach discharge (mm/day)
        #(Qr_av_i would be y[8] here, but it's 0 at the start of every time step)
        # Sediment
        Msus_i = y[9]  # Mass of suspended sediment in the stream reach (kg)
        # (Msus_out_i would be y[10], but it's 0 at the start of every time step)
        # Phosphorus
        PlabA_i = y[11] # Mass of labile P in agricultural soil (kg)
        PlabNC_i = y[12] # Mass of labile P in newly converted land class (kg)
        TDPsA_i = y[13] # Mass of TDP in agricultural soil water (kg)
        TDPsNC_i = y[14] # Mass of TDP in newly converted land class (kg)
        TDPr_i = y[15] # Mass of total dissolved P in stream reach (kg)
        # (TDPr_out_i would be y[16], but it's 0 at the start of every time step)
        PPr_i = y[17]  # Mass of particulate P in stream reach (kg)
        # (PPr_out_i would be y[18], but it's 0 at the start of every time step)
        # Soil water vol & flow for newly converted land class
        if NC_type == 'A':  # If semi-natural converted to arable, assume has arable hydrol
            VsNC_i = VsA_i
            QsNC_i = QsA_i
        else:
            VsNC_i = VsS_i  # If arable converted to SN, assume has semi-natural hydrol
            QsNC_i = QsS_i
    
        # HYDROLOGY
        
        # Soil hydrology equations (units mm or mm/day): Agricultural land
        dQsA_dV = ((((VsA_i - fc)*np.exp(fc - VsA_i))/(T_s['A']*((np.exp(fc-VsA_i) + 1)**2)))
                    +(1/(T_s['A']*(np.exp(fc-VsA_i) + 1))))
        dVsA_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*VsA_i)) - QsA_i
        dQsA_dt = dQsA_dV*dVsA_dt
        
        # Soil hydrology equations (units mm or mm/day): Semi-natural/other land
        dQsS_dV = ((((VsS_i - fc)*np.exp(fc - VsS_i))/(T_s['S']*((np.exp(fc-VsS_i) + 1)**2)))
                    +(1/(T_s['S']*(np.exp(fc-VsS_i) + 1))))
        dVsS_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*VsS_i)) - QsS_i
        dQsS_dt = dQsS_dV*dVsS_dt
        
        # Groundwater equations (units mm or mm/day)
        dQg_dt = (beta*(f_A*QsA_i + f_S*QsS_i) - Qg_i)/T_g
        dVg_dt = beta*(f_A*QsA_i + f_S*QsS_i) - Qg_i
        
        # Instream equations (units mm or mm/day)
        dQr_dt = (((Qq_i + (1-beta)*(f_A*QsA_i + f_S*QsS_i) + Qg_i) - Qr_i)
                  *a_Q*(Qr_i**b_Q)*(8.64*10**7)/((1-b_Q)*(L_reach*1000))) #U/L=1/T. Units (m/s)(s/d)(mm/m)/m(mm/m) 
        dVr_dt = Qq_i + (1-beta)*(f_A*QsA_i + f_S*QsS_i) + Qg_i - Qr_i
        dQr_av_dt = Qr_i  # Daily mean flow
        
        # SEDIMENT
        # Instream suspended sediment (kg; change in kg/day)
        dMsus_dt = (f_Ar*Mland_i['A'] + f_IG*Mland_i['IG'] + f_S* Mland_i['S']     # Delivery from the land (kg/day)
                   + E_Q*S_reach*(Qr_i**k_EQ)                        # Entrainment from the stream bed (kg/d)
                   - (Msus_i/Vr_i)*Qr_i)                             # Outflow from the reach;(kg/mm)*(mm/day)
        
        dMsus_out_dt = Qr_i*Msus_i/Vr_i  # Daily flux of SS
        
        # PHOSPHORUS
        
        # Agricultural soil labile phosphorus mass (kg) (two alternative formulations; give same results)
        # Assume semi-natural land has no labile soil P
        dPlabA_dt = Kf*Msoil*((TDPsA_i/VsA_i)-EPC0_A)  # Net sorption
        # dPlabA_dt = (PlabA_i/EPC0_A)*((TDPsA_i/VsA_i)-EPC0_A)
        
        # Newly-conveted soil labile P mass (kg)
        dPlabNC_dt = Kf*Msoil*((TDPsNC_i/VsNC_i)-EPC0_NC)
        
        # Change in dissolved P mass in agricultural soil water (kg/day)
        # Assume semi-natural land has no dissolved soil water P
        dTDPsA_dt = ((P_netInput['A']*100*A_catch/365)      # Net inputs (fert + manure - uptake) (kg/ha/yr)
                   - Kf*Msoil*((TDPsA_i/VsA_i)-EPC0_A)      # Net sorpn (kg/day)
                                                            # Alternative: (PlabA_i/EPC0_A)*((TDPsA_i/VsA_i)-EPC0_A) 
                   - (QsA_i*TDPsA_i/VsA_i)                  # Outflow via soil water flow (kg/day)
                   - (Qq_i*TDPsA_i/VsA_i))                  # Outflow via quick flow (kg/day)
        
        # And in newly converted land class soil water
        dTDPsNC_dt = ((P_netInput['NC']*100*A_catch/365)    # Net inputs (fert + manure - uptake) (kg/ha/yr)
                   - Kf*Msoil*((TDPsNC_i/VsNC_i)-EPC0_NC)   # Net sorpn (kg/day)
                   - (QsNC_i*TDPsNC_i/VsNC_i)               # Outflow via soil water flow (kg/day)
                   - (Qq_i*TDPsNC_i/VsNC_i))                # Outflow via quick flow (kg/day)
        
        # Change in in-stream TDP mass (kg/d).
        # Semi-natural inputs not specified as assume 0 for soil water & quick flow
        dTDPr_dt = ((1-beta)*f_A*QsA_i*(TDPsA_i/VsA_i)  # Soil water input, old agri land. Units: (mm/d)(kg/mm)
                   + (1-beta)*f_NC_A*QsNC_i*(TDPsNC_i/VsNC_i)  # Soil water input, newly converted agri land (may be 0)
                   + (1-beta)*f_NC_S*QsNC_i*(TDPsNC_i/VsNC_i)  # Soil water input, newly converted SN land (may be 0)
                   + f_A*Qq_i*(TDPsA_i/VsA_i)           # Quick flow input, old agri land. Units: (mm/d)(kg/mm)
                   + f_NC_A*Qq_i*(TDPsNC_i/VsNC_i)      # Quick flow inputs, newly-converted agri (may be 0)
                   + f_NC_S*Qq_i*(TDPsNC_i/VsNC_i)      # Quick flow inputs, newly-converted SN (may be 0)
                   + Qg_i*UC_Cinv(TDPg,A_catch)         # Groundwater input. Units: (mm/d)(kg/mm)
                   + TDPeff                             # Effluent input (kg/day)
                   - Qr_i*(TDPr_i/Vr_i))                # Outflow from the reach. Units: (mm/d)(kg/mm)
        
        dTDPr_out_dt = Qr_i*TDPr_i/Vr_i  # Daily TDP flux out of reach. Units: (mm/d)(kg/mm)=kg/d
        
        # Change in in-stream PP mass (kg/d)
        dPPr_dt = (E_PP *
                   (f_Ar*Mland_i['A']*(PlabA_i+P_inactive)/Msoil         # From old arable land
                   + f_IG*Mland_i['IG']*(PlabA_i+P_inactive)/Msoil       # From old improved grassland
                   + f_S*Mland_i['S']*P_inactive/Msoil)                  # From semi-natural land
                   + f_NC_Ar*Mland_i['A']*(PlabNC_i+P_inactive)/Msoil    # From newly-converted arable
                   + f_NC_IG*Mland_i['IG']*(PlabNC_i+P_inactive)/Msoil   # From newly-converted IG
                   + f_NC_S*Mland_i['S']*(PlabNC_i+P_inactive)/Msoil     # From newly-converted semi-natural
                   + (E_Q*S_reach*(Qr_i**k_EQ))*E_PP*(f_A*(PlabA_i+P_inactive)+f_S*P_inactive)/Msoil  # Entrained                + PPeff                      # Effluent input (kg/day)
                   - Qr_i*(PPr_i/Vr_i))                                  # Outflow from reach. Units: (mm/d)(kg/mm)
        
        dPPr_out_dt = Qr_i*PPr_i/Vr_i  # Daily mean flux
        
        # Add results of equations to an array
        res = np.array([dVsA_dt, dQsA_dt, dVsS_dt, dQsS_dt, dVg_dt, dQg_dt, dVr_dt, dQr_dt, dQr_av_dt,
                        dMsus_dt, dMsus_out_dt, dPlabA_dt, dPlabNC_dt, dTDPsA_dt, dTDPsNC_dt,
                        dTDPr_dt, dTDPr_out_dt, dPPr_dt, dPPr_out_dt])
        return res
    
    # ###########################################################################################
    # --------------------------------------------------------------------------------------------

    # INITIAL CONDITIONS AND DERIVED PARAMETERS
    # Unpack user-supplied initial conditions and calculate other initial conditions which are not modified
    # during looping over met data
    
    # 1) Hydrology
    VsA0 = p['fc']   # Initial soil volume (mm). Assume it's at field capacity.
    VsS0 = VsA0      # Initial soil vol, semi-natural land (mm). Assume same as agricultural for now!!
    Qg0 = p['Qg0_init']      # Initial groundwater flow (mm/d)
    Qr0 = UC_Qinv(p['Qr0_init'], p['A_catch'])  # Convert units of initial reach Q from m3/s to mm/day
    
    # 2) Initial instream suspended sediment mass (kg). Assume it's 0 kg
    Msus0 = 0.0
    
    # 3) Terrestrial P
    # Calculate initial conditions and convert input param units where necessary
    p['Msoil'] = p['Msoil_m2'] * 10**6 * p['A_catch'] # Soil mass (kg). Units: kgSoil/m2 * m2/km2 * km2
    # Inactive soil P mass (kg). Assume equals semi-natural total soil P mass for all LU classes
    p['P_inactive'] = 10**-6*p_LU['S']['SoilPconc']*p['Msoil']
    
    # Varying by LU (intensive agricultural or semi-natural)
    for LU in ['A','S']:
        # Convert units of EPC0 from mg/l to kg/mm to give initial EPC0
        p_LU.ix['EPC0_0',LU] = UC_Cinv(p_LU[LU]['EPC0_init_mgl'], p['A_catch'])
        # Initial labile P. Units: kgP/mgP * mgP/kgSoil * kgSoil. Assume Plab0 = 0 for semi-natural
        p_LU.ix['Plab0',LU] = 10**-6*(p_LU[LU]['SoilPconc']-p_LU['S']['SoilPconc']) * p['Msoil']  
        # Initial soil water TDP mass (kg); Units: (kg/mm)*mm
        p_LU.ix['TDPs0',LU] = p_LU[LU]['EPC0_0'] * (p['f_A']*VsA0+p['f_S']*VsS0)
    
    # Set initial agricultural labile P and soil TDP masses as variables to be updated during looping
    # (assume semi-natural remain at 0 for both)
    Plab0_A, TDPs0_A = p_LU.ix['Plab0','A'], p_LU.ix['TDPs0','A']
    # Initial labile P and soil TDP mass on newly converted land use class
    if p['NC_type']=='S':
        Plab0_NC = Plab0_A  # New class is semi-natural, from arable, therefore start with arable labile P
        TDPs0_NC = TDPs0_A
    else:
        Plab0_NC = 0.0        # New class is arable, converted from SN, therefore start with no labile P
        TDPs0_NC = p_LU.ix['TDPs0','S']
    
    # Set the value for Kf, the adsorption coefficient (mm/kg soil)
    if run_mode == 'cal':
        # If the calibration period, calculate. Assume SN has EPC0=0, PlabConc =0
        Kf = 10**-6*(p_LU['A']['SoilPconc']-p_LU['S']['SoilPconc'])/p_LU['A']['EPC0_0']  # (kg/mg)(mg/kgSoil)(mm/kg)
    else:
        # If not the calibration period, read Kf in from the series of param values
        Kf = p['Kf']

    # 4) Initial instream TDP and PP masses (kg; assume both equal 0.0)
    TDPr0 = 0.0
    PPr0 = 0.0
    
    # --------------------------------------------------------------------------------------------    
    # SETUP ADMIN FOR LOOPING OVER TIME STEPS
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
        # NB this is the flux assuming the land use covered the whole catchment. Divide by catchment
        # area to get areal flux to e.g. compare to literature values
        Mland_i = pd.Series(3*[np.NaN],['A','S','IG'])
        for LU in ['A','S','IG']:
            Mland_i[LU] = p_LU[LU]['E_land']*(Qq_i**p['k_Eland']) #Units: (kg/mm)*(mm/day)
        
        # Calculate dynamic EPC0 as a function of labile P mass
        if dynamic_dict['Dynamic_EPC0'] == 'y':
            # Agricultural EPC0
            EPC0_A_i = Plab0_A/(Kf*p['Msoil']) # First timestep this equals EPC0_0
            # EPC0 on newly-converted land
            EPC0_NC_i = Plab0_NC/(Kf*p['Msoil'])
        # Or, have a constant EPC0 throughout the model run
        else:
            EPC0_A_i = p_LU['A']['EPC0_0']
            # New semi-natural land has agricultural EPC0; new agricultural has SN EPC0
            # NB, there's no point in having a new land class with constant EPC0...
            if p['NC_type'] == 'S':
                EPC0_NC_i = p_LU['A']['EPC0_0']
            else:
                EPC0_NC_i = p_LU['S']['EPC0_0']
        
        # Append to non ODE results
        output_nonODE_i = [Qq_i, Mland_i['A'],Mland_i['IG'],Mland_i['S'], EPC0_A_i, EPC0_NC_i]
        output_nonODE.append(output_nonODE_i)

        # Calculate additional initial conditions from user-input initial conditions/ODE solver output
        QsA0 = (VsA0 - p['fc'])/(p_LU['A']['T_s']*(1 + np.exp(p['fc'] - VsA0))) # Soil flow (mm/d)
        QsS0 = (VsS0 - p['fc'])/(p_LU['S']['T_s']*(1 + np.exp(p['fc'] - VsS0)))
        Vg0 = Qg0 *p['T_g'] # groundwater vol (mm)
        Tr0 = (1000*p['L_reach'])/(p['a_Q']*(Qr0**p['b_Q'])*(8.64*10**7)) #Reach time constant (days); T=L/aQ^b
        Vr0 = Qr0*Tr0 # Reach volume (V=QT) (mm)

        # Vector of initial conditions for start of time step (assume 0 for initial Qr_av, Msus_av, TDPr_av
        # and PPr_av, the daily fluxes of water or mass
        y0 = [VsA0, QsA0, VsS0, QsS0, Vg0, Qg0, Vr0, Qr0, 0.0, Msus0, 0.0, Plab0_A, Plab0_NC,
              TDPs0_A, TDPs0_NC, TDPr0, 0.0, PPr0, 0.0]

        # Today's rainfall, ET & model parameters for input to solver. NB the order must be the same as the
        # order in which they are unpacked within the odeint function
        ode_params = [P, E, Qq_i, Mland_i,
                      p['f_A'],p['f_Ar'],p['f_IG'],p['f_S'],
                      p['f_NC_A'],p['f_NC_Ar'],p['f_NC_IG'],p['f_NC_S'],p['NC_type'],
                      p['f_IExcess'],p['alpha'], p['beta'],
                      p_LU.ix['T_s'], p['T_g'], p['fc'],
                      p['L_reach'], p['S_reach'], p['A_catch'],
                      p['a_Q'], p['b_Q'],
                      p['E_Q'], p['k_EQ'],
                      p_LU.ix['P_netInput'], EPC0_A_i, EPC0_NC_i, Kf, p['Msoil'],
                      p['TDPeff'], p['TDPg'],
                      p['E_PP'], p['PPeff'], p['P_inactive']]
        
        # Solve ODEs
        y = odeint(ode_f, y0, ti, args=(ode_params,))
        
        # Extract values for the end of the step
        res = y[1]
        res[res<0] = 0 # Numerical errors may result in very tiny values <0; set these back to 0
        output_ODEs.append(res)
        
        # Update initial conditions for next step (for Vs, Qg0, Qr0, Msus0, Plab0, TDPs0)
        VsA0 = res[0]  # QsA0 would be res[1]
        VsS0 = res[2]  # QsS0 would be res[3]
        # Vg0 would be res[4], but calculate from Qg0
        # FUDGE to re-set groundwater to user-supplied min flow at start of each time step!!!
        if p['Qg_min'] > res[5]:
            Qg0 = p['Qg_min']
        else:
            Qg0 = res[5]
        # Vr0 would be res[6], but calculate from Qr0 instead
        Qr0 = res[7]
        # Qr_av_0 would be res[8], but it's always 0
        Msus0 = res[9]
        # Msus_out_0 would be res[10]
        Plab0_A = res[11]
        Plab0_NC = res[12]
        TDPs0_A = res[13]
        TDPs0_NC = res[14]
        TDPr0 = res[15] # TDPr_out_0 would be res[16], but it's always 0
        PPr0 = res[17] # TDPr_out_0 would be res[18], but it's always 0
        
        # END LOOP OVER MET DATA
    # --------------------------------------------------------------------------------------------
    
    # Build a dataframe of ODE results
    df1 = pd.DataFrame(data=np.vstack(output_ODEs),
                       columns=['VsA', 'QsA','VsS', 'QsS', 'Vg', 'Qg', 'Vr', 'Qr', 'Qr_av',
                               'Msus', 'Msus_out_kgd', 'P_labile_A', 'P_labile_NC', 'TDPs_A', 'TDPs_NC',
                               'TDPr', 'TDPr_out_kgd', 'PPr', 'PPr_out_kgd'],
                       index=met_df.index)
    
    # Dataframe of non ODE results
    df2 = pd.DataFrame(data=np.vstack(output_nonODE), columns=['Qq','Mland_A','Mland_IG','Mland_S',
                       'EPC0_A_kgmm', 'EPC0_NC_kgmm'], index=met_df.index)

    # Concatenate results dataframes
    df = pd.concat([df1,df2], axis=1)
    return (df, Kf)
# #################################################################################################
	

#%% (standard cell separator)
def P_model_v1(met_df, p_LU, p_SC, p, dynamic_dict, run_mode, step_len=1):
    """
    Simple hydrology, sediment and phosphorus model, with processes varying by land use,
	sub-catchment and reach. First version of full working model. Still contains very simple
	erosion from the land and relationship between labile & soluble P in the soil
	
	Saved 13/01/2016

    Inputs:
        met_df         Dataframe containing columns 'P', the precipitation+snowmelt input to
                       the soil box, and 'PET' (units mm)

        p              Series of parameter values which don't vary by land use (index: param name)
        
        p_LU           Dataframe of parameter values which vary by land use
                       Index:   parameter name
                       Columns: 'A', 'S', 'IG' (agricultural, semi-natural, improved grassland)
                                IG only has values for parameter 'E_land' (soil erodibility), in
								which case value in 'A' column is for arable land
        
        p_SC           Dataframe of parameter values which vary by sub-catchment or reach
                       Index:   parameter name
                       Columns: 1, 2, ...n sub-catchments

        dynamic_dict   Dictionary of options controlling whether inputs/variables are calculated
                       dynamically or kept constant.
                       Dictionary keys: 'Dynamic_EPC0', 'Dynamic_effluent_inputs',
                                        'Dynamic_terrestrialP_inputs', 'Dynamic_GW_TDP'. 'y' or 'n'
                       **NB only the first is implemented in the model at the moment***

        run_mode       'cal' or 'val'. Determines whether the soil sorption coefficient, Kf,
                       is calculated (calibration period) or read in (validation period)

        step_len       Length of each step in the input dataset (days). Default=1

    Returns a three-element tuple (df, Kf):
    
    1)  df_TC_dict: A dictionary containing dataframes of results for the terrestrial compartment,
        with one dataframe per sub-catchment (key is the sub-catchment number as an integer).
		The dataframe has column headings:
        
        VsA:          Soil water volume, agricultural land (mm)
        VsS:          Soil water volume, semi-natural land (mm)
        QsA:          Soil water flow, agricultural land (mm/d)
        QsS:          Soil water flow, semi-natural land (mm/d)
        Vg:           Groundwater volume (mm)
        Qg:           Groundwater flow (mm/d)
        Qq:           Quick flow (mm/d)
        Mland_A:      Sediment mass delivered to the stream, arable land (kg/d),
        Mland_IG:     Sediment mass delivered to the stream, improved grassland (kg/d)
        Mland_S:      Sediment mass delivered to the stream, semi-natural land (kg/d)
        P_labile_A:   Labile soil P mass in agricultural soil (kg)
        P_labile_NC:  Labile soil P mass in newly-converted agricultural or semi-natural land (kg)
        EPC0_A_kgmm:  EPC0 in agricultural soil (kg/mm)
        EPC0_NC_kgmm: EPC0 in newly-converted agricultural or semi-natural soil (kg/mm)       
        TDPs_A:       Soil water TDP mass, agricultural land (kg)
        TDPs_NC:      Soil water TDP mass, newly-converted agricultural or semi-natural land (kg)

    2)  df_R_dict: A dictionary containing dataframes of results for the stream reach, with one
        dataframe per sub-catchment (dictionary key is the sub-catchment number as an integer).
		Dataframe column headings:
        
        Vr:   Reach volume (mm)
        Qr:   Mean daily discharge from the reach (mm/d)
        Msus: Daily flux of suspended sediment from the reach (kg/day)
        TDPr: Daily flux of TDP from the reach (kg/day)
        PPr:  Daily flux of PP from the reach (kg/day)
        
        Note that instantaneous fluxes of in-stream Q and masses of SS, PP and TDP are also
		calculated by the model, and could be saved to this output dataframe if there was a need,
		rather than these daily fluxes. However, daily fluxes are used for calculating mean
		volume-weighted concentrations.

    3)  Kf: the soil adsorption coefficient (units mm/kg Soil). From (mgP/kgSoil)(mm/kgP).
        Multiply by: 10^-6 for (mm/kgSoil), by A_catch*10^6 for (l/kgSoil)
    """
    
    #########################################################################################
    # Define the ODE system
    def ode_f(y, t, ode_params):
        """
        Define ODE system
        Inputs:
            y: list of variables expressed as dy/dx. y is determined for the end of the time step
            t: array of time points of interest
            params: tuple of input values & model parameter values
        """
        
        # Unpack params. Params that vary by LU are series, with indices ['A','S','IG'],
        # LU-varying params: T_s,P_netInput,EPC0,Mland_i
        (P, E, Qq_i, Qr_US_i, Mland_i, Msus_US_i, TDPr_US_i, PPr_US_i,
         f_A, f_Ar, f_IG, f_S, f_NC_A, f_NC_Ar, f_NC_IG, f_NC_S, NC_type,
         f_IExcess, alpha, beta, T_s, T_g, fc, L_reach, S_reach, A_catch, a_Q, b_Q, E_Q, k_EQ,
         P_netInput, EPC0_A, EPC0_NC, Kf, Msoil, TDPeff, TDPg, E_PP, P_inactive) = ode_params
        
        # Unpack initial conditions for this time step
        # Hydrology
        VsA_i = y[0] # Agricultural soil water volume (mm)
        QsA_i = y[1] # Agricultural soil water flow (mm/day)
        VsS_i = y[2] # Semi-natural soil water volume (mm)
        QsS_i = y[3] # Semi-natural soil water flow (mm/day)
        Vg_i = y[4]  # Groundwater volume (mm)
        Qg_i = y[5]  # Groundwater discharge (mm/day)
        Vr_i = y[6]  # Reach volume (mm)
        Qr_i = y[7]  # Instantaneous reach discharge (mm/day)
        #(Qr_av_i would be y[8] here, but it's 0 at the start of every time step)
        # Sediment
        Msus_i = y[9]  # Mass of suspended sediment in the stream reach (kg)
        # (Msus_out_i would be y[10], but it's 0 at the start of every time step)
        # Phosphorus
        PlabA_i = y[11] # Mass of labile P in agricultural soil (kg)
        PlabNC_i = y[12] # Mass of labile P in newly converted land class (kg)
        TDPsA_i = y[13] # Mass of TDP in agricultural soil water (kg)
        TDPsNC_i = y[14] # Mass of TDP in newly converted land class (kg)
        TDPr_i = y[15] # Mass of total dissolved P in stream reach (kg)
        # (TDPr_out_i would be y[16], but it's 0 at the start of every time step)
        PPr_i = y[17]  # Mass of particulate P in stream reach (kg)
        # (PPr_out_i would be y[18], but it's 0 at the start of every time step)
        # Soil water vol & flow for newly converted land class
        if NC_type == 'A':  # If semi-natural converted to arable, assume has arable hydrol
            VsNC_i = VsA_i
            QsNC_i = QsA_i
        else:
            VsNC_i = VsS_i  # If arable converted to SN, assume has semi-natural hydrol
            QsNC_i = QsS_i
    
        # HYDROLOGY
        
        # Soil hydrology equations (units mm or mm/day): Agricultural land
        dQsA_dV = ((((VsA_i - fc)*np.exp(fc - VsA_i))/(T_s['A']*((np.exp(fc-VsA_i) + 1)**2)))
                    +(1/(T_s['A']*(np.exp(fc-VsA_i) + 1))))
        dVsA_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*VsA_i)) - QsA_i
        dQsA_dt = dQsA_dV*dVsA_dt
        
        # Soil hydrology equations (units mm or mm/day): Semi-natural/other land
        dQsS_dV = ((((VsS_i - fc)*np.exp(fc - VsS_i))/(T_s['S']*((np.exp(fc-VsS_i) + 1)**2)))
                    +(1/(T_s['S']*(np.exp(fc-VsS_i) + 1))))
        dVsS_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*VsS_i)) - QsS_i
        dQsS_dt = dQsS_dV*dVsS_dt
        
        # Groundwater equations (units mm or mm/day)
        dQg_dt = (beta*(f_A*QsA_i + f_S*QsS_i) - Qg_i)/T_g
        dVg_dt = beta*(f_A*QsA_i + f_S*QsS_i) - Qg_i
        
        # Instream equations (units mm or mm/day)
        dQr_dt = ((Qq_i + (1-beta)*(f_A*QsA_i + f_S*QsS_i) + Qg_i + Qr_US_i - Qr_i) # Fluxes (mm/d)
                  *a_Q*(Qr_i**b_Q)*(8.64*10**7)/((1-b_Q)*(L_reach*1000)))
				  # 2nd row is U/L=1/T. Units:(m/s)(s/d)(mm/m)(1/m)(m/mm)
        dVr_dt = Qq_i + (1-beta)*(f_A*QsA_i + f_S*QsS_i) + Qg_i + Qr_US_i - Qr_i
        dQr_av_dt = Qr_i  # Daily mean flow
        
        # SEDIMENT
        # Instream suspended sediment (kg; change in kg/day)
        dMsus_dt = (f_Ar*Mland_i['A'] + f_IG*Mland_i['IG'] + f_S* Mland_i['S']  # From land (kg/day)
                   + E_Q*S_reach*(Qr_i**k_EQ)             # Entrainment from the stream bed (kg/d)
                   + Msus_US_i                            # Inputs from upstream
                   - (Msus_i/Vr_i)*Qr_i)                  # Outflow from the reach;(kg/mm)*(mm/day)
        
        dMsus_out_dt = Qr_i*Msus_i/Vr_i  # Daily flux of SS
        
        # PHOSPHORUS
        
        # Agricultural soil labile P mass (kg). Assume semi-natural land has no labile soil P
        dPlabA_dt = Kf*Msoil*((TDPsA_i/VsA_i)-EPC0_A)  # Net sorption
        # dPlabA_dt = (PlabA_i/EPC0_A)*((TDPsA_i/VsA_i)-EPC0_A) # Alternative sorpn formulation
        
        # Newly-conveted soil labile P mass (kg)
        dPlabNC_dt = Kf*Msoil*((TDPsNC_i/VsNC_i)-EPC0_NC)
        
        # Change in dissolved P mass in agricultural soil water (kg/day)
        # Assume semi-natural land has no dissolved soil water P
        dTDPsA_dt = ((P_netInput['A']*100*A_catch/365)  # Net inputs (fert+manure-uptake) (kg/ha/yr)
                   - Kf*Msoil*((TDPsA_i/VsA_i)-EPC0_A)  # Net sorpn (kg/day) (could be alt above)
                   - (QsA_i*TDPsA_i/VsA_i)              # Outflow via soil water flow (kg/day)
                   - (Qq_i*TDPsA_i/VsA_i))              # Outflow via quick flow (kg/day)
        
        # And in newly converted land class soil water
        dTDPsNC_dt = ((P_netInput['NC']*100*A_catch/365)    # Net inputs (kg/ha/yr)
                   - Kf*Msoil*((TDPsNC_i/VsNC_i)-EPC0_NC)   # Net sorpn (kg/day)
                   - (QsNC_i*TDPsNC_i/VsNC_i)               # Outflow via soil water flow (kg/day)
                   - (Qq_i*TDPsNC_i/VsNC_i))                # Outflow via quick flow (kg/day)
        
        # Change in in-stream TDP mass (kg/d)
        # Semi-natural inputs not specified as assume 0 for soil water & quick flow
        dTDPr_dt = ((1-beta)*f_A*QsA_i*(TDPsA_i/VsA_i)  # Soil input, old agri. Units:(mm/d)(kg/mm)
                   + (1-beta)*f_NC_A*QsNC_i*(TDPsNC_i/VsNC_i)  # Soil input, new agri land
                   + (1-beta)*f_NC_S*QsNC_i*(TDPsNC_i/VsNC_i)  # Soil input, new SN land
                   + f_A*Qq_i*(TDPsA_i/VsA_i)           # Quick input, old agri. Units:(mm/d)(kg/mm)
                   + f_NC_A*Qq_i*(TDPsNC_i/VsNC_i)      # Quick input, newly-converted agri
                   + f_NC_S*Qq_i*(TDPsNC_i/VsNC_i)      # Quick inputs, newly-converted SN
                   + Qg_i*UC_Cinv(TDPg,A_catch)         # Groundwater input. Units: (mm/d)(kg/mm)
                   + TDPeff                             # Effluent input (kg/day)
                   + TDPr_US_i                          # Inputs from upstream 
                   - Qr_i*(TDPr_i/Vr_i))                # Reach outflow. Units: (mm/d)(kg/mm)
        
        dTDPr_out_dt = Qr_i*TDPr_i/Vr_i  # Daily TDP flux out of reach. Units: (mm/d)(kg/mm)=kg/d
        
        # Change in in-stream PP mass (kg/d)
        dPPr_dt = (E_PP *
                   (f_Ar*Mland_i['A']*(PlabA_i+P_inactive)/Msoil         # Old arable land
                   + f_IG*Mland_i['IG']*(PlabA_i+P_inactive)/Msoil       # Old improved grassland
                   + f_S*Mland_i['S']*P_inactive/Msoil)                  # Semi-natural land
                   + f_NC_Ar*Mland_i['A']*(PlabNC_i+P_inactive)/Msoil    # Newly-converted arable
                   + f_NC_IG*Mland_i['IG']*(PlabNC_i+P_inactive)/Msoil   # Newly-converted IG
                   + f_NC_S*Mland_i['S']*(PlabNC_i+P_inactive)/Msoil     # New semi-natural
                   + (E_Q*S_reach*(Qr_i**k_EQ))*E_PP*(f_A*(PlabA_i+P_inactive)+f_S*P_inactive)/Msoil
																		 # Entrained
                   + PPr_US_i                                            # Inputs from upstream 
                   - Qr_i*(PPr_i/Vr_i))                               # Reach outflow (mm/d)(kg/mm)
        
        dPPr_out_dt = Qr_i*PPr_i/Vr_i  # Daily mean flux
        
        # Add results of equations to an array
        res = np.array([dVsA_dt, dQsA_dt, dVsS_dt, dQsS_dt, dVg_dt, dQg_dt, dVr_dt, dQr_dt,
						dQr_av_dt, dMsus_dt, dMsus_out_dt, dPlabA_dt, dPlabNC_dt, dTDPsA_dt,
						dTDPsNC_dt, dTDPr_dt, dTDPr_out_dt, dPPr_dt, dPPr_out_dt])
        return res
    
    ##########################################################################################
    #-----------------------------------------------------------------------------------------
    # SETUP ADMIN FOR LOOPING OVER TIME STEPS AND SUB-CATCHMENTS
    
    # Dictionary to store results in for different sub-catchments
    df_TC_dict = {} # Key: Sub-catchment number (int); returns df of terrestrial compartment results
    df_R_dict = {}  # Key: Sub-catchment number (int); returns df of in-stream (reach) results
    
    # Time points to evaluate ODEs at (we're only interested in the start and end of each step)
    ti = [0, step_len]

    #-----------------------------------------------------------------------------------------
    # INITIAL CONDITIONS AND DERIVED PARAMETERS THAT DON'T VARY BY SUB-CATCHMENT
    # Unpack user-supplied initial conditions, calculate any others, convert units
    
    # 1) Terrestrial - constant over LU and SC
    # Hydrol
    VsA0 = p['fc']   # Initial soil volume (mm). Assume it's at field capacity.
    VsS0 = VsA0      # Initial soil vol, semi-natural land (mm). Assumed same as agricultural!!
    Qg0 = p['Qg0_init']      # Initial groundwater flow (mm/d)
    
    #-----------------------------------------------------------------------------------------
    # START LOOP OVER SUB-CATCHMENTS
    for SC in p['SC_list']:
        
        # INITIAL CONDITIONS THAT VARY BY SUB-CATCHMENT
    
        # Soil mass and inactive soil P (kg)
		# Assume inactive soil P is equivalent to semi-natural total soil P for all LU classes
        Msoil = p['Msoil_m2']*10**6*p_SC.ix['A_catch',SC] # Soil mass (kg): (kgSoil/m2)(m2/km2)km2
        P_inactive = 10**-6*p_LU['S']['SoilPconc']*Msoil

        # 2) Terrestrial - varying by land use and by sub-catchment
        for LU in ['A','S']:
            # Convert units of EPC0 from mg/l to kg/mm to give initial EPC0
            p_LU.ix['EPC0_0',LU] = UC_Cinv(p_LU[LU]['EPC0_init_mgl'], p_SC.ix['A_catch',SC])
            ### CHECK units here! Should same mg/l conc give diff kg/mm conc in diff SCs?????
            
            # Initial labile P. Units: (kgP/mgP)(mgP/kgSoil)kgSoil. Assume Plab0=0 for semi-natural
            p_LU.ix['Plab0',LU] = 10**-6*(p_LU[LU]['SoilPconc']-p_LU['S']['SoilPconc']) * Msoil
            # Initial soil water TDP mass (kg); Units: (kg/mm)*mm
            if LU == 'A':
                p_LU.ix['TDPs0',LU] = p_LU[LU]['EPC0_0']*VsA0
            else:
                p_LU.ix['TDPs0',LU] = 0
                        
        # Set initial agricultural labile P and soil TDP masses as variables to be updated during
		# looping (assume semi-natural remain at 0 for both)
        Plab0_A, TDPs0_A = p_LU.ix['Plab0','A'], p_LU.ix['TDPs0','A']
        # Initial labile P and soil TDP mass on newly converted land use class
        if p_SC.ix['NC_type',SC]=='S':
            Plab0_NC = Plab0_A  # New class is SN, from arable, therefore start with arable labile P
            TDPs0_NC = TDPs0_A
        else:
            Plab0_NC = 0.0      # New class is arable, from SN, therefore start with no labile P
            TDPs0_NC = p_LU.ix['TDPs0','S']

        # Set the value for Kf, the adsorption coefficient (mm/kg soil)
        if run_mode == 'cal': # If the calibration period, calculate.
			# Assume SN has EPC0=0, PlabConc =0. Units: (kg/mg)(mg/kgSoil)(mm/kg)
            Kf = 10**-6*(p_LU['A']['SoilPconc']-p_LU['S']['SoilPconc'])/p_LU['A']['EPC0_0']  
        else:  # If not the calibration period, read Kf in from the series of param values
            Kf = p['Kf']

        # 3) In-stream initial conditions (all reaches the same for chem)
        TDPr0, PPr0, Msus0 = 0.0, 0.0, 0.0  # Initial TDP, PP & SS masses (kg; assume all equal 0.0)
        # Initial in-stream flow
        if SC == 1:
			# Convert units of initial reach Q from m3/s to mm/day
            Qr0 = UC_Qinv(p['Qr0_init'], p_SC.ix['A_catch',SC])
        else:        # Outflow from the reach upstream from the first time step
            Qr0 = df_R_dict[SC-1].ix[0,'Qr']  # N.B. 'Qr' here is Qr_av, the daily mean flow
        
        # ADMIN
        # Lists to store output
        output_ODEs = []    # From ode_f function
        output_nonODE = []  # Will include: Qq, Mland, Qr_US (latter for checking only)        
        
		#-------------------------------------------------------------------------------------
        # START LOOP OVER MET DATA
        for idx in range(len(met_df)):

            # Get precipitation and evapotranspiration for this day
            P = met_df.ix[idx, 'P']
            E = met_df.ix[idx, 'PET']

            # Calculate infiltration excess (mm/(day * catchment area))
            Qq_i = p['f_IExcess']*P
            
            # Inputs to reach from up-stream reaches
            if SC == 1:
                # For the top reach, set the input from upstream reaches to 0
                Qr_US_i, Msus_US_i, TDPr_US_i, PPr_US_i = 0.0, 0.0, 0.0, 0.0
            else:
                # Below reach 1, the upstream input is the daily mean flux from up-stream for the
				# current day
                Qr_US_i = df_R_dict[SC-1].ix[idx,'Qr']
                Msus_US_i = df_R_dict [SC-1].ix[idx, 'Msus']
                TDPr_US_i = df_R_dict [SC-1].ix[idx, 'TDPr']
                PPr_US_i = df_R_dict [SC-1].ix[idx, 'PPr']

            # Calculate delivery of sediment from the land to the stream (kg/day). NB this flux
            # assumes the land use covers the whole catchment. Divide by area to get areal flux
            Mland_i = pd.Series(3*[np.NaN],['A','S','IG'])
            for LU in ['A','S','IG']:
                Mland_i[LU] = p_LU[LU]['E_land']*(Qq_i**p['k_Eland']) #Units: (kg/mm)*(mm/day)

            # Calculate dynamic EPC0 as a function of labile P mass
            if dynamic_dict['Dynamic_EPC0'] == 'y':
                EPC0_A_i = Plab0_A/(Kf*Msoil) # Agricultural EPC0; equals EPC0_0 on the 1st timestep
                EPC0_NC_i = Plab0_NC/(Kf*Msoil) # EPC0 on newly-converted land
            # Or, have a constant EPC0 throughout the model run
            else:
                EPC0_A_i = p_LU['A']['EPC0_0']
                if p_SC.ix['NC_type',SC] == 'S': # (little point in a new class with constant EPC0)
                    EPC0_NC_i = p_LU['A']['EPC0_0']  # New semi-natural land has agricultural EPC0
                else:
                    EPC0_NC_i = p_LU['S']['EPC0_0']  # New agricultural has SN EPC0

            # Append to non-ODE results
#             output_nonODE_i = [Qq_i, Qr_US_i, Mland_i['A'],Mland_i['IG'],Mland_i['S'], EPC0_A_i,
#								EPC0_NC_i, Msus_US_i, TDPr_US_i, PPr_US_i]
            output_nonODE_i = [Qq_i, Mland_i['A'],Mland_i['IG'],Mland_i['S'], EPC0_A_i, EPC0_NC_i]
            output_nonODE.append(output_nonODE_i)

            # Calculate additional initial conditions from user-input values/ODE solver output
            # Soil flow, agricultural (mm/d)
			QsA0 = (VsA0 - p['fc'])/(p_LU['A']['T_s']*(1 + np.exp(p['fc'] - VsA0)))
            # Soil flow, semi-natural (mm/d)
			QsS0 = (VsS0 - p['fc'])/(p_LU['S']['T_s']*(1 + np.exp(p['fc'] - VsS0)))
            Vg0 = Qg0 *p['T_g']     # Groundwater vol (mm)
            Tr0 = ((1000*p_SC.ix['L_reach',SC])/
                   (p['a_Q']*(Qr0**p['b_Q'])*(8.64*10**7))) # Reach time constant (days); T=L/aQ^b
            Vr0 = Qr0*Tr0 # Reach volume (V=QT) (mm)

            # Vector of initial conditions for start of time step (assume 0 for initial Qr_av,
			# Msus_out, TDPr_out and PPr_out, the mass or vol of water lost per time step
            y0 = [VsA0, QsA0, VsS0, QsS0, Vg0, Qg0, Vr0, Qr0, 0.0, Msus0, 0.0, Plab0_A, Plab0_NC,
                  TDPs0_A, TDPs0_NC, TDPr0, 0.0, PPr0, 0.0]

            # Today's rainfall, ET & model parameters for input to solver. NB the order must be the
			# same as the order in which they are unpacked within the odeint function
            ode_params = [P, E, Qq_i, Qr_US_i, Mland_i, Msus_US_i, TDPr_US_i, PPr_US_i,
                          p_SC.ix['f_A',SC],p_SC.ix['f_Ar',SC],p_SC.ix['f_IG',SC],p_SC.ix['f_S',SC],
                          p_SC.ix['f_NC_A',SC],p_SC.ix['f_NC_Ar',SC],p_SC.ix['f_NC_IG',SC],
                          p_SC.ix['f_NC_S',SC],p_SC.ix['NC_type',SC],
                          p['f_IExcess'],p['alpha'], p['beta'],
                          p_LU.ix['T_s'], p['T_g'], p['fc'],
                          p_SC.ix['L_reach',SC], p_SC.ix['S_reach',SC], p_SC.ix['A_catch',SC],
                          p['a_Q'], p['b_Q'], p['E_Q'], p['k_EQ'],
                          p_LU.ix['P_netInput'], EPC0_A_i, EPC0_NC_i, Kf, Msoil,
                          p_SC.ix['TDPeff',SC], p['TDPg'], p['E_PP'], P_inactive]

            # Solve ODEs
            y = odeint(ode_f, y0, ti, args=(ode_params,))

            # Extract values for the end of the step
            res = y[1]
            res[res<0] = 0 # Numerical errors may result in very tiny values <0; set these back to 0
            output_ODEs.append(res)

            # Update initial conditions for next step (for Vs, Qg0, Qr0, Msus0, Plab0, TDPs0)
            VsA0 = res[0]  # QsA0 would be res[1]
            VsS0 = res[2]  # QsS0 would be res[3], Vg0 would be res[4] (but calculate from Qg0)
            # Re-set groundwater to user-supplied minimum flow at start of each time step. Non-ideal
            # solution to the problem or maintaining stream flow during baseflow conditions!!!
            if p['Qg_min'] > res[5]:
                Qg0 = p['Qg_min']
            else:
                Qg0 = res[5]
            # Vr0 would be res[6], but calculate from Qr0 instead
            Qr0 = res[7]       # Qr_av_0 would be res[8], but it's always 0
            Msus0 = res[9]     # Msus_out_0 would be res[10]
            Plab0_A = res[11]
            Plab0_NC = res[12]
            TDPs0_A = res[13]
            TDPs0_NC = res[14]
            TDPr0 = res[15]    # TDPr_out_0 would be res[16], but it's always 0
            PPr0 = res[17]     # TDPr_out_0 would be res[18], but it's always 0

            # END LOOP OVER MET DATA
        #-------------------------------------------------------------------------------------
    
        # Build a dataframe of ODE results
        df_ODE = pd.DataFrame(data=np.vstack(output_ODEs),
                           columns=['VsA', 'QsA','VsS', 'QsS', 'Vg', 'Qg', 'Vr', 'Qr_instant', 'Qr',
                           'Msus_instant', 'Msus', 'P_labile_A', 'P_labile_NC', 'TDPs_A', 'TDPs_NC',
                           'TDPr_instant', 'TDPr', 'PPr_instant', 'PPr'], index=met_df.index)

        # Dataframe of non ODE results
        df_nonODE = pd.DataFrame(data=np.vstack(output_nonODE),
								columns=['Qq','Mland_A','Mland_IG','Mland_S',
                                'EPC0_A_kgmm', 'EPC0_NC_kgmm'], index=met_df.index)

        # Rearrange ODE and non-ODE result dataframes into results dataframes for the terrestrial
        # compartment (df_TC) and the stream reach (df_R)
        df_TC = pd.concat([df_ODE[['VsA', 'QsA','VsS', 'QsS', 'Vg', 'Qg', 'P_labile_A',
						'P_labile_NC','TDPs_A', 'TDPs_NC']],df_nonODE], axis=1)
        # NB masses of SS and P are all total fluxes for the day, and Q is the daily mean flow
        df_R = df_ODE[['Vr', 'Qr','Msus','TDPr', 'PPr']]
        
        # Add to dictionaries; key = sub-catchment/reach number
        df_TC_dict[SC] = df_TC
        df_R_dict[SC] = df_R
        
        # END LOOP OVER SUB-CATCHMENTS
        #-------------------------------------------------------------------------------------

    return (df_TC_dict, df_R_dict, Kf)  # NB Kf is returned for the last SC
	
# #################################################################################################
	

#%% (standard cell separator)
def P_model_v2_beta1(met_df, p_LU, p_SC, p, dynamic_dict, run_mode, step_len=1):
    """
    Simple hydrology, sediment and phosphorus model, with processes varying by land use,
    sub-catchment and reach.
	
	v2_beta1: Altered land sediment equations - more process-based, and with a dynamic option.

    Inputs:
        met_df         Dataframe containing columns 'P', the precipitation+snowmelt input to
                       the soil box, and 'PET' (units mm)

        p              Series of parameter values which don't vary by land use (index: param name)
        
        p_LU           Dataframe of parameter values which vary by land use
                       Index:   parameter name
                       Columns: 'A', 'S', 'IG' (agricultural, semi-natural, improved grassland)
                                IG only has values for parameter 'E_land' (soil erodibility), in
                                which case value in 'A' column is for arable land
        
        p_SC           Dataframe of parameter values which vary by sub-catchment or reach
                       Index:   parameter name
                       Columns: 1, 2, ...n sub-catchments

        dynamic_dict   Dictionary of options controlling whether inputs/variables are calculated
                       dynamically or kept constant.
                       Dictionary keys: 'Dynamic_EPC0', 'Dynamic_effluent_inputs',
                                        'Dynamic_terrestrialP_inputs', 'Dynamic_GW_TDP'. 'y' or 'n'
                       **NB only the first is implemented in the model at the moment***

        run_mode       'cal' or 'val'. Determines whether the soil sorption coefficient, Kf,
                       is calculated (calibration period) or read in (validation period)

        step_len       Length of each step in the input dataset (days). Default=1

    Returns a three-element tuple (df, Kf):
    
    1)  df_TC_dict: A dictionary containing dataframes of results for the terrestrial compartment,
        with one dataframe per sub-catchment (key is the sub-catchment number as an integer).
        The dataframe has column headings:
        
        VsA:          Soil water volume, agricultural land (mm)
        VsS:          Soil water volume, semi-natural land (mm)
        QsA:          Soil water flow, agricultural land (mm/d)
        QsS:          Soil water flow, semi-natural land (mm/d)
        Vg:           Groundwater volume (mm)
        Qg:           Groundwater flow (mm/d)
        Qq:           Quick flow (mm/d)
        Mland_A:      Sediment mass delivered to the stream, arable land (kg/d),
        Mland_IG:     Sediment mass delivered to the stream, improved grassland (kg/d)
        Mland_S:      Sediment mass delivered to the stream, semi-natural land (kg/d)
        P_labile_A:   Labile soil P mass in agricultural soil (kg)
        P_labile_NC:  Labile soil P mass in newly-converted agricultural or semi-natural land (kg)
        EPC0_A_kgmm:  EPC0 in agricultural soil (kg/mm)
        EPC0_NC_kgmm: EPC0 in newly-converted agricultural or semi-natural soil (kg/mm)       
        TDPs_A:       Soil water TDP mass, agricultural land (kg)
        TDPs_NC:      Soil water TDP mass, newly-converted agricultural or semi-natural land (kg)

    2)  df_R_dict: A dictionary containing dataframes of results for the stream reach, with one
        dataframe per sub-catchment (dictionary key is the sub-catchment number as an integer).
        Dataframe column headings:
        
        Vr:   Reach volume (mm)
        Qr:   Mean daily discharge from the reach (mm/d)
        Msus: Daily flux of suspended sediment from the reach (kg/day)
        TDPr: Daily flux of TDP from the reach (kg/day)
        PPr:  Daily flux of PP from the reach (kg/day)
        
        Note that instantaneous fluxes of in-stream Q and masses of SS, PP and TDP are also
        calculated by the model, and could be saved to this output dataframe if there was a need,
        rather than these daily fluxes. However, daily fluxes are used for calculating mean
        volume-weighted concentrations.

    3)  Kf: the soil adsorption coefficient (units mm/kg Soil). From (mgP/kgSoil)(mm/kgP).
        Multiply by: 10^-6 for (mm/kgSoil), by A_catch*10^6 for (l/kgSoil)
    """
    
    #########################################################################################
    # Define the ODE system
    def ode_f(y, t, ode_params):
        """
        Define ODE system
        Inputs:
            y: list of variables expressed as dy/dx. y is determined for the end of the time step
            t: array of time points of interest
            params: tuple of input values & model parameter values
        """
        
        # Unpack params. Params that vary by LU are series, with indices ['A','S','IG'],
        # LU-varying params: T_s,P_netInput,EPC0,Mland_i
        (P, E, Qq_i, Qr_US_i, Mland_i, Msus_US_i, TDPr_US_i, PPr_US_i,
         f_A, f_Ar, f_IG, f_S, f_NC_A, f_NC_Ar, f_NC_IG, f_NC_S, NC_type,
         f_IExcess, alpha, beta, T_s, T_g, fc, L_reach, S_reach, A_catch, a_Q, b_Q, E_Q, k_EQ,
         P_netInput, EPC0_A, EPC0_NC, Kf, Msoil, TDPeff, TDPg, E_PP, P_inactive) = ode_params
        
        # Unpack initial conditions for this time step
        # Hydrology
        VsA_i = y[0] # Agricultural soil water volume (mm)
        QsA_i = y[1] # Agricultural soil water flow (mm/day)
        VsS_i = y[2] # Semi-natural soil water volume (mm)
        QsS_i = y[3] # Semi-natural soil water flow (mm/day)
        Vg_i = y[4]  # Groundwater volume (mm)
        Qg_i = y[5]  # Groundwater discharge (mm/day)
        Vr_i = y[6]  # Reach volume (mm)
        Qr_i = y[7]  # Instantaneous reach discharge (mm/day)
        #(Qr_av_i would be y[8] here, but it's 0 at the start of every time step)
        # Sediment
        Msus_i = y[9]  # Mass of suspended sediment in the stream reach (kg)
        # (Msus_out_i would be y[10], but it's 0 at the start of every time step)
        # Phosphorus
        PlabA_i = y[11] # Mass of labile P in agricultural soil (kg)
        PlabNC_i = y[12] # Mass of labile P in newly converted land class (kg)
        TDPsA_i = y[13] # Mass of TDP in agricultural soil water (kg)
        TDPsNC_i = y[14] # Mass of TDP in newly converted land class (kg)
        TDPr_i = y[15] # Mass of total dissolved P in stream reach (kg)
        # (TDPr_out_i would be y[16], but it's 0 at the start of every time step)
        PPr_i = y[17]  # Mass of particulate P in stream reach (kg)
        # (PPr_out_i would be y[18], but it's 0 at the start of every time step)
        # Soil water vol & flow for newly converted land class
        if NC_type == 'A':  # If semi-natural converted to arable, assume has arable hydrol
            VsNC_i = VsA_i
            QsNC_i = QsA_i
        else:
            VsNC_i = VsS_i  # If arable converted to SN, assume has semi-natural hydrol
            QsNC_i = QsS_i
    
        # HYDROLOGY
        
        # Soil hydrology equations (units mm or mm/day): Agricultural land
        dQsA_dV = ((((VsA_i - fc)*np.exp(fc - VsA_i))/(T_s['A']*((np.exp(fc-VsA_i) + 1)**2)))
                    +(1/(T_s['A']*(np.exp(fc-VsA_i) + 1))))
        dVsA_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*VsA_i)) - QsA_i
        dQsA_dt = dQsA_dV*dVsA_dt
        
        # Soil hydrology equations (units mm or mm/day): Semi-natural/other land
        dQsS_dV = ((((VsS_i - fc)*np.exp(fc - VsS_i))/(T_s['S']*((np.exp(fc-VsS_i) + 1)**2)))
                    +(1/(T_s['S']*(np.exp(fc-VsS_i) + 1))))
        dVsS_dt = P*(1-f_IExcess) - alpha*E*(1 - np.exp(-0.02*VsS_i)) - QsS_i
        dQsS_dt = dQsS_dV*dVsS_dt
        
        # Groundwater equations (units mm or mm/day)
        dQg_dt = (beta*(f_A*QsA_i + f_S*QsS_i) - Qg_i)/T_g
        dVg_dt = beta*(f_A*QsA_i + f_S*QsS_i) - Qg_i
        
        # Instream equations (units mm or mm/day)
        dQr_dt = ((Qq_i + (1-beta)*(f_A*QsA_i + f_S*QsS_i) + Qg_i + Qr_US_i - Qr_i) # Fluxes (mm/d)
                  *a_Q*(Qr_i**b_Q)*(8.64*10**7)/((1-b_Q)*(L_reach*1000)))
                  # 2nd row is U/L=1/T. Units:(m/s)(s/d)(mm/m)(1/m)(m/mm)
        dVr_dt = Qq_i + (1-beta)*(f_A*QsA_i + f_S*QsS_i) + Qg_i + Qr_US_i - Qr_i
        dQr_av_dt = Qr_i  # Daily mean flow
        
        # SEDIMENT
        # Instream suspended sediment (kg; change in kg/day)
        dMsus_dt = (f_Ar*Mland_i['A'] + f_IG*Mland_i['IG'] + f_S* Mland_i['S']  # From land (kg/day)
                   + E_Q*S_reach*(Qr_i**k_EQ)             # Entrainment from the stream bed (kg/d)
                   + Msus_US_i                            # Inputs from upstream
                   - (Msus_i/Vr_i)*Qr_i)                  # Outflow from the reach;(kg/mm)*(mm/day)
        
        dMsus_out_dt = Qr_i*Msus_i/Vr_i  # Daily flux of SS
        
        # PHOSPHORUS
        
        # Agricultural soil labile P mass (kg). Assume semi-natural land has no labile soil P
        dPlabA_dt = Kf*Msoil*((TDPsA_i/VsA_i)-EPC0_A)  # Net sorption
        # dPlabA_dt = (PlabA_i/EPC0_A)*((TDPsA_i/VsA_i)-EPC0_A) # Alternative sorpn formulation
        
        # Newly-conveted soil labile P mass (kg)
        dPlabNC_dt = Kf*Msoil*((TDPsNC_i/VsNC_i)-EPC0_NC)
        
        # Change in dissolved P mass in agricultural soil water (kg/day)
        # Assume semi-natural land has no dissolved soil water P
        dTDPsA_dt = ((P_netInput['A']*100*A_catch/365)  # Net inputs (fert+manure-uptake) (kg/ha/yr)
                   - Kf*Msoil*((TDPsA_i/VsA_i)-EPC0_A)  # Net sorpn (kg/day) (could be alt above)
                   - (QsA_i*TDPsA_i/VsA_i)              # Outflow via soil water flow (kg/day)
                   - (Qq_i*TDPsA_i/VsA_i))              # Outflow via quick flow (kg/day)
        
        # And in newly converted land class soil water
        dTDPsNC_dt = ((P_netInput['NC']*100*A_catch/365)    # Net inputs (kg/ha/yr)
                   - Kf*Msoil*((TDPsNC_i/VsNC_i)-EPC0_NC)   # Net sorpn (kg/day)
                   - (QsNC_i*TDPsNC_i/VsNC_i)               # Outflow via soil water flow (kg/day)
                   - (Qq_i*TDPsNC_i/VsNC_i))                # Outflow via quick flow (kg/day)
        
        # Change in in-stream TDP mass (kg/d)
        # Semi-natural inputs not specified as assume 0 for soil water & quick flow
        dTDPr_dt = ((1-beta)*f_A*QsA_i*(TDPsA_i/VsA_i)  # Soil input, old agri. Units:(mm/d)(kg/mm)
                   + (1-beta)*f_NC_A*QsNC_i*(TDPsNC_i/VsNC_i)  # Soil input, new agri land
                   + (1-beta)*f_NC_S*QsNC_i*(TDPsNC_i/VsNC_i)  # Soil input, new SN land
                   + f_A*Qq_i*(TDPsA_i/VsA_i)           # Quick input, old agri. Units:(mm/d)(kg/mm)
                   + f_NC_A*Qq_i*(TDPsNC_i/VsNC_i)      # Quick input, newly-converted agri
                   + f_NC_S*Qq_i*(TDPsNC_i/VsNC_i)      # Quick inputs, newly-converted SN
                   + Qg_i*UC_Cinv(TDPg,A_catch)         # Groundwater input. Units: (mm/d)(kg/mm)
                   + TDPeff                             # Effluent input (kg/day)
                   + TDPr_US_i                          # Inputs from upstream 
                   - Qr_i*(TDPr_i/Vr_i))                # Reach outflow. Units: (mm/d)(kg/mm)
        
        dTDPr_out_dt = Qr_i*TDPr_i/Vr_i  # Daily TDP flux out of reach. Units: (mm/d)(kg/mm)=kg/d
        
        # Change in in-stream PP mass (kg/d)
        dPPr_dt = (E_PP *
                   (f_Ar*Mland_i['A']*(PlabA_i+P_inactive)/Msoil         # Old arable land
                   + f_IG*Mland_i['IG']*(PlabA_i+P_inactive)/Msoil       # Old improved grassland
                   + f_S*Mland_i['S']*P_inactive/Msoil)                  # Semi-natural land
                   + f_NC_Ar*Mland_i['A']*(PlabNC_i+P_inactive)/Msoil    # Newly-converted arable
                   + f_NC_IG*Mland_i['IG']*(PlabNC_i+P_inactive)/Msoil   # Newly-converted IG
                   + f_NC_S*Mland_i['S']*(PlabNC_i+P_inactive)/Msoil     # New semi-natural
                   + (E_Q*S_reach*(Qr_i**k_EQ))*E_PP*(f_A*(PlabA_i+P_inactive)+f_S*P_inactive)/Msoil
                                                                         # Entrained
                   + PPr_US_i                                            # Inputs from upstream 
                   - Qr_i*(PPr_i/Vr_i))                               # Reach outflow (mm/d)(kg/mm)
        
        dPPr_out_dt = Qr_i*PPr_i/Vr_i  # Daily mean flux
        
        # Add results of equations to an array
        res = np.array([dVsA_dt, dQsA_dt, dVsS_dt, dQsS_dt, dVg_dt, dQg_dt, dVr_dt, dQr_dt,
                        dQr_av_dt, dMsus_dt, dMsus_out_dt, dPlabA_dt, dPlabNC_dt, dTDPsA_dt,
                        dTDPsNC_dt, dTDPr_dt, dTDPr_out_dt, dPPr_dt, dPPr_out_dt])
        return res
    
    #########################################################################################
    # Define other useful functions
    def lin_interp(x, x0, x1, y0, y1):
        y = y0 + (y1-y0)*(x-x0)/(x1-x0)
        return y
    
    ##########################################################################################
    #-----------------------------------------------------------------------------------------
    # SETUP ADMIN FOR LOOPING OVER TIME STEPS AND SUB-CATCHMENTS
    
    # Dictionary to store results in for different sub-catchments
    df_TC_dict = {} # Key: Sub-catchment number (int); returns df of terrestrial compartment results
    df_R_dict = {}  # Key: Sub-catchment number (int); returns df of in-stream (reach) results
    
    # Time points to evaluate ODEs at (we're only interested in the start and end of each step)
    ti = [0, step_len]
    
    # Dictionary of parameters defining the dynamic crop cover-related erodibility (for the
    # M_land calculation)
    Erisk_dict = {}  # Key: 'spr' or 'aut'. Returns series with indices 'start','end','mid
    E_risk_period = 60.0
    for season in ['spr','aut']:
        d_dict = {'start':p['d_maxE_%s' %season]-E_risk_period/2.,
                  'end':p['d_maxE_%s' %season]+E_risk_period/2.,
                  'mid':p['d_maxE_%s' %season]}
        d_S = pd.Series(d_dict)
        Erisk_dict[season] = d_S
        
    #-----------------------------------------------------------------------------------------
    # INITIAL CONDITIONS AND DERIVED PARAMETERS THAT DON'T VARY BY SUB-CATCHMENT
    # Unpack user-supplied initial conditions, calculate any others, convert units
    
    # 1) Terrestrial - constant over LU and SC
    # Hydrol
    VsA0 = p['fc']   # Initial soil volume (mm). Assume it's at field capacity.
    VsS0 = VsA0      # Initial soil vol, semi-natural land (mm). Assumed same as agricultural!!
    Qg0 = p['Qg0_init']      # Initial groundwater flow (mm/d)
    
    #-----------------------------------------------------------------------------------------
    # START LOOP OVER SUB-CATCHMENTS
    for SC in p['SC_list']:
        
        # INITIAL CONDITIONS THAT VARY BY SUB-CATCHMENT
    
        # Soil mass and inactive soil P (kg)
        # Assume inactive soil P is equivalent to semi-natural total soil P for all LU classes
        Msoil = p['Msoil_m2']*10**6*p_SC.ix['A_catch',SC] # Soil mass (kg): (kgSoil/m2)(m2/km2)km2
        P_inactive = 10**-6*p_LU['S']['SoilPconc']*Msoil

        # 2) Terrestrial - varying by land use and by sub-catchment
        for LU in ['A','S']:
            # Convert units of EPC0 from mg/l to kg/mm to give initial EPC0
            p_LU.ix['EPC0_0',LU] = UC_Cinv(p_LU[LU]['EPC0_init_mgl'], p_SC.ix['A_catch',SC])
            ### CHECK units here! Should same mg/l conc give diff kg/mm conc in diff SCs?????
            
            # Initial labile P. Units: (kgP/mgP)(mgP/kgSoil)kgSoil. Assume Plab0=0 for semi-natural
            p_LU.ix['Plab0',LU] = 10**-6*(p_LU[LU]['SoilPconc']-p_LU['S']['SoilPconc']) * Msoil
            # Initial soil water TDP mass (kg); Units: (kg/mm)*mm
            if LU == 'A':
                p_LU.ix['TDPs0',LU] = p_LU[LU]['EPC0_0']*VsA0
            else:
                p_LU.ix['TDPs0',LU] = 0
                        
        # Set initial agricultural labile P and soil TDP masses as variables to be updated during
        # looping (assume semi-natural remain at 0 for both)
        Plab0_A, TDPs0_A = p_LU.ix['Plab0','A'], p_LU.ix['TDPs0','A']
        # Initial labile P and soil TDP mass on newly converted land use class
        if p_SC.ix['NC_type',SC]=='S':
            Plab0_NC = Plab0_A  # New class is SN, from arable, therefore start with arable labile P
            TDPs0_NC = TDPs0_A
        else:
            Plab0_NC = 0.0      # New class is arable, from SN, therefore start with no labile P
            TDPs0_NC = p_LU.ix['TDPs0','S']

        # Set the value for Kf, the adsorption coefficient (mm/kg soil)
        if run_mode == 'cal': # If the calibration period, calculate.
            # Assume SN has EPC0=0, PlabConc =0. Units: (kg/mg)(mg/kgSoil)(mm/kg)
            Kf = 10**-6*(p_LU['A']['SoilPconc']-p_LU['S']['SoilPconc'])/p_LU['A']['EPC0_0']  
        else:  # If not the calibration period, read Kf in from the series of param values
            Kf = p['Kf']

        # 3) In-stream initial conditions (all reaches the same for chem)
        TDPr0, PPr0, Msus0 = 0.0, 0.0, 0.0  # Initial TDP, PP & SS masses (kg; assume all equal 0.0)
        # Initial in-stream flow
        if SC == 1:
            # Convert units of initial reach Q from m3/s to mm/day
            Qr0 = UC_Qinv(p['Qr0_init'], p_SC.ix['A_catch',SC])
        else:        # Outflow from the reach upstream from the first time step
            Qr0 = df_R_dict[SC-1].ix[0,'Qr']  # N.B. 'Qr' here is Qr_av, the daily mean flow
        
        # ADMIN
    
        # Dictionary of slopes for different land use classes in the sub-catchment:
        slope_dict = {'A':p_SC.ix['S_Ar',SC], 'IG':p_SC.ix['S_IG',SC], 'S':p_SC.ix['S_SN',SC]}
        
        # Lists to store output
        output_ODEs = []    # From ode_f function
        output_nonODE = []  # Will include: Qq, Mland, Qr_US (latter for checking only)        
        
        #-------------------------------------------------------------------------------------
        # START LOOP OVER MET DATA
        for idx in range(len(met_df)):

            # Get precipitation and evapotranspiration for this day
            P = met_df.ix[idx, 'P']
            E = met_df.ix[idx, 'PET']

            # Calculate infiltration excess (mm/(day * catchment area))
            Qq_i = p['f_IExcess']*P
            
            # Inputs to reach from up-stream reaches
            if SC == 1:
                # For the top reach, set the input from upstream reaches to 0
                Qr_US_i, Msus_US_i, TDPr_US_i, PPr_US_i = 0.0, 0.0, 0.0, 0.0
            else:
                # Below reach 1, the upstream input is the daily mean flux from up-stream for the
                # current day
                Qr_US_i = df_R_dict[SC-1].ix[idx,'Qr']
                Msus_US_i = df_R_dict [SC-1].ix[idx, 'Msus']
                TDPr_US_i = df_R_dict [SC-1].ix[idx, 'TDPr']
                PPr_US_i = df_R_dict [SC-1].ix[idx, 'PPr']

            # Calculate delivery of sediment from the land to the stream (kg/day). NB this flux
            # assumes the land use covers the whole catchment. Divide by area to get areal flux
            Mland_i = pd.Series(3*[np.NaN],['A','S','IG']) # Empty series to store results in later
            dayNo = met_df.index[idx].dayofyear  # Day of the year (1 to 365)
            for LU in ['A','S','IG']:
                if LU == 'A':
                    # If arable land, work out a dynamic crop cover factor, to account for the variation
                    # in erodibility through the year due to harvesting and planting practices.
                    if dynamic_dict['Dynamic_erodibility'] == 'y':
                        
                        # Using a sine wave to simulate the annual change in erodibility
#                         C_spr_t = p_LU[LU]['C_cover']*(np.cos((2*np.pi/365)*(dayNo-p['d_maxE_spr']))+1)
#                         C_aut_t = p_LU[LU]['C_cover']*(np.cos((2*np.pi/365)*(dayNo-p['d_maxE_aut']))+1)
#                         C_cover = (p_SC.ix['f_spr',SC]*C_spr_t +(1-p_SC.ix['f_spr',SC])*C_aut_t)
                        
                        # Using a triangular wave
                        C_cov_dict = {} # Dict for storing spring & autumn results in
                        for s in ['spr','aut']:  # Loop through seasons
                            d = Erisk_dict # Defined above; dict defining high erosion risk period
                            d_start, d_end, d_mid = d[s]['start'], d[s]['end'], d[s]['mid']
                            if dayNo in np.arange(d_start, d_end):
                                if dayNo < d_mid: # If within high risk period, before mid-point
                                    C_season = lin_interp(dayNo, x0=d_start, x1=d_mid,
                                                     y0=p_LU[LU]['C_cover'], y1=1.0)
                                else: # If in high risk period, after mid-point
                                    C_season = lin_interp(dayNo, x0=d_mid, x1=d_end,
                                                     y0=1.0, y1=p_LU[LU]['C_cover'])
                            else: # Otherwise, outside high risk period
                                C_season = (p_LU[LU]['C_cover']-(E_risk_period*(1-p_LU[LU]['C_cover'])
                                           /(2*(365-E_risk_period))))
                            C_cov_dict[s] = C_season
                        # Average the dynamic factor over spring and autumn-sown crops
                        C_cover = (p_SC.ix['f_spr',SC]*C_cov_dict['spr']
                                   + (1-p_SC.ix['f_spr',SC])*C_cov_dict['aut'])  
                        
                    else:  # If not calculating a dynamic crop cover, then just assign user parameter
                        C_cover = p_LU[LU]['C_cover']
                        
                    C_cover_A = C_cover  # Store this for arable land, for checking
                
                else:  # For non-arable LU, the cover factor is always constant throughout the year
                    C_cover = p_LU[LU]['C_cover']
                
                # Sediment yield per land use class (kg/d). See documentation for rationale/source
                Mland_i[LU] = 10**5*(p['E_land']*(Qq_i**p['k_Eland'])
                              *slope_dict[LU]
                              *p_SC.ix['K_erosion',SC]
                              *C_cover
                              *(1-p_LU[LU]['Sed_measures']))

            # Calculate dynamic EPC0 as a function of labile P mass
            if dynamic_dict['Dynamic_EPC0'] == 'y':
                EPC0_A_i = Plab0_A/(Kf*Msoil) # Agricultural EPC0; equals EPC0_0 on the 1st timestep
                EPC0_NC_i = Plab0_NC/(Kf*Msoil) # EPC0 on newly-converted land
            # Or, have a constant EPC0 throughout the model run
            else:
                EPC0_A_i = p_LU['A']['EPC0_0']
                if p_SC.ix['NC_type',SC] == 'S': # (little point in a new class with constant EPC0)
                    EPC0_NC_i = p_LU['A']['EPC0_0']  # New semi-natural land has agricultural EPC0
                else:
                    EPC0_NC_i = p_LU['S']['EPC0_0']  # New agricultural has SN EPC0

            # Append to non-ODE results
            output_nonODE_i = [Qq_i, Mland_i['A'],Mland_i['IG'],Mland_i['S'], EPC0_A_i, EPC0_NC_i,C_cover_A ]
            output_nonODE.append(output_nonODE_i)

            # Calculate additional initial conditions from user-input values/ODE solver output
            # Soil flow, agricultural (mm/d)
            QsA0 = (VsA0 - p['fc'])/(p_LU['A']['T_s']*(1 + np.exp(p['fc'] - VsA0)))
            # Soil flow, semi-natural (mm/d)
            QsS0 = (VsS0 - p['fc'])/(p_LU['S']['T_s']*(1 + np.exp(p['fc'] - VsS0)))
            Vg0 = Qg0 *p['T_g']     # Groundwater vol (mm)
            Tr0 = ((1000*p_SC.ix['L_reach',SC])/
                   (p['a_Q']*(Qr0**p['b_Q'])*(8.64*10**7))) # Reach time constant (days); T=L/aQ^b
            Vr0 = Qr0*Tr0 # Reach volume (V=QT) (mm)

            # Vector of initial conditions for start of time step (assume 0 for initial Qr_av,
            # Msus_out, TDPr_out and PPr_out, the mass or vol of water lost per time step
            y0 = [VsA0, QsA0, VsS0, QsS0, Vg0, Qg0, Vr0, Qr0, 0.0, Msus0, 0.0, Plab0_A, Plab0_NC,
                  TDPs0_A, TDPs0_NC, TDPr0, 0.0, PPr0, 0.0]

            # Today's rainfall, ET & model parameters for input to solver. NB the order must be the
            # same as the order in which they are unpacked within the odeint function
            ode_params = [P, E, Qq_i, Qr_US_i, Mland_i, Msus_US_i, TDPr_US_i, PPr_US_i,
                          p_SC.ix['f_A',SC],p_SC.ix['f_Ar',SC],p_SC.ix['f_IG',SC],p_SC.ix['f_S',SC],
                          p_SC.ix['f_NC_A',SC],p_SC.ix['f_NC_Ar',SC],p_SC.ix['f_NC_IG',SC],
                          p_SC.ix['f_NC_S',SC],p_SC.ix['NC_type',SC],
                          p['f_IExcess'],p['alpha'], p['beta'],
                          p_LU.ix['T_s'], p['T_g'], p['fc'],
                          p_SC.ix['L_reach',SC], p_SC.ix['S_reach',SC], p_SC.ix['A_catch',SC],
                          p['a_Q'], p['b_Q'], p['E_Q'], p['k_EQ'],
                          p_LU.ix['P_netInput'], EPC0_A_i, EPC0_NC_i, Kf, Msoil,
                          p_SC.ix['TDPeff',SC], p['TDPg'], p['E_PP'], P_inactive]

            # Solve ODEs
            # N.B. rtol is part of the error tolerance. Default is ~1e-8, but reducing it removes the
            # risk of the solver exiting before finding a solution due to reaching max number of steps
            # (in which case can get odd output). Also speeds up runtime.
            # This issue can also be circumvented by increasing the max. number of steps to 5000.
            y, output_dict = odeint(ode_f, y0, ti, args=(ode_params,),full_output=1, rtol=1.5e-3,
                                    mxstep=5000)

            # Extract values for the end of the step
            res = y[1]
            res[res<0] = 0 # Numerical errors may result in very tiny values <0; set these back to 0
            output_ODEs.append(res)

            # Update initial conditions for next step (for Vs, Qg0, Qr0, Msus0, Plab0, TDPs0)
            VsA0 = res[0]  # QsA0 would be res[1]
            VsS0 = res[2]  # QsS0 would be res[3], Vg0 would be res[4] (but calculate from Qg0)
            # Re-set groundwater to user-supplied minimum flow at start of each time step. Non-ideal
            # solution to the problem or maintaining stream flow during baseflow conditions!!!
            if p['Qg_min'] > res[5]:
                Qg0 = p['Qg_min']
            else:
                Qg0 = res[5]
            # Vr0 would be res[6], but calculate from Qr0 instead
            Qr0 = res[7]       # Qr_av_0 would be res[8], but it's always 0
            Msus0 = res[9]     # Msus_out_0 would be res[10]
            Plab0_A = res[11]
            Plab0_NC = res[12]
            TDPs0_A = res[13]
            TDPs0_NC = res[14]
            TDPr0 = res[15]    # TDPr_out_0 would be res[16], but it's always 0
            PPr0 = res[17]     # TDPr_out_0 would be res[18], but it's always 0

            # END LOOP OVER MET DATA
        #-------------------------------------------------------------------------------------
    
        # Build a dataframe of ODE results
        df_ODE = pd.DataFrame(data=np.vstack(output_ODEs),
                           columns=['VsA', 'QsA','VsS', 'QsS', 'Vg', 'Qg', 'Vr', 'Qr_instant', 'Qr',
                           'Msus_instant', 'Msus', 'P_labile_A', 'P_labile_NC', 'TDPs_A', 'TDPs_NC',
                           'TDPr_instant', 'TDPr', 'PPr_instant', 'PPr'], index=met_df.index)

        # Dataframe of non ODE results
        df_nonODE = pd.DataFrame(data=np.vstack(output_nonODE),
                                 columns=['Qq','Mland_A','Mland_IG','Mland_S',
                                'EPC0_A_kgmm', 'EPC0_NC_kgmm', 'C_cover_A'], index=met_df.index)

        # Rearrange ODE and non-ODE result dataframes into results dataframes for the terrestrial
        # compartment (df_TC) and the stream reach (df_R)
        df_TC = pd.concat([df_ODE[['VsA', 'QsA','VsS', 'QsS', 'Vg', 'Qg', 'P_labile_A',
                                  'P_labile_NC','TDPs_A', 'TDPs_NC']],df_nonODE], axis=1)
        # NB masses of SS and P are all total fluxes for the day, and Q is the daily mean flow
        df_R = df_ODE[['Vr', 'Qr','Msus','TDPr', 'PPr']]
        
        # Add to dictionaries; key = sub-catchment/reach number
        df_TC_dict[SC] = df_TC
        df_R_dict[SC] = df_R
        
        # END LOOP OVER SUB-CATCHMENTS
        #-------------------------------------------------------------------------------------

    return (df_TC_dict, df_R_dict, Kf, output_dict)  # NB Kf is returned for the last SC

