"""
File containing main model definitions
"""

# ################################################################################
# Import modules
import os
import csv
import numpy as np
import pandas as pd
from scipy.integrate import odeint
# from scipy import optimize
##from scipy.stats import norm

# Import local model modules
from inputs import snow_hydrol_inputs, daily_PET
import helper_functions as hf

# ######################################################################################

def ode_f(y, t, ode_params):
    """ Define ODE system.

    Args:
        y:      List of variables expressed as dy/dx. y is determined for the end of the time step
        t:      Array of time points of interest
        params: Tuple of input values & model parameter values

    Returns:
        Array of gradients for end of time step

            [dVsA_dt, dQsA_dt, dVsS_dt, dQsS_dt, dVg_dt, dQg_dt, dVr_dt, dQr_dt, dQr_av_dt, dMsus_dt,
             dMsus_out_dt, dPlabA_dt, dPlabNC_dt, dTDPsA_dt, dTDPsNC_dt, dTDPr_dt, dTDPr_out_dt,
             dPPr_dt, dPPr_out_dt]
    """    
    # Unpack params. Params that vary by LU are series, with indices ['A','S','IG'],
    # LU-varying params: T_s,P_netInput,EPC0,Esus_i
    (P, E, mu, Qq_i, Qr_US_i, Esus_i, Msus_US_i, TDPr_US_i, PPr_US_i,
     f_A, f_Ar, f_IG, f_S, f_NC_A, f_NC_Ar, f_NC_IG, f_NC_S, NC_type,
     f_quick, alpha, beta, T_s, T_g, fc, L_reach, A_catch,
     a_Q, b_Q, E_M, k_M, P_netInput, EPC0_A, EPC0_NC, Kf, Msoil, TDPeff,
     TDPg, E_PP, P_inactive, dynamic_EPC0) = ode_params
    
    # -----------------------------------------------------------------------------
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
    PlabA_i = y[11]  # Mass of labile P in agricultural soil (kg)
    PlabNC_i = y[12] # Mass of labile P in newly converted land class (kg)
    TDPsA_i = y[13]  # Mass of TDP in agricultural soil water (kg)
    TDPsNC_i = y[14] # Mass of TDP in newly converted land class (kg)
    TDPr_i = y[15]   # Mass of total dissolved P in stream reach (kg)
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
    
    # Inputs of sediment to the stream. This is a series, one value per LU
    Msus_in_i = Esus_i * Qr_i**k_M
    
    # -----------------------------------------------------------------------------
    # HYDROLOGY EQUATIONS
        
    # Soil hydrology (units mm or mm/day): Agricultural land
    dQsA_dV = ((((VsA_i - fc)*np.exp(fc - VsA_i))/(T_s['A']*((np.exp(fc-VsA_i) + 1)**2)))
                +(1/(T_s['A']*(np.exp(fc-VsA_i) + 1))))
    dVsA_dt = P*(1-f_quick) - alpha*E*(1 - np.exp(-mu*VsA_i)) - QsA_i  # mu a function of fc
    dQsA_dt = dQsA_dV*dVsA_dt
        
    # Soil hydrology (units mm or mm/day): Semi-natural/other land
    dQsS_dV = ((((VsS_i - fc)*np.exp(fc - VsS_i))/(T_s['S']*((np.exp(fc-VsS_i) + 1)**2)))
                +(1/(T_s['S']*(np.exp(fc-VsS_i) + 1))))
    dVsS_dt = P*(1-f_quick) - alpha*E*(1 - np.exp(-mu*VsS_i)) - QsS_i
    dQsS_dt = dQsS_dV*dVsS_dt
        
    # Groundwater (units mm or mm/day)
    dQg_dt = (beta*((f_A+f_NC_A)*QsA_i + (f_S+f_NC_S)*QsS_i) - Qg_i)/T_g
    dVg_dt = beta*((f_A+f_NC_A)*QsA_i + (f_S+f_NC_S)*QsS_i) - Qg_i
    
    # Instream (units mm or mm/day)
    dQr_dt = ((Qq_i                                 # Quick flow
               + (1-beta)*((f_A+f_NC_A)*QsA_i       # Soil water in: Arable land
               + (f_S+f_NC_S)*QsS_i)                # Soil water in: Semi-natural land
               + Qg_i + Qr_US_i - Qr_i)             # Groundwater in (mm/d)
              *a_Q*(Qr_i**b_Q)*(8.64*10**4)/((1-b_Q)*(L_reach))) # Outflow. U/L=1/T. Units:(m/s)(s/d)(1/m)
    dVr_dt = (Qq_i + (1-beta)*((f_A+f_NC_A)*QsA_i + (f_S+f_NC_S)*QsS_i)+ Qg_i + Qr_US_i - Qr_i)
    dQr_av_dt = Qr_i  # Daily mean flow
    
    # -----------------------------------------------------------------------------
    # SEDIMENT EQUATIONS
    # Instream suspended sediment (kg; change in kg/day)
    dMsus_dt = ((f_Ar+f_NC_Ar)*Msus_in_i['A']
                + (f_IG+f_NC_IG)*Msus_in_i['IG']
                + (f_S+f_NC_S)*Msus_in_i['S']      # Terrestrial inputs (kg/day)
                + Msus_US_i                       # Inputs from upstream
                - (Msus_i/Vr_i)*Qr_i)             # Outflow from the reach;(kg/mm)*(mm/day)     

    dMsus_out_dt = (Msus_i/Vr_i)*Qr_i  # Daily flux of SS
    
    # -----------------------------------------------------------------------------
    # PHOSPHORUS EQUATIONS
          
    # Change in dissolved P mass in agricultural soil water (kg/day)
    # Assume semi-natural land has no dissolved soil water P
    # Only simulate a change if the user has decided to simulate a dynamic EPC0
    
    if dynamic_EPC0 == 'y':
        
        # Agricultural soil labile P mass (kg). Assume semi-natural land has no labile soil P
        # Adding the logic in to check for no soil water REALLY speeds up the model, though doesn't
        # seem to affect the output like I would expect it would (by stopping infs)
        if VsA_i > 0:
            dPlabA_dt = Kf*Msoil*((TDPsA_i/VsA_i)-EPC0_A)  # Net sorption
        else:
            dPlabA_dt = 0.

        # Newly-conveted soil labile P mass (kg)
        if VsNC_i > 0:
            dPlabNC_dt = Kf*Msoil*((TDPsNC_i/VsNC_i)-EPC0_NC)
        else:
            dPlabNC_dt = 0.
        
        dTDPsA_dt = ((P_netInput['A']*100*A_catch/365)    # Net inputs (fert+manure-uptake) (kg/ha/yr)
                     - dPlabA_dt                          # Net sorpn (kg/day)
                     - QsA_i*TDPsA_i/VsA_i  # Soil water outflow (kg/day)
                     - Qq_i*TDPsA_i/VsA_i)   # Quick flow out (kg/day)

        # And in newly converted land class soil water
        dTDPsNC_dt = ((P_netInput['NC']*100*A_catch/365)     # Net inputs (kg/ha/yr)
                      - dPlabNC_dt                           # Net sorpn (kg/day)
                      - QsNC_i*TDPsNC_i/VsNC_i               # Outflow via soil water flow (kg/day)
                      - Qq_i*TDPsNC_i/VsNC_i)                # Outflow via quick flow (kg/day)
        
        # If dynamic, agricultural soil water TDP concentration is calculated
        agri_soilwater_P_conc = TDPsA_i/VsA_i
        NC_soilwater_P_conc = TDPsNC_i/VsNC_i
    
    # If no dynamic EPC0, soil labile P and soil water TDP are constant over time
    else:
        dPlabA_dt = 0.
        dPlabNC_dt = 0.
        dTDPsA_dt = 0.
        dTDPsNC_dt = 0.
        agri_soilwater_P_conc = EPC0_A # Soil water TDP concentration constant at EPC0
        NC_soilwater_P_conc = EPC0_NC
    
    # Change in in-stream TDP mass (kg/d)
    # Semi-natural inputs not specified as assume 0 for soil water & quick flow
    dTDPr_dt = ((1-beta)*f_A*QsA_i*(agri_soilwater_P_conc)  # Soilwater, old agri. (mm/d)(kg/mm)
                + (1-beta)*f_NC_A*QsNC_i*(NC_soilwater_P_conc)  # Soil input, new agri land
                + (1-beta)*f_NC_S*QsNC_i*(NC_soilwater_P_conc)  # Soil input, new SN land
                + f_A*Qq_i*(agri_soilwater_P_conc)          # Quick input, old agri. Units:(mm/d)(kg/mm)
                + f_NC_A*Qq_i*(NC_soilwater_P_conc)             # Quick input, newly-converted agri
                + f_NC_S*Qq_i*(NC_soilwater_P_conc)             # Quick inputs, newly-converted SN
                + Qg_i * hf.UC_Cinv(TDPg,A_catch)           # Groundwater input. Units: (mm/d)(kg/mm)
                + TDPeff                                    # Effluent input (kg/day)
                + TDPr_US_i                                 # Inputs from upstream 
                - Qr_i*(TDPr_i/Vr_i))                       # Reach outflow. Units: (mm/d)(kg/mm)

    dTDPr_out_dt = Qr_i*(TDPr_i/Vr_i)  # Daily TDP flux out of reach. Units: (mm/d)(kg/mm)=kg/d
        
    # Change in in-stream PP mass (kg/d)
    dPPr_dt = (E_PP *(f_Ar*Msus_in_i['A']*(PlabA_i+P_inactive)/Msoil       # Old arable land
                      + f_IG*Msus_in_i['IG']*(PlabA_i+P_inactive)/Msoil    # Old improved grassland
                      + f_S*Msus_in_i['S']*P_inactive/Msoil                # Semi-natural land
                      + f_NC_Ar*Msus_in_i['A']*(PlabNC_i+P_inactive)/Msoil  # Newly-converted arable
                      + f_NC_IG*Msus_in_i['IG']*(PlabNC_i+P_inactive)/Msoil # Newly-converted IG
                      + f_NC_S*Msus_in_i['S']*(PlabNC_i+P_inactive)/Msoil)  # New semi-natural
              + PPr_US_i                                                    # Inputs from upstream 
              - Qr_i*(PPr_i/Vr_i))                                          # Reach outflow (mm/d)(kg/mm)                           
        
    dPPr_out_dt = Qr_i*PPr_i/Vr_i  # Daily mean flux
    
    # -----------------------------------------------------------------------------
    # Add results of equations to an array
    gradients = np.array([dVsA_dt, dQsA_dt, dVsS_dt, dQsS_dt, dVg_dt, dQg_dt, dVr_dt, dQr_dt,
                    dQr_av_dt, dMsus_dt, dMsus_out_dt, dPlabA_dt, dPlabNC_dt, dTDPsA_dt,
                    dTDPsNC_dt, dTDPr_dt, dTDPr_out_dt, dPPr_dt, dPPr_out_dt])
    
    return gradients

# #################################################################################################

# Main model function (calls ODE function)
    
def run_simply_p(met_df, p_struc, p_SU, p_LU, p_SC, p, dynamic_options, step_len=1.):
    """ Simple hydrology, sediment and phosphorus model, with processes varying by land use, sub-catchment
        and reach.

        Version 0.1, alpha release. Comparison to previous version:

            Correction to Actual ET calculation (previously mu constant at 0.02, now varies with FC)

    Args:
        met_df          Dataframe. Must contain columns 'P', the precipitation+snowmelt input to
                        the soil box and either 'PET' (potential evapotranspiration, units mm/day)
                        or 'T_air' (air temperature, in which case PET will be estimated)
        
        p_struc         Dataframe. Parameters defining the spatial arrangement of the sub-catchments/
                        reaches relative to one another.
                        Index: reach number (integer)
                        Columns: 'Upstream_SCs': list of reaches directly upstream of current reach)
                                 'In_final_flux?': Whether to include reach in flux calculation to a
                                                   receiving waterbody. If all 0s, no sum calculated

        p               Series. Parameter values which don't vary by land use (index: param name)

        p_SU            Series. Setup parameters
        
        p_LU            Dataframe. Parameter values which do vary by land use
                        Index:   parameter name
                        Columns: 'A', 'S', 'IG', 'NC' (agricultural, semi-natural, improved grassland,
                                  newly-converted agricultural or semi-natural land)
                                 IG only has values for parameter 'E_land' (soil erodibility), in
                                 which case value in 'A' column is for arable land
        
        p_SC            Dataframe. Parameter values which vary by sub-catchment or reach
                        Index:   parameter name
                        Columns: 1, 2, ...n sub-catchments

        dynamic_options Series of options controlling whether inputs/variables are calculated
                        dynamically or kept constant.
                        Row indices: 'Dynamic_EPC0', 'Dynamic_erodibility', 'Dynamic_effluent_inputs',
                                     'Dynamic_terrestrialP_inputs'. 'y' or 'n'
                        **NB only the first 2 are implemented in the model at the moment**

        step_len        Length of each step in the input dataset (days). Default=1

    Returns:
        4-element tuple, with the following elements:
    
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
            P_labile_A:   Labile soil P mass in agricultural soil (kg)
            P_labile_NC:  Labile soil P mass in newly-converted agricultural or semi-natural land (kg)
            EPC0_A_kgmm:  EPC0 in agricultural soil (kg/mm)
            EPC0_NC_kgmm: EPC0 in newly-converted agricultural or semi-natural soil (kg/mm)       
            TDPs_A:       Soil water TDP mass, agricultural land (kg)
            TDPs_NC:      Soil water TDP mass, newly-converted agricultural or semi-natural land (kg)

        2)  df_R_dict: A dictionary containing dataframes of results for the stream reach, with one
            dataframe per sub-catchment (dictionary key is the sub-catchment number as an integer).
            Dataframe column headings:
            
            Model output:
            Vr:   Reach volume (mm)
            Qr:   Mean daily discharge from the reach (mm/d)
            Msus: Daily flux of suspended sediment from the reach (kg/day)
            TDPr: Daily flux of TDP from the reach (kg/day)
            PPr:  Daily flux of PP from the reach (kg/day)
            
            Post-processing:
            'Sim_Q_cumecs': Discharge (m3/s)
            Concentrations in mg/l:
                'PP_mgl'
                'SRP_mgl'
                'SS_mgl'
                'TDP_mgl'
                'TP_mgl': Sum of TDP and PP (mg/l)
            
            Note that instantaneous fluxes of in-stream Q and masses of SS, PP and TDP are also
            calculated by the model, and could be saved to this output dataframe if there was a need,
            rather than these daily fluxes. However, daily fluxes are used for calculating mean
            volume-weighted concentrations.

        3)  Kf: the soil adsorption coefficient (units mm/kg Soil). From (mgP/kgSoil)(mm/kgP).
            Multiply by A_catch*10^6 for (l/kgSoil).
        
        4)  output_dict: Full output from the ODE solver, providing technical details of solving the
            ODE system.
    """    
    
    ##########################################################################################
    # ---------------------------------------------------------------------------------------
    # CALCULATE PARAMETERS DERIVED FROM USER INPUT
    # All params: index = param names, values in cols
    
    # LAND USE PARAMS
    # Add empty rows to land use param dataframe, to be populated later in model
    # (4 classes: 'A','S','NC','IG')
    p_LU.loc['EPC0_0',:] = 4*[np.NaN]  # Initial EPC0 (kg/mm)
    p_LU.loc['Plab0',:] = 4*[np.NaN]   # Initial labile P (kg)
    p_LU.loc['TDPs0',:] = 4*[np.NaN]   # Initial soil water TDP mass (kg)

    # SUB-CATCHMENT/REACH PARAMS
    # Calculate fraction of total area as intensive agricultural land
    p_SC.loc['f_A'] = p_SC.loc['f_IG']+p_SC.loc['f_Ar']
    p_SC.loc['f_NC_A'] = p_SC.loc['f_NC_Ar'] + p_SC.loc['f_NC_IG']
    # Check that land use proportions add to 1 in all sub-catchments; raise an error if not
    for SC in p['SC_list']:
        if (p_SC.loc['f_A',SC]+p_SC.loc['f_S',SC]
            +p_SC.loc['f_NC_A',SC]+p_SC.loc['f_NC_S',SC]) != 1:
            raise ValueError('Land use proportions do not add to 1 in SC %s' % SC)
        # Determine whether newly-converted land is agri or SN (raise error if both)
        if p_SC.loc['f_NC_A',SC] > 0:
            if p_SC.loc['f_NC_S',SC] > 0: 
                raise ValueError("SC %s has 2 kinds of newly-converted land;\n\
                only one permitted (SN or agri)" % SC)
            else:
                NC_type = 'A'
        elif p_SC.loc['f_NC_S',SC]>0:
            NC_type = 'S'
        else:
            NC_type = 'None'
        p_SC.loc['NC_type',SC] = NC_type
    
    # ---------------------------------------------------------------------------------------
    # SETUP ADMIN FOR LOOPING OVER TIME STEPS AND SUB-CATCHMENTS
    
    # Dictionary to store results for different sub-catchments
    df_TC_dict = {} # Key: SC number (int); returns df of terrestrial compartment results
    df_R_dict = {}  # Key: SC number (int); returns df of in-stream (reach) results
    
    # Time points to evaluate ODEs at (we're only interested in the start and end of each step)
    ti = [0, step_len]
    
    # Calculate the value of the shape parameter mu, in the exponential relating actual and
    # potential ET
    mu = -np.log(0.01)/p['fc']
     
    # Dictionary of parameters defining the dynamic crop cover-related erodibility (for the
    # M_land calculation)
    Erisk_dict = {}  # Key: 'spr' or 'aut'. Returns series with indices 'start','end','mid
    E_risk_period = 60.0
    for season in ['spr','aut']:
        # Check pars are valid to avoid Julian days <0 or >365
        assert (30 < p['d_maxE_%s' %season] < 335), "'d_maxE_%s' must be between 30 and 335" % season
        d_dict = {'start':p['d_maxE_%s' %season]-E_risk_period/2.,
                  'end':p['d_maxE_%s' %season]+E_risk_period/2.,
                  'mid':p['d_maxE_%s' %season]}
        Erisk_dict[season] = pd.Series(d_dict)
        
    #################################################################################################
    # START LOOP OVER SUB-CATCHMENTS
    for SC in p['SC_list']:
        
        print ('Starting model run for sub-catchment: %s' %SC)
        
        #-----------------------------------------------------------------------------------------
        # Unpack user-supplied initial conditions, calculate any others, convert units
        
        # INITIAL CONDITIONS THAT DON'T VARY BY SUB-CATCHMENT
        # Note: could have this section outside the sub-catchment loop, but then would need
        # to change variable names

        # 1) Hydrology - constant over LU and SC

        # Soil water volume (agricultural and semi-natural; mm)
        VsA0 = p['fc']   # Initial soil volume (mm). Assume it's at field capacity.
        VsS0 = VsA0      # Initial soil vol, semi-natural land (mm). Assumed same as agricultural!!

        # Soilwater flow (agricultural and semi-natural; mm/day)
        QsA0 = (VsA0 - p['fc'])/(p_LU['A']['T_s']*(1 + np.exp(p['fc'] - VsA0)))
        QsS0 = (VsS0 - p['fc'])/(p_LU['S']['T_s']*(1 + np.exp(p['fc'] - VsS0)))

        # Initial in-stream flow. This is provided for a single reach and converted to units of mm/day.
        # Once in units of mm/day, can assume the same initial discharge for every reach.
        Qr0 = hf.UC_Qinv(p['Qr0_init'], p_SC.loc['A_catch',p['SC_Qr0']])

        # Groundwater flow and volume (mm/d, mm). Assume equal to BFI*initial flow
        Qg0 = p['beta'] * Qr0
        Vg0 = Qg0 * p['T_g']

        # 2) In-stream masses

        # Initial TDP, PP & SS masses (kg). Assume all equal 0.0 and have warm-up period. Only a short
        # warm-up period needed for most systems. Could be altered though.
        TDPr0, PPr0, Msus0 = 0.0, 0.0, 0.0
        
        # ---------------------------------------------------------------------------------------
        # INITIAL CONDITIONS AND VARIABLES THAT VARY BY SUB-CATCHMENT
    
        # Soil mass and inactive soil P (kg)
        # Assume inactive soil P is equivalent to semi-natural total soil P for all LU classes
        Msoil = p['Msoil_m2']*10**6*p_SC.loc['A_catch',SC] # Soil mass (kg): (kgSoil/m2)(m2/km2)km2
        P_inactive = 10**-6*p_LU['S']['SoilPconc']*Msoil

        # 3) Terrestrial - varying by land use and by sub-catchment
        for LU in ['A','S']:
            
            # Initial EPC0: Convert units of EPC0 from mg/l to kg/mm
            p_LU.loc['EPC0_0',LU] = hf.UC_Cinv(p_LU[LU]['EPC0_init_mgl'], p_SC.loc['A_catch',SC])
            
            # Initial labile P. Units: (kgP/mgP)(mgP/kgSoil)kgSoil. Assume Plab0=0 for semi-natural
            p_LU.loc['Plab0',LU] = 10**-6*(p_LU[LU]['SoilPconc']-p_LU['S']['SoilPconc']) * Msoil
            
            # Initial soil water TDP mass (kg); Units: (kg/mm)*mm
            if LU == 'A':
                p_LU.loc['TDPs0',LU] = p_LU[LU]['EPC0_0']*VsA0
            else:
                p_LU.loc['TDPs0',LU] = 0
                        
        # Set initial agricultural labile P and soil TDP masses as variables to be updated during
        # looping (assume semi-natural remain at 0 for both)
        
        Plab0_A, TDPs0_A = p_LU.loc['Plab0','A'], p_LU.loc['TDPs0','A']
        
        # Initial labile P and soil TDP mass on newly converted land use class
        if p_SC.loc['NC_type',SC]=='S':
            Plab0_NC = Plab0_A  # New class is SN, from arable, therefore start with arable labile P
            TDPs0_NC = TDPs0_A
        else:
            Plab0_NC = 0.0      # New class is arable, from SN, therefore start with no labile P
            TDPs0_NC = p_LU.loc['TDPs0','S']

        # Set the value for Kf, the adsorption coefficient (mm/kg soil)
        if p_SU.run_mode == 'cal': # If the calibration period, calculate.
            # Assume SN has EPC0=0, PlabConc =0. Units: (kg/mg)(mg/kgSoil)(mm/kg)
            Kf = 10**-6*(p_LU['A']['SoilPconc']-p_LU['S']['SoilPconc'])/p_LU['A']['EPC0_0']  
        else:  # If not the calibration period, read Kf in from the series of param values
            Kf = p['Kf']    
        
        # 4) Initial reach volume (mm)
        Tr0 = (p_SC.loc['L_reach',SC]/
               (p['a_Q'] * Qr0**p['b_Q'] * 8.64*10**4)) # Reach time constant (days); T=distance/speed = L/aQ^b
        Vr0 = Tr0*Qr0 # (V=TQ; mm=days*mm/day)
        
        # If sewage effluent inputs left blank, replace with 0
        if np.isnan(p_SC.loc['TDPeff',SC]):
            p_SC.loc['TDPeff',SC] = 0.
        
        # --------------------------------------------------------------------------------------
        # ADMIN
    
        # Dictionary of slopes for different land use classes in the sub-catchment:
        slope_dict = {'A':p_SC.loc['S_Ar',SC], 'IG':p_SC.loc['S_IG',SC], 'S':p_SC.loc['S_SN',SC]}
              
        # Lists to store output
        output_ODEs = []    # From ode_f function
        output_nonODE = []  # Will include: Qq, Qr_US (latter for checking only)   
        
        # List of upstream reaches
        
        # This is extracted from parameter file. User could supply nothing (blank cell),
        # a single integer, or a list of integers (read as a single string). Convert all to list type.
        # If a string (i.e. a list of reaches), convert cell value to a list of integers
        if isinstance(p_struc.loc[SC,'Upstream_SCs'], basestring):
            upstream_SCs = [int(x.strip()) for x in p_struc.loc[SC,'Upstream_SCs'].split(',')]
        # If an integer (or not a NaN, just in case...), read cell value directly into list
        elif isinstance(p_struc.loc[SC,'Upstream_SCs'], int) or not np.isnan(p_struc.loc[SC,'Upstream_SCs']):
            upstream_SCs = [p_struc.loc[SC,'Upstream_SCs']]
        # If cell is blank, make an empty list
        else:
            upstream_SCs = []
        
        # #####################################################################################
        # START LOOP OVER MET DATA
        for idx in range(len(met_df)):

            # -----------------------------------------------------------------------------
            # Daily hydrology
            
            # Get hydrological inputs and evapotranspiration for this day
            P = met_df.ix[idx, 'P']
            E = met_df.ix[idx, 'PET']

            # Calculate infiltration excess (mm/(day * catchment area))
            Qq_i = p['f_quick']*P
            
            # -----------------------------------------------------------------------------
            # Inputs to reach from upstream. This is long-winded, could simplify by e.g. using
            # a dictionary and looping
            
            # If have upstream reaches
            if len(upstream_SCs)>0:
                
                # Print some output
                if idx==0:
                    print 'Reaches directly upstream of this reach: %s' %upstream_SCs
                
                # Empty lists to be populated with end-of-day values from upstream reaches
                Qr_US_li = [] # Discharge (mm/d)
                Msus_US_li = [] # Suspended sediment (kg)
                TDPr_US_li = [] # Total dissolved P (kg)
                PPr_US_li = [] # Particulate P (kg)
                
                # Loop and populate lists
                for upstream_SC in upstream_SCs:
                    # For discharge, mm/d value needs scaling by difference in areas between the two
                    # sub-catchments (i.e. convert to cumecs then back again)
                    Qr_US = (df_R_dict[upstream_SC].ix[idx,'Qr']
                             * (p_SC.loc['A_catch',upstream_SC]/p_SC.loc['A_catch',SC])) # Ratio of areas
                    Msus_US = df_R_dict[upstream_SC].ix[idx,'Msus']
                    TDPr_US = df_R_dict[upstream_SC].ix[idx,'TDPr']
                    PPr_US = df_R_dict[upstream_SC].ix[idx,'PPr']
                    # Append to lists
                    Qr_US_li.append(Qr_US)
                    Msus_US_li.append(Msus_US)
                    TDPr_US_li.append(TDPr_US)
                    PPr_US_li.append(PPr_US)
                # Sum inputs from all reaches directly upstream
                Qr_US_i = sum(Qr_US_li)
                Msus_US_i = sum(Msus_US_li)
                TDPr_US_i = sum(TDPr_US_li)
                PPr_US_i = sum(PPr_US_li)
                
            else:
                if idx==0:
                    print 'No reaches directly upstream of this reach'
                # If no upstream reaches, set the input from upstream reaches to 0
                Qr_US_i, Msus_US_i, TDPr_US_i, PPr_US_i = 0.0, 0.0, 0.0, 0.0

            # -----------------------------------------------------------------------------
            # Calculate delivery of sediment to the stream (kg/day)
            # NB this flux assumes the land use covers the whole catchment. Divide by area to get areal flux
            Esus_i = pd.Series(3*[np.NaN],['A','S','IG']) # Empty series to store results in later
            dayNo = met_df.index[idx].dayofyear  # Day of the year (1 to 365)
            for LU in ['A','S','IG']:
                if LU == 'A':
                    # If arable land, work out a dynamic crop cover factor, to account for the variation
                    # in erodibility through the year due to harvesting and planting practices.
                    if dynamic_options['Dynamic_erodibility'] == 'y':
                        
                        # Using a sine wave to simulate the annual change in erodibility
#                         C_spr_t = p_LU[LU]['C_cover']*(np.cos((2*np.pi/365)*(dayNo-p['d_maxE_spr']))+1)
#                         C_aut_t = p_LU[LU]['C_cover']*(np.cos((2*np.pi/365)*(dayNo-p['d_maxE_aut']))+1)
#                         C_cover = (p_SC.loc['f_spr',SC]*C_spr_t +(1-p_SC.loc['f_spr',SC])*C_aut_t)
                        
                        # Using a triangular wave
                        C_cov_dict = {} # Dict for storing spring & autumn results in
                        for s in ['spr','aut']:  # Loop through seasons
                            d = Erisk_dict # Defined above; dict defining high erosion risk period
                            d_start, d_end, d_mid = d[s]['start'], d[s]['end'], d[s]['mid']
                            if dayNo in np.arange(d_start, d_end):
                                if dayNo < d_mid: # If within high risk period, before mid-point
                                    C_season = hf.lin_interp(dayNo, x0=d_start, x1=d_mid,
                                                     y0=p_LU[LU]['C_cover'], y1=1.0)
                                else: # If in high risk period, after mid-point
                                    C_season = hf.lin_interp(dayNo, x0=d_mid, x1=d_end,
                                                     y0=1.0, y1=p_LU[LU]['C_cover'])
                            else: # Otherwise, outside high risk period
                                C_season = (p_LU[LU]['C_cover']-(E_risk_period*(1-p_LU[LU]['C_cover'])
                                           /(2*(365-E_risk_period))))
                            C_cov_dict[s] = C_season
                        # Average the dynamic factor over spring and autumn-sown crops
                        C_cover = (p_SC.loc['f_spr',SC]*C_cov_dict['spr']
                                   + (1-p_SC.loc['f_spr',SC])*C_cov_dict['aut'])  
                        
                    else:  # If not calculating a dynamic crop cover, then just assign user parameter
                        C_cover = p_LU[LU]['C_cover']
                        
                    C_cover_A = C_cover  # Store this for arable land, for checking
                
                else:  # For non-arable LU, the cover factor is always constant throughout the year
                    C_cover = p_LU[LU]['C_cover']
                
                # Reach sed input coefficient per land use class (kg/d). See documentation for rationale/source
                Esus_i[LU] = (p['E_M'] * p_SC.loc['S_reach',SC]
                              * slope_dict[LU]
                              *C_cover
                              *(1-p_LU[LU]['C_measures']))

            # -----------------------------------------------------------------------------
            # EPC0
            
            # Calculate dynamic EPC0 as a function of labile P mass
            if dynamic_options['Dynamic_EPC0'] == 'y':
                EPC0_A_i = Plab0_A/(Kf*Msoil) # Agricultural EPC0; equals EPC0_0 on the 1st timestep
                EPC0_NC_i = Plab0_NC/(Kf*Msoil) # EPC0 on newly-converted land
           
            # Or, have a constant EPC0 throughout the model run
            else:
                EPC0_A_i = p_LU['A']['EPC0_0']
                if p_SC.loc['NC_type',SC] == 'S': # (little point in a new class with constant EPC0)
                    EPC0_NC_i = p_LU['A']['EPC0_0']  # New semi-natural land has agricultural EPC0
                else:
                    EPC0_NC_i = p_LU['S']['EPC0_0']  # New agricultural has SN EPC0

            # -----------------------------------------------------------------------------
            # Append quick flow, EPC0 and crop cover factor to non-ODE results
            output_nonODE_i = [Qq_i, EPC0_A_i, EPC0_NC_i, C_cover_A ]
            output_nonODE.append(output_nonODE_i)

            # #############################################################################
            # Input to solver & solve
            
            # Vector of initial conditions for start of time step (assume 0 for initial Qr_av,
            # Msus_out, TDPr_out and PPr_out, the mass or vol of water lost per time step
            y0 = [VsA0, QsA0, VsS0, QsS0, Vg0, Qg0, Vr0, Qr0, 0.0, Msus0, 0.0, Plab0_A, Plab0_NC,
                  TDPs0_A, TDPs0_NC, TDPr0, 0.0, PPr0, 0.0]

            # Today's hydrol inputs, PET & model parameters for input to solver. NB the order must be the
            # same as the order in which they are unpacked within the odeint function
            ode_params = [P, E, mu, Qq_i, Qr_US_i, Esus_i, Msus_US_i, TDPr_US_i, PPr_US_i,
                          p_SC.loc['f_A',SC], p_SC.loc['f_Ar',SC], p_SC.loc['f_IG',SC], p_SC.loc['f_S',SC],
                          p_SC.loc['f_NC_A',SC], p_SC.loc['f_NC_Ar',SC], p_SC.loc['f_NC_IG',SC],
                          p_SC.loc['f_NC_S',SC], p_SC.loc['NC_type',SC],
                          p['f_quick'], p['alpha'], p['beta'],
                          p_LU.loc['T_s',['A','S']], p['T_g'], p['fc'],
                          p_SC.loc['L_reach',SC], p_SC.loc['A_catch',SC],
                          p['a_Q'], p['b_Q'], p['E_M'], p['k_M'],
                          p_LU.loc['P_netInput',['A','NC']], EPC0_A_i, EPC0_NC_i, Kf, Msoil,
                          p_SC.loc['TDPeff',SC], p['TDPg'], p['E_PP'], P_inactive,
                          dynamic_options['Dynamic_EPC0']]

            # Solve ODEs. Returns: y: array of results evaluated at the time points of interest.
            # N.B. rtol is part of the error tolerance. Default is ~1e-8, but reducing it removes the
            # risk of the solver exiting before finding a solution due to reaching max number of steps
            # (in which case can get odd output). Also speeds up runtime.
            # This issue can also be circumvented by increasing the max. number of steps.
            y, output_dict = odeint(ode_f, y0, ti, args=(ode_params,),full_output=1, rtol=0.01, mxstep=5000)

            # Extract values for the end of the step and store
            res = y[1]
            res[res<0] = 0
            output_ODEs.append(res)

            # ----------------------------------------------------------------------------------------
            # Update initial conditions for next step
            VsA0 = res[0]
            VsS0 = res[2]
            
            # Reading QsA0 and QsS0 from the results should give the same result as
            # recalcultaing from VsA0 or VsS0. It does with James' simple version of the hydrol
            # model. It doesn't here - because of setting results back to zero (line 611 above).
            # Not setting results back to 0 and reading results directly is a bit better, but still
            # some dodginess. And means have negative soil water flows which give artefacts in-stream.
            # All down to the sigmoid function and soil water volumes just under field capacity giving
            # negative values. Would be nice to replace with something smoother, but can't think of an
            # easy fix which doesn't really change the conceptualisation of the model. 
            # Unsatisfying, leave for now
#             QsA0 = res[1]
#             QsS0 = res[3]
            QsA0 = (VsA0 - p['fc'])/(p_LU['A']['T_s']*(1 + np.exp(p['fc'] - VsA0)))
            QsS0 = (VsS0 - p['fc'])/(p_LU['S']['T_s']*(1 + np.exp(p['fc'] - VsS0)))
            
            # Vg0 = res[4], but it's calculated from Qg0 to take into account the minimum flow
            # Re-set groundwater to user-supplied minimum flow at start of each time step. Non-ideal
            # solution to the problem of maintaining stream flow during baseflow conditions.
            if p['Qg_min'] > res[5]:
                Qg0 = p['Qg_min']
            else:
                Qg0 = res[5]
            Vg0 = Qg0 *p['T_g']
            
            Vr0 = res[6]
            Qr0 = res[7]       # Qr_av_0 would be res[8], but it's always 0
            Msus0 = res[9]     # Msus_out_0 would be res[10]
            Plab0_A = res[11]
            Plab0_NC = res[12]
            TDPs0_A = res[13]
            TDPs0_NC = res[14]
            TDPr0 = res[15]    # TDPr_out_0 would be res[16], but it's always 0
            PPr0 = res[17]     # TDPr_out_0 would be res[18], but it's always 0

            # END LOOP OVER MET DATA
            
        # #############################################################################################
    
        # Build a dataframe of ODE results
        df_ODE = pd.DataFrame(data=np.vstack(output_ODEs),
                              columns=['VsA', 'QsA','VsS', 'QsS', 'Vg', 'Qg', 'Vr', 'Qr_instant', 'Qr',
                                       'Msus_instant', 'Msus', 'P_labile_A', 'P_labile_NC', 'TDPs_A', 'TDPs_NC',
                                       'TDPr_instant', 'TDPr', 'PPr_instant', 'PPr'], 
                              index=met_df.index)

        # Dataframe of non ODE results
        df_nonODE = pd.DataFrame(data=np.vstack(output_nonODE),
                                 columns=['Qq','EPC0_A_kgmm', 'EPC0_NC_kgmm', 'C_cover_A'], 
                                 index=met_df.index)        
    
        # ####################################################################################
        # POST-PROCESSING OF MODEL OUTPUT

        # 1) Terrestrial compartment
        
        # Rearrange ODE and non-ODE result dataframes into results dataframes for the terrestrial
        # compartment (df_TC) and the stream reach (df_R)
        df_TC = pd.concat([df_ODE[['VsA', 'QsA','VsS', 'QsS', 'Vg', 'Qg', 'P_labile_A',
                                   'P_labile_NC','TDPs_A', 'TDPs_NC']],
                           df_nonODE], 
                           axis=1)

        # Calculate simulated concentrations and add to results
        df_TC['TDPs_A_mgl'] = hf.UC_C(df_TC['TDPs_A']/df_TC['VsA'], p_SC.loc['A_catch',SC])
        df_TC['EPC0_A_mgl'] = hf.UC_C(df_TC['EPC0_A_kgmm'], p_SC.loc['A_catch',SC])
        df_TC['Plabile_A_mgkg'] = (10**6*df_TC['P_labile_A']/(p['Msoil_m2']
                                   * 10**6 * p_SC.loc['A_catch',SC]))
        
        # If have some newly-converted land, add results to dataframe
        if p_SC.loc['NC_type',SC] != 'None':
            if p_SC.loc['NC_type',SC] == 'A':  # If SN converted to arable, assume arable hydrol
                df_TC['VsNC'] = df_TC['VsA']
                df_TC['QsNC'] = df_TC['QsA']
            else:  # If arable converted to SN, assume instantly has semi-natural hydrol
                df_TC['VsNC'] = df_TC['VsS']
                df_TC['QsNC'] = df_TC['QsS']
            df_TC['TDPs_NC_mgl'] = hf.UC_C(df_TC['TDPs_NC']/df_TC['VsNC'], p_SC.loc['A_catch',SC])
            df_TC['Plabile_NC_mgkg'] = (10**6*df_TC['P_labile_NC']
                                                /(p['Msoil_m2']*10**6 *p_SC.loc['A_catch',SC]))
        # Add snow depth (if calculated)
        if p_SU.inc_snowmelt == 'y':
            df_TC['D_snow'] = met_df['D_snow_end']

        # 2) In-stream
        # NB masses of SS and P are all total fluxes for the day, and Q is the daily mean flow
        df_R = df_ODE.drop(['VsA', 'QsA','VsS', 'QsS', 'Vg', 'Qg', 'Qr_instant',
                            'Msus_instant','P_labile_A', 'P_labile_NC', 'TDPs_A', 'TDPs_NC',
                            'TDPr_instant', 'PPr_instant'], axis=1)
        
        # Post-processing of results
        
        # Convert flow units from mm/d to m3/s
        df_R['Sim_Q_cumecs'] = df_R['Qr']*p_SC.loc['A_catch',SC]*1000/86400
        
        # Calculate concentrations (mg/l); generally from (kg/d)(d/mm)
        df_R['SS_mgl'] = hf.UC_C(df_R['Msus']/df_R['Qr'],p_SC.loc['A_catch',SC])
        df_R['TDP_mgl'] = hf.UC_C(df_R['TDPr']/df_R['Qr'],p_SC.loc['A_catch',SC])
        df_R['PP_mgl'] = hf.UC_C(df_R['PPr']/df_R['Qr'],p_SC.loc['A_catch',SC])
        
        # Calculate derived variables: TP, SRP, and add to results dataframe
        df_R = derived_P_species(df_R, p['f_TDP'])
        
        # ------------------------------------------------------------------------------------
        # Sort indices & add to dictionaries; key = sub-catchment/reach number
        df_TC = df_TC.sort_index(axis=1)
        df_R = df_R.sort_index(axis=1)
        df_TC_dict[SC] = df_TC
        df_R_dict[SC] = df_R
        
        print ('Finished!\n')
        # END LOOP OVER SUB-CATCHMENTS
        
    # #########################################################################################
    # OUTPUT
    
    # If in calibration mode, print the calculated Kf value
    if p_SU.run_mode == 'cal':
        print ("Kf (the soil P sorption coefficient; mm/kg): %s\n" % Kf)
    
    # Save csvs of results
    if p_SU.save_output_csvs == 'y':
        for SC in df_R_dict.keys():
            df_TC_dict[SC].to_csv(os.path.join(p_SU.output_fpath, "Results_TC_SC%s.csv" % SC))
            df_R_dict[SC].to_csv(os.path.join(p_SU.output_fpath, "Instream_results_Reach%s.csv" % SC))
        print ('Results saved to csv\n')
    
        
    return (df_TC_dict, df_R_dict, Kf, output_dict)  # NB Kf is returned for the last SC only

# ###############################################################################################
# Post-processing of reach results
def derived_P_species(df_R, f_TDP):
    """
    Calculate: Total P as the sum of TDP and PP
               SRP as a function of TDP
    Inputs: reach results dataframe; value to multiply TDP by to get SRP
    Returns: dataframe of reach results, with extra columns added
    """
    df_R['TP_mgl'] = df_R['TDP_mgl'] + df_R['PP_mgl']
    
    # Calculate SRP from TDP using a constant user-supplied factor
    df_R['SRP_mgl'] = df_R['TDP_mgl']*f_TDP
    
    return df_R

# ###############################################################################################

def sum_to_waterbody(p_struc, n_SC, df_R_dict, f_TDP):
    
    vars_to_sum = ['Sim_Q_cumecs','Msus','TDPr','PPr']
    reaches_in_final_flux = p_struc['In_final_flux?'][p_struc['In_final_flux?']==1].index.values
    
    # Check that the reach structure sheet matches the number of sub-catchments the user told
    # the model to expect
    if len(reaches_in_final_flux)>n_SC:
        raise ValueError("Mismatch between the number of subcatchments in the 'Setup' parameter sheet \n(parameter 'n_SC') and in the 'Reach_structure' parameter sheet")
    
    print ('Sub-catchments flowing directly into receiving waterbody: %s' %reaches_in_final_flux)
    
    summed_series_li = []
    for var in vars_to_sum:
        reach_data_li = []
        for reach in reaches_in_final_flux:
            reach_data_li.append(df_R_dict[reach][var])
        summed_series = pd.DataFrame(zip(*reach_data_li), columns=reaches_in_final_flux, index=df_R_dict[reach][var].index).sum(axis=1)
        summed_series_li.append(summed_series)

    df_summed = pd.DataFrame(zip(*summed_series_li), columns=vars_to_sum, index=df_R_dict[reach][var].index)
    
    # Calculating concentrations in mg/l: (daily mass/Q_cumecs) * (kg/day)(s/m3) (1 day/86400 s) (10**6 mg/kg) (10**-3 m3/l))
    df_summed['SS_mgl'] = (df_summed['Msus']/df_summed['Sim_Q_cumecs']) * (1000./86400.)
    df_summed['TDP_mgl'] = (df_summed['TDPr']/df_summed['Sim_Q_cumecs']) * (1000./86400.)
    df_summed['PP_mgl'] = (df_summed['PPr']/df_summed['Sim_Q_cumecs']) * (1000./86400.)
    
    # Derive total P and SRP
    df_summed = derived_P_species(df_summed, f_TDP)
    
    return df_summed 
    
# ###############################################################################################

# For testing when running the script as standalone (__name__ = 'simply_p' when imported as a module)
if (__name__ == '__main__'):
    # Write the output you want to appear here
    pass