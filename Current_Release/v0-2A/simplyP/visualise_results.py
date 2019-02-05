"""
File containing functions for plotting and calculating performance statistics
"""
import pandas as pd, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.ticker import MaxNLocator
import os

# Plot styling
# mpl.rcParams['pdf.fonttype'] = 42
# sn.set_context('notebook') # changes graph font attributes
sn.set_context(rc={'lines.markeredgewidth': 0.4}) # otherwise seaborn over-writes mpl
plt.style.use('ggplot')

# Global variables
_max_yticks = 5

def _setformat(fig_display_type):
    if fig_display_type=='paper':
        w = 7.5     # figure width, in inches; fits with elsevier journals
        h = 1       # subplot height, inches
        ticklabelsize = 6
        axlabelsize = 9
    else:
        w = 16      # figure width, in inches; larger for on-screen displaying
        h = 3.0     # subplot height, inches
        ticklabelsize = 9
        axlabelsize = 12
        return(w, h, ticklabelsize, axlabelsize)

def plot_snow(met_df, p_SU, fig_display_type):
    """ Plot results from snow module, including met data. Only relevant when inc_snowmelt == 'y'
        in parameter template.

    Args:
        met_df:           Dataframe. Meteorological dataframe modified by hydrol_inputs() function
        p_SU:             Series. User-specified setup options
        fig_display_type: Str. 'paper' or 'notebook'
        
    Returns:
        Axes. Plot is saved using parameters specified in p_SU. Results will also be displayed in
        Jupyter if using %matplotlib inline.
    """
    
    # Set format
    w, h, ticklabelsize, axlabelsize = _setformat(fig_display_type)

    # Dictionary for re-naming ODE-output dataframe columns to match columns in obs dataframe
    # (used in stats calc as well as in in-stream plot)
    rename_dict = {'SS_mgl':'SS','TDP_mgl':'TDP','PP_mgl':'PP','TP_mgl':'TP','Q_cumecs':'Q'}

    # PLOT RESULTS OF SNOW MODULE, WITH MET DATA
    if p_SU.inc_snowmelt == 'y':
        # Dictionary for y-axis labels
        met_ylab_d = {'T_air':'Air temp\n(deg C)','PET':'PET (mm/d)', 'Precipitation':'Total pptn\n(mm/d)',
                      'P_snow':'Precipitation as\nsnow (mm/d)', 'P_rain':'Precipitation as\nrain (mm/d)',
                      'P_melt':'Snow melt\n(mm/d)', 'D_snow_end':'Snow depth\n(mm)','P':'Rain &\nmelt (mm/d)' }
        
        met_plotVars = ['T_air','Precipitation','P_snow','P_melt','D_snow_end','P']  # Variables to plpot
        
        # PLOT
        met_df_forPlot = met_df[met_plotVars]
        fig_snow_axes = met_df_forPlot.plot(subplots=True,figsize=(w,len(met_plotVars)*h+1),
                                            legend=False)
        # Tidy plot & save
        for i, ax in enumerate(fig_snow_axes):
            fig_snow_axes[i].set_ylabel(met_ylab_d[met_plotVars[i]], fontsize=axlabelsize)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=_max_yticks, prune='upper'))
            plt.xlabel("")
            ax.tick_params(axis='both', which='major', labelsize=ticklabelsize)
            ax.tick_params(axis='both', which='minor', labelsize=ticklabelsize)
        if p_SU.plot_snow == 'y':
            fname_snow = os.path.join(p_SU.output_fpath, "Fig_snow_results.%s" %p_SU.output_figtype)
            plt.savefig(fname_snow, bbox_inches='tight', dpi=p_SU.output_fig_dpi)

    else:
        raise ValueError('Snowfall/melt has not been estimated because inc_snowmelt != "y" in the setup file.') 

# ########################################################################################
def plot_terrestrial(p_SU, p_SC, p, df_TC_dict, met_df, fig_display_type):
    """ Plot results from terrestrial calculations.

    Args:
        p_SU:             Series. User-specified setup options
        p_SC:             Series. Sub-catchment-specific parameters
        p:                Series. Constant parameters
        df_TC_dict:       Dict. Returned from run_simply_p()
        met_df:           Dataframe. Meteorological dataframe modified by hydrol_inputs() function
        fig_display_type: Str. 'paper' or 'notebook'
        
    Returns:
        None. Plots are saved using parameters specified in p_SU. Results will also be displayed in
        Jupyter if using %matplotlib inline.
    """
    # Set format
    w, h, ticklabelsize, axlabelsize = _setformat(fig_display_type)

    # Dictionary for re-naming y-axis label, to include full words and units
    TC_ylab_d = {'P':'Rain & melt\n(mm/d)', 'PET':'Potential ET\n(mm/d)',
                 'Qq':'Quick Q\n(mm/d)',
                 'QsA':'SW Q, Agri\n(mm/d)','QsS':'SW Q, SN\n(mm/d)',
                 'Qg': 'GW Q\n(mm/d)','VsA': 'SW vol,\nAgri (mm)',
                 'VsS': 'SW vol,\nSN (mm)','Vg':'GW vol\n(mm)',
                 'Plabile_A_mgkg':'Labile P\nAgri (mg/kg)', 'EPC0_A_mgl':'EPC$_0$,\nAgri (mg/l)',
                 'TDPs_A_mgl':'SW TDP,\nAgri (mg/l)', 'Plabile_NC_mgkg':'Labile P\nNC (mg/kg)',
                 'EPC0_NC_mgl':'EPC$_0$,\n NC (mg/l)', 'TDPs_NC_mgl':'SW TDP,\nNC (mg/l)',
                 'C_cover_A':'Erodibility\nC factor','Mland_A':'Sed yield, Agri\n(kg km$^{-2}$d$^{-1}$)',
                 'Mland_IG':'Sed yield, IG\n(kg km$^{-2}$d$^{-1}$)',
                 'Mland_S':'Sed yield, SN\n(kg km$^{-2}$d$^{-1}$)'}

    # Start plotting

    # Plot 1: hydrology
    TC_f1_vars = ['P','PET','Qq','QsA','QsS','Qg','VsA','VsS','Vg'] # Variables for 1st plot
    
    df_TC_hydrol = df_TC_dict[1][TC_f1_vars[2:]] # Just plot for 1st sub-catchment
    df_TC_hydrol = pd.concat([met_df[['P', 'PET']], df_TC_hydrol], axis=1)
    TC_fig1_axes = df_TC_hydrol.plot(subplots=True, figsize=(w, len(TC_f1_vars)*h+1), legend=False)
    for i, ax in enumerate(TC_fig1_axes):
        # If soil water volume, add on field capacity
        if i in [6,7]:
            ax.axhline(p.fc, color='0.4', alpha=0.5, lw=1.3, label='Field capacity')
        TC_fig1_axes[i].set_ylabel(TC_ylab_d[TC_f1_vars[i]], fontsize=axlabelsize)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=_max_yticks, prune='upper'))
        plt.xlabel("")
        ax.tick_params(axis='both', which='major', labelsize=ticklabelsize)
        ax.tick_params(axis='both', which='minor', labelsize=ticklabelsize)
    if p_SU.plot_TC == 'y':
        fname_TC1 = os.path.join(p_SU.output_fpath, "Fig_TC_hydrol.%s" % p_SU.output_figtype)
        plt.savefig(fname_TC1, bbox_inches='tight', dpi=p_SU.output_fig_dpi)

    # Plot 2: soil P   
    if p_SU.Dynamic_EPC0 == 'y':
        
        # Variables in 2nd plot; depends if have NC land
        if p_SC.loc['NC_type',1] != 'None':
            TC_f2_vars = ['Plabile_A_mgkg', 'EPC0_A_mgl', 'TDPs_A_mgl', 'Plabile_NC_mgkg',
                            'EPC0_NC_mgl', 'TDPs_NC_mgl']
        else:
            TC_f2_vars = ['Plabile_A_mgkg', 'EPC0_A_mgl', 'TDPs_A_mgl']
                
        df_TC_soilP = df_TC_dict[1][TC_f2_vars] # Just plot for 1st sub-catchment
        TC_fig2_axes = df_TC_soilP.plot(subplots=True, figsize=(w, len(TC_f2_vars)*h+1), legend=False)
        plt.xlabel("")
        for i, ax in enumerate(TC_fig2_axes):
            TC_fig2_axes[i].set_ylabel(TC_ylab_d[TC_f2_vars[i]], fontsize=axlabelsize)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=_max_yticks, prune='upper'))
            ax.tick_params(axis='both', which='major', labelsize=ticklabelsize)
            ax.tick_params(axis='both', which='minor', labelsize=ticklabelsize)
        if p_SU.plot_TC == 'y':
            fname_TC2 = os.path.join(p_SU.output_fpath, "Fig_TC_soilP.%s" % p_SU.output_figtype)
            plt.savefig(fname_TC2, bbox_inches='tight', dpi=p_SU.output_fig_dpi)
    
    # Plot 3: 
    if p_SU.Dynamic_erodibility == 'y':  # Variables for 3rd plot; depends on erodibility option
        TC_f3_vars = ['C_cover_A', 'Mland_A', 'Mland_IG', 'Mland_S']
    else:
        TC_f3_vars = ['Mland_A', 'Mland_IG', 'Mland_S']

        
# ########################################################################################
def plot_in_stream(p_SU, obs_dict, df_R_dict, fig_display_type):
    """ Plot results from in-stream calculations.

    Args:
        p_SU:             Series. User-specified setup options
        df_R_dict:        Dict. Returned from run_simply_p()
        obs_dict:         Dict. Observed discharge and chemistry data
        fig_display_type: Str. 'paper' or 'notebook'
        
    Returns:
        None. Plots are saved using parameters specified in p_SU. Results will also be displayed in
        Jupyter if using %matplotlib inline.
    """
       
    # SET UP
    
    # Decide whether or not observations are plotted, according to the run_mode setup parameter
    if p_SU.run_mode == 'scenario':
        plot_obs = 'n'
    else:
        plot_obs = 'y'  # i.e. only plot obs for calibration & validation plots

    # Set format
    w, h, ticklabelsize, axlabelsize = _setformat(fig_display_type)
    
    # Dictionary for re-naming y-axis label
    y_lab_d = {'SS': 'SS (mg/l)', 'TDP': 'TDP (mg/l)', 'PP':'PP (mg/l)', 'TP':'TP (mg/l)',
               'Q':'Q (m$^3$/s)', 'SRP': 'SRP (mg/l)'}
    # Formatting choices for observation points or line
    obs_lc_d = {'line':'0.5', 'point':'None'} # Line colour
    obs_ls_d = {'line':'-', 'point':'none'}   # Line style
    obs_marker_d = {'line':'.', 'point':'^'}  # Marker style
    obs_ms_d = {'line':3, 'point':3}          # Marker size
    obs_mc_d = {'line':'0.3', 'point':'w'}    # Marker colour
    obs_mec_d = {'line':'None', 'point':'k'}  # Marker edge colour
    obs_lw_d = {'line':1.5, 'point':1}        # Line width

    # Formatting set-up for the simulated line
    if p_SU.colour_option == 'colour':
        sim_color = 'r'
    else:
        sim_color = 'k'

    # List of reaches user wants to plot results for
    # If a string, could be 'all' or a list of reaches (eg. '1,2')
    if isinstance(p_SU.plot_reaches, basestring):
        if p_SU.plot_reaches == 'all':
            reach_list = df_R_dict.keys() # If all, populate with all reaches
        else:
            # If just some reaches, extract these from param file
            reach_list = [int(x.strip()) for x in p_SU.plot_reaches.split(',')]
    else:
        # If just one reach, this won't be a string, so extract directly
        reach_list = [p_SU.plot_reaches]
        

    # User-supplied list of variables for plotting
    R_vars_to_plot = [x.strip() for x in p_SU.R_vars_to_plot.split(',')] # Stripping whitespace

    # Plotting options - log-transforming y axis
    logy_li = [x.strip() for x in p_SU.logy_list.split(',')] # List of variables with log y axes
    # Create logy dict
    logy_dict = {}
    for var in R_vars_to_plot:
        if var in logy_li:
            logy_dict[var] = True
        else:
            logy_dict[var] = False

    for SC in reach_list:

        # Extract simulated data    
        df_R_toPlot = df_R_dict[SC][['SS_mgl','TDP_mgl','PP_mgl','TP_mgl','Q_cumecs','SRP_mgl']] # All vars
        df_R_toPlot.columns = ['SS','TDP','PP','TP','Q','SRP'] # Rename columns to match obs & param file
        df_R_toPlot = df_R_toPlot[R_vars_to_plot] # Remove any columns that aren't to be plotted

        # PLOT
        fig = plt.figure(figsize=(w, len(R_vars_to_plot)*h+1)) 
        for i, var in enumerate(R_vars_to_plot):
            ax = fig.add_subplot(len(R_vars_to_plot),1,i+1)

            # Plot observed, if have observations for this reach
            if SC in obs_dict.keys() and plot_obs=='y':  # If have obs for this SC and are in cal or val period
                obs_vars = obs_dict[SC].columns  # Variables with obs in this SC
                # If necessary, modify simulated R_vars_to_plot list, if don't have obs.
                # Do by picking out the common elements from a & b. Returns a set, so convert to a list
                R_obsVars_toPlot = list(set(R_vars_to_plot).intersection(obs_vars))
                obs_df = obs_dict[SC][R_obsVars_toPlot]  # Extract data for this SC
                if var in obs_df.columns: # If have observations for this variable
                    n_obs = sum(obs_df[var].notnull()) # Number of observations
                    if n_obs>0:  # If no observations for this time period, then don't plot
                        if var in logy_li:
                            log_yn = True
                        else:
                            log_yn = False
                        # Determine the plot style - line if Q, otherwise user-specified
                        if var == 'Q' or p_SU.plot_obs_style == 'line':
                            style='line'
                        else:
                            style='point'
                        obs_df[var].plot(ax=ax, marker=obs_marker_d[style],
                                         ls=obs_ls_d[style], ms=obs_ms_d[style],
                                         mfc=obs_mc_d[style], mec=obs_mec_d[style], color=obs_lc_d[style],
                                         lw=obs_lw_d[style],
                                         logy=log_yn, label='Obs')

            # Plot simulated
            df_R_toPlot[var].plot(ax=ax, color=sim_color, lw=0.6, logy=logy_dict[var], label='Sim')

            # Tidy up plot
            if SC in obs_dict.keys() and var in obs_df.columns and plot_obs=='y':
                ax.legend(loc='best', prop={'size':6}, frameon=True)   # If have two lines on plot, add a legend
            if var not in logy_li:      # If not log-transformed, cut down tick labels on y-axis
                ax.yaxis.set_major_locator(MaxNLocator(nbins=_max_yticks, prune='upper'))
            if var == 'SS' and var in logy_li:  # !!!May not be appropriate outside the Tarland!!
                ax.set_ylim(1)
            plt.ylabel(y_lab_d[var],fontsize=axlabelsize)
            plt.xlabel("")
            plt.suptitle("Reach %s" %SC)
            if i != len(R_vars_to_plot)-1:   # Turn off x-axis tick labels unless it's the bottom sub-plot
                plt.tick_params(axis='x', labelbottom='off')
            plt.tick_params(axis='both', which='major', labelsize=ticklabelsize)
            plt.tick_params(axis='both', which='minor', labelsize=ticklabelsize)

        if p_SU.plot_R == 'y':
            # Save figure
            fname_reach_ts = os.path.join(p_SU.output_fpath, "Fig_reach%s_timeseries.%s" % (SC, p_SU.output_figtype))
            plt.savefig(fname_reach_ts, bbox_inches='tight', dpi=p_SU.output_fig_dpi)


# ########################################################################################
def plot_instream_summed(p_SU, df_summed, fig_display_type):
    """ Plot results from summing reach inputs to produce single set of fluxes to a receiving waterbody.
        Optionally also save plot, and save results to csv

        Args:
            p_SU:             Series. User-specified setup options
            df_summed:        Dataframe of reach results
            fig_display_type: Str. 'paper' or 'notebook'

        Returns:
            None. Plots are saved using parameters specified in p_SU. Results will also be displayed in
            Jupyter if using %matplotlib inline.
    """   

    # Set formatting
    w, h, ticklabelsize, axlabelsize = _setformat(fig_display_type)
    if p_SU.colour_option == 'colour':
        sim_color = 'r'
    else:
        sim_color = 'k'
        
    # Dictionary for re-naming y-axis label
    y_lab_d = {'SS': 'SS (mg/l)', 'TDP': 'TDP (mg/l)', 'PP':'PP (mg/l)', 'TP':'TP (mg/l)',
               'Q':'Q (m$^3$/s)', 'SRP': 'SRP (mg/l)'}

    # User-supplied list of variables for plotting
    R_vars_to_plot = [x.strip() for x in p_SU.R_vars_to_plot.split(',')] # Stripping whitespace

    # Plotting options - log-transforming y axis
    logy_li = [x.strip() for x in p_SU.logy_list.split(',')] # List of variables with log y axes
    # Create logy dict
    logy_dict = {}
    for var in R_vars_to_plot:
        if var in logy_li:
            logy_dict[var] = True
        else:
            logy_dict[var] = False
    
    df_summed_toPlot = df_summed[['Q_cumecs','SS_mgl','TDP_mgl','PP_mgl','TP_mgl','SRP_mgl']]
    df_summed_toPlot.columns = ['Q','SS','TDP','PP','TP','SRP'] # Rename columns to match obs & param file
    df_summed_toPlot = df_summed_toPlot[R_vars_to_plot] # Remove any columns that aren't to be plotted
    
    # Start plotting
    fig = plt.figure(figsize=(w, len(R_vars_to_plot)*h+1)) 
    for i, var in enumerate(R_vars_to_plot):
        ax = fig.add_subplot(len(R_vars_to_plot),1,i+1)
        df_summed_toPlot[var].plot(ax=ax, color=sim_color, lw=0.6, logy=logy_dict[var])

        # Tidy up plot
        if var not in logy_li:      # If not log-transformed, cut down tick labels on y-axis
            ax.yaxis.set_major_locator(MaxNLocator(nbins=_max_yticks, prune='upper'))
        if var == 'SS' and var in logy_li:  # !!!May not be appropriate outside the Tarland!!
            ax.set_ylim(1)
        plt.ylabel(y_lab_d[var],fontsize=axlabelsize)
        plt.xlabel("")
        plt.suptitle("Inputs to receiving waterbody from all upstream areas")
        if i != len(R_vars_to_plot)-1:   # Turn off x-axis tick labels unless it's the bottom sub-plot
            plt.tick_params(axis='x', labelbottom='off')
        plt.tick_params(axis='both', which='major', labelsize=ticklabelsize)
        plt.tick_params(axis='both', which='minor', labelsize=ticklabelsize)

    if p_SU.plot_R == 'y':
        # Save figure
        fname_summed_ts = os.path.join(p_SU.output_fpath, "Fig_sum_to_waterbody_timeseries.%s" %p_SU.output_figtype)
        plt.savefig(fname_summed_ts, bbox_inches='tight', dpi=p_SU.output_fig_dpi)
        print ('Graph saved to file')
        
    if p_SU.save_output_csvs == 'y':
        df_summed.to_csv(os.path.join(p_SU.output_fpath, "Instream_results_receiving_waterbody.csv"))
        print ('Results saved to csv')
            

def goodness_of_fit_stats(p_SU, df_R_dict, obs_dict):
    """ Tabulates (and optionally saves) various goodness-of-fit statistics.

    Args:
        p_SU:             Series. User-specified setup options
        df_R_dict:        Dict. Returned from run_simply_p()
        obs_dict:         Dict. Observed discharge and chemistry data
        
    Returns:
        Dataframe of statistics.
    """
    # PERFORMANCE METRICS (if in calibration or validation mode)
    # If in calibration or validation run mode, and if have some observations, calculate stats
    if p_SU.run_mode != 'scenario' and len(obs_dict)>0:
        stats_var_li = ['Q','SS','TDP','PP','TP','SRP'] # All vars we're potentially interested in
        stats_df_li = [] # List of dfs with GoF results, one df per reach

        # Start loop over subcatchments
        for SC in df_R_dict.keys():
            stats_vars = list(stats_var_li) # List of vars that have enough obs for stats to be calculated
                                            # (amended during looping, below)
            if SC in obs_dict.keys():  # If have any observed data for this sub-catchment

                # Extract data
                # Simulated
                df_statsData = df_R_dict[SC][['Q_cumecs','SS_mgl','PP_mgl','TP_mgl','TDP_mgl','SRP_mgl']] # Simulated
                df_statsData.columns = ['Q','SS','PP','TP','TDP','SRP']   # Rename columns to match obs

                # Observed (only for observed data that can be simulated, in case file has other data)
                obs_vars = obs_dict[SC].columns  # Variables with obs in this SC according to input data
                R_obsVars_forStats = list(set(df_statsData.columns).intersection(obs_vars))        
                obs_df = obs_dict[SC][R_obsVars_forStats]

                stats_li = [] # Empty list for storing results for the different variables in this reach

                # Loop over all possible variables and, if have data for this variable in this SC,
                # and if have enough data (>10 observations per reach), calculate stats
                for var in stats_var_li:

                    # Check if have observed data for this variable
                    if var in obs_df.columns:
                        obs = obs_df[var]
                        n_obs = sum(obs.notnull()) # Number of observations

                        if n_obs>10:

                            # Get data in nice format for calculating stats
                            sim = df_statsData[var]
                            tdf = pd.concat([obs,sim],axis=1) # Temp df of aligned sim & obs data
                            tdf = tdf.dropna(how='any')       # Strip out NaNs
                            tdf.columns = ['obs','sim']       # Re-name columns
                            tldf = np.log(tdf)                # Temp df of logged values

                            # Calculate stats
                            NSE = 1 - (np.sum((tdf['obs']-tdf['sim'])**2)/np.sum((tdf['obs']-np.mean(tdf['obs']))**2))
                            log_NSE = (1 - (np.sum((tldf['obs']-tldf['sim'])**2)/
                                            np.sum((tldf['obs']-np.mean(tldf['obs']))**2)))
                            spearmans_r_array = tdf.corr(method='spearman')
                            spearmans_r = spearmans_r_array.ix[0,1]
                            r2_array = tdf.corr(method='pearson')**2
                            r2 = r2_array.ix[0,1]
                            pbias = 100*np.sum(tdf['sim']-tdf['obs'])/np.sum(tdf['obs'])
                            RMSD_norm = 100*np.mean(np.abs(tdf['sim']-tdf['obs']))/np.std(tdf['obs'])

                            # Append to results list
                            stats_li.append([n_obs, NSE, log_NSE, spearmans_r, r2, pbias, RMSD_norm])

                        else: # Not enough obs to calculate stats, so drop this var from the variables list
                            stats_vars.remove(var)

                    else: # No observed data for this variable, so drop from the variables list
                        stats_vars.remove(var)

                # Save results in a dataframe for outputing directly to notebook
                stats_df = pd.DataFrame(data=stats_li, columns=['N obs', 'NSE','log NSE','Spearmans r',
                                        'r$^2$','Bias (%)', 'nRMSD (%)'], index=stats_vars)
                stats_df['Reach'] = SC # Add Reach number as a column
                stats_df_li.append(stats_df)

        stats_df_allSC = pd.concat(stats_df_li)  # Concatenate results from all sub-catchments

        if p_SU.save_stats_csv == 'y':           # Save output to csv
            stats_fpath = os.path.join(p_SU.output_fpath, "GoF_stats.csv")
            stats_df_allSC.to_csv(stats_fpath)

        return stats_df_allSC 
    else:
        print ('No observations read in, therefore cannot calculate model performance statistics')
