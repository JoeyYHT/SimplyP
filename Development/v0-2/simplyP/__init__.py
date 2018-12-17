# Automatic imports of simplyP modules contained within the package

import simplyP as sp

from .model import (
    ode_f,
    run_simply_p,
    derived_P_species,
    sum_to_waterbody
)

from .inputs import (
    read_input_data,
    snow_hydrol_inputs,
    daily_PET
)

from .helper_functions import (
    UC_Q,
    UC_Qinv,
    UC_C,
    UC_Cinv,
    UC_V,
    lin_interp
)

from .visualise_results import(
    plot_snow,
    plot_terrestrial,
    plot_in_stream,
    plot_instream_summed,
    goodness_of_fit_stats
)

# Functions imported when import * is used for the package
__all__ = [
    # Main model
    'ode_f',
    'run_simply_p',
    'derived_P_species',
    'sum_to_waterbody'
    
    # Inputs
    'read_input_data',
    'snow_hydrol_inputs',
    'daily_PET',
    
    # Helper functions
    'UC_Q',
    'UC_Qinv',
    'UC_C',
    'UC_Cinv',
    'UC_V',
    'lin_interp',
    
    # Visualise results
    'plot_snow',
    'plot_terrestrial',
    'plot_in_stream',
    'plot_instream_summed',
    'goodness_of_fit_stats',
]

