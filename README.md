# SimplyP: a simple phosphorus model

This repository contains code to run a simple hydrology, sediment and phosphorus model, named **SimplyP**. The model is dynamic, working on a daily time step, and is spatially semi-distributed i.e. there is the ability to differentiate between hydrology, sediment and phosphorus processes, and between land use types and sub-catchments (with associated stream reaches). An application of the model is described in this paper:

> Jackson-Blake LA, Sample JE, Wade AJ, Helliwell RC, Skeffington RA. 2017. *Are our dynamic water quality models too complex? A comparison of a new parsimonious phosphorus model, SimplyP, and INCA-P*. Water Resources Research, **53**, 5382–5399. [doi:10.1002/2016WR020132](http://onlinelibrary.wiley.com/doi/10.1002/2016WR020132/abstract;jsessionid=7E1F1066482B9FFDBC29BA6B5A80042C.f04t01)

and full details of version 0.1 of the model are provided in the [Supplementary Information](http://onlinelibrary.wiley.com/store/10.1002/2016WR020132/asset/supinfo/wrcr22702-sup-0001-2016WR020132-s01.pdf?v=1&s=fc5ee61527c9fc914b4c14b35562f30b85d3c927). Several changes have been made to the model since then, and are detailed in the change log.

The hydrology model used by SimplyP builds upon the one described [here](https://github.com/JamesSample/enviro_mod_notes.git) (see especially [notebook 5](http://nbviewer.jupyter.org/github/JamesSample/enviro_mod_notes/blob/master/notebooks/05_A_Hydrological_Model.ipynb)). The remainder was developed by Leah Jackson-Blake and forms part of her PhD thesis. The work was funded by the RESAS, the Rural and Environment Science and Analytical Services Division of the Scottish Government and by NIVA (the Norwegian Institute for Water Research).

The current release version of the model is v0-2A. There are a number of known issues with the model (see 'Issues'); feel free to add to these. Known bugs will be corrected as soon as possible.

Please report any bugs or errors either by [submitting a pull request via GitHub](https://github.com/LeahJB/SimplyP/pulls) or by emailing Leah Jackson-Blake (<ljb@niva.no>).

## Installation

If you don't have an up-to-date Python installation, a good option is the [Anaconda Python distribution](https://www.anaconda.com/download/). Model development was carried out using **Python 2.7**, but the current code has also been tested with **Python 3.6**.

1. From the Anaconda prompt, create a new clean conda environment for SimplyP

       conda create -n simplyp python=2.7
    
2. Activate the new environment

       activate simplyp
    
3. Install the Jupyter Notebook

       conda install jupyter=1.0.0 notebook=5.4.1
    
4. Download the [SimplyP repository](https://github.com/LeahJB/SimplyP) and unzip it to a location on your system. The folder Current_Release/v0-2A contains the main model code you will need to set up and run the model (v0-2A refers to the version number). The folder 'Example_Data' includes data to get you started with a model application using data from the Tarland catchment in Scotland.

5. From the Anaconda command prompt, change directories to the 'SimplyP/v0-1A' folder (which contains 'setup.py') and run

        python setup.py install    
    
## Running the model

A simple example illustrating how the model can be used is [here](http://nbviewer.jupyter.org/github/LeahJB/SimplyP/blob/Hydrology_Model/SimplyP_v0-1A.ipynb). To run this example:

1. From the Anaconda prompt, activate your SimplyP environment

       activate simplyp
    
2. Change to the directory containing `'Run_SimplyP_v0-2A.ipynb'` within the 'SimplyP/Current_Release/v0-2A' folder and run

       jupyter notebook
    
   then click the link to open `'Run_SimplyP_v0-2A_LongExample.ipynb'`. You should now be able to work through the notebook, running the code cells interactively. More instructions are given within the notebook itself.

3. When you have finished working with the model, click 'File > Close and Halt' (remember to save any changes first, if desired), then close the browser tab. Close the `'Home'` tab too, then `CTRL + C'` twice at the Anaconda prompt to shut down any active Python kernels.

## Working with your own data

A description of the input data required by the model is given [here](https://github.com/LeahJB/SimplyP/blob/Hydrology_Model/Input_output_data_description.txt), and will be updated periodically. This file also describes the output data that may optionally be saved to file.
