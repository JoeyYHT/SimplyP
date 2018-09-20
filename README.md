# SimplyP: a simple phosphorus model

This repository contains code to run a simple hydrology, sediment and phosphorus model, named **SimplyP**. The model is dynamic, working on a daily time step, and is spatially semi-distributed i.e. there is the ability to differentiate between hydrology, sediment and phosphorus processes, and between land use types and sub-catchments (with associated stream reaches). An application of the model is described in this paper:

> Jackson-Blake LA, Sample JE, Wade AJ, Helliwell RC, Skeffington RA. 2017. *Are our dynamic water quality models too complex? A comparison of a new parsimonious phosphorus model, SimplyP, and INCA-P*. Water Resources Research, **53**, 5382–5399. [doi:10.1002/2016WR020132](http://onlinelibrary.wiley.com/doi/10.1002/2016WR020132/abstract;jsessionid=7E1F1066482B9FFDBC29BA6B5A80042C.f04t01)

and full details of the model itself are provided in the [Supplementary Information](http://onlinelibrary.wiley.com/store/10.1002/2016WR020132/asset/supinfo/wrcr22702-sup-0001-2016WR020132-s01.pdf?v=1&s=fc5ee61527c9fc914b4c14b35562f30b85d3c927). 

The hydrology model used by SimplyP builds upon the one described [here](https://github.com/JamesSample/enviro_mod_notes.git) (see especially [notebook 5](http://nbviewer.jupyter.org/github/JamesSample/enviro_mod_notes/blob/master/notebooks/05_A_Hydrological_Model.ipynb)). The remainder was developed by Leah Jackson-Blake and forms part of her PhD thesis. The work was funded by the RESAS, the Rural and Environment Science and Analytical Services Division of the Scottish Government.

Please report any bugs or errors either by [submitting a pull request via GitHub](https://github.com/LeahJB/SimplyP/pulls) or by emailing Leah Jackson-Blake (<ljb@niva.no>).

## Installation

If you don't have an up-to-date Python installation, a good option is the [Anaconda Python distribution](https://www.anaconda.com/download/). Model development was carried out using **Python 2.7**, but the current code has also been tested with **Python 3.6**.

1. From the Anaconda prompt, create a new clean conda environment for SimplyP

       conda create -n simplyp python=2.7
    
2. Activate the new environment

       activate simplyp
    
3. Install the Jupyter Notebook

       conda install jupyter=1.0.0 notebook=5.4.1
    
4. Download the [SimplyP repository](https://github.com/LeahJB/SimplyP) and unzip it to a location on your system

5. From the Anaconda command prompt, change directories to the 'SimplyP' folder (the one containing 'setup.py') and run

        python setup.py install    
    
## Running the model

A simple example illustrating how the model can be used is [here](http://nbviewer.jupyter.org/github/LeahJB/SimplyP/blob/Hydrology_Model/SimplyP_v0-1A.ipynb). To run this example:

1. From the Anaconda prompt, activate your SimplyP environment

       activate simplyp
    
2. Change to the directory containing `'SimplyP_v0-1A.ipynb'` and run

       jupyter notebook
    
   then click the link to open `'SimplyP_v0-1A.ipynb'`. You should now be able to work through the notebook, running the code cells interactively.

3. When you have finished working with the model, click 'File > Close and Halt' (remember to save any changes first, if desired), then close the browser tab. Close the `'Home'` tab too, then `CTRL + C'` twice at the Anaconda prompt to shut down any active Python kernels.
