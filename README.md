# A simple phosphorus model

This repository contains code to run a simple hydrology, sediment and phosphorus model. The model is dynamic, working on a daily time step, and spatially semi-distributed - there is the ability to differentiate between hydrology, sediment and phosphorus processes between land use types and sub-catchments (with associated stream reaches).

The hydrology model builds on that described here: https://github.com/JamesSample/enviro_mod_notes.git

The remainder was developed by Leah Jackson-Blake and forms part of her PhD thesis. The work was funded by the RESAS, the Rural and Environment Science and Analytical Services Division of the Scottish Government.

Please report any bugs or errors by submitting a pull request, or by email to ljb at niva.no

## Running the model

A static version of the model can be viewed with [nbviewer](http://nbviewer.ipython.org/). However, to run the model it needs to be downloaded. The following steps should get you started on Windows:

1. If you don't have an up-to-date IPython installation, a good option is [WinPython](http://winpython.sourceforge.net/), which is a comprehensive and portable Python distribution that won't interfere with anything else on your system. Model development was carried out using Python 2.7.<br>

2. Once WinPython is installed, click on the download icon at the top right of the screen, to save the model and some example files to your computer.<br>

3. On your computer, open the folder containing your WinPython installation and double click on the **WinPython Command Prompt** to open it (not the normal Windows Command Prompt).<br>

4. Within the WinPython command window, **Change directories** to wherever you saved the folder containing the model. This is done using normal command line syntax (e.g. cd C:\Working\NewModelFolder). Then type `ipython notebook` at the command prompt. Your browser should open to display the IPython dashboard and you'll see a link to the notebook you just downloaded. **Do not close the WinPython Command Prompt**.<br>

5. Click on the SimplyP.ipynb file to open the notebook containing the model.<br>

6. When you have finished running the model and looking at any results in the browser, click 'File > Close and Halt'. The browser can then be closed. In the WinPython Command Prompt Window type 'Ctrl C' to shut down any Python kernels which may still be open, and then close this too.


Further instructions for running the model are provided at the top of the notebook containing the model
