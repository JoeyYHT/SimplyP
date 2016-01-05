# A simple phosphorus model

This repository contains code to run a simple hydrology, sediment and phosphorus model. The model is dynamic, working on a daily time step, and spatially semi-distributed - there is the ability to differentiate between hydrology, sediment and phosphorus processes between land use types and sub-catchments (with associated stream reaches).

The hydrology model builds on that described here: (insert link to James Sample's Github notebook)

The remainder was developed by Leah Jackson-Blake and forms part of her PhD thesis. The work was funded by the RESAS, the Rural and Environment Science and Analytical Services Division of the Scottish Government.

## IPython notebooks

The links at the top of this page will take you to static versions of these notebooks rendered with [nbviewer](http://nbviewer.ipython.org/). However, the notebooks can also be downloaded and run interactively. The following steps should get you started on Windows:

1. You'll need an up-to-date IPython installation. If you don't have one already try [WinPython](http://winpython.sourceforge.net/), which is a comprehensive and portable Python distribution that won't interfere with anything else on your system.<br><br> 

2. Once WinPython is installed, go to one of the notebooks and download the **.ipynb** file to your computer (the "download" icon is at the top-right of the screen).<br><br>

3. Open the folder containing your WinPython installation and run the **WinPython Command Prompt** (not the normal Windows Command Prompt).<br><br>

4. **Change directories** to wherever you saved the **.ipynb** file and then type `ipython notebook` at the command prompt. Your browser should open to display the IPython dashboard and you'll see a link to the notebook you just downloaded.<br><br>

5. Click to open the notebook then choose `Cell > Run All` from the menu bar. Python will import all the necessary modules and run the notebook cells, which might take a few moments.<br><br>

You can then work through the notebook interactively.
