# Teaching
This repository contains all code for the practicals given by RSDA. Note that the code and data is distributed in subfolders per course.
The following courses are currently documented in this repo:
    
    - Hydrological Processes

## Practicals Hydrological Processes
The folder `Hydrologische_processen` contains the assignment and processed data for the computer practicals of the course [Hydrological Processes](https://onderwijsaanbod.kuleuven.be/syllabi/n/I0J56A). Note that repository contains the solutions for the practical, so it should not be distributed directly to the students.

### Installation instructions (local setup) 
First, make a local copy of this repository using

```
git clone https://github.com/KUL-RSDA/teaching.git
```

or download as zip file and unzip. In each case, make sure to navigate inside the `teaching` folder before executing any of the command line interface (CLI) instructions below.

Next, make sure you have (Mini)Conda installed (download links found [here](https://docs.anaconda.com/miniconda/)) to handle virtual environments in Python. For Linux and MacOS users: open a regular terminal (CLI); Windows users: make sure you open a command prompt CLI and not the (windows) powershell one. In Visual Studio Code (VScode), you can open a terminal as follows: `Terminal -> new Termial`. Windows users can then open a command prompt by pressing the `+`-bottom (next to powershell) and selecting the Command prompt. Alternatively, you can use the Anaconda prompt CLI if you have Anaconda installed. Once you have opened the proper CLI, type: 

```
conda env create -f Hydrologische_processen/environment.yml
```

This ensures that the correct version of Python is used in your CLI. Next, to activate the environment, run in your CLI:

```
conda activate hydroprocess_env
```

Finally, when running a notebook, you can select the interpreter (virtual environment) as follows: `Ctrl+shift+P -> type: select interpreter -> select the hydroprocess_env`. 