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

Next, make sure you have (Mini)Conda installed (download links found [here](https://docs.anaconda.com/miniconda/)) to handle virtual environments in Python. To install the environment used to run the code for hydrological processes, open your CLI (or Anaconda prompt) and type:

```
conda env create -f Hydrologische_processen/environment.yml
```
This ensures that the correct version of Python is used in your CLI. Next, to activate the environment, run in your CLI (or Anaconda prompt):

```
conda activate hydroprocess_env
```