# Ground-based DC resistivity for permafrost

Author: _Cara James_

## Overview

My project code for EOSC 556 geophysical inversion class March/April 2025. 

This project explores using DC resistivity data from the Canadian Arctic to investigate subsurface permafrost characteristics using 2.5d forward simulations and inversions using various simpeg functions (https://github.com/simpeg/simpeg).

## Table of Contents
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Files and folders](#files-and-folders)
- [License](#license)

## Installation
1. Clone the repo
```bash
git clone https://github.com/CaraBGJames/eosc-556-project.git
```
2. Open environment.yml file and comment out corresponding line for your windows/mac machine
```bash
# uncomment the next line if you are on an intel platform
  # - pydiso # if on intel pc
# uncomment this line if you want to install mumps solvers (mac)
  - python-mumps
```

3. Enter the repository folder ```cd eosc-556-project```, then create environment from included yml file
```bash
conda env create -f environment.yml
```
3. Activate environment
```bash
conda activate eosc-556-project
```

## Getting started
1. Open the jupyter notebook, follow along and run cells to see how it works.
```bash
jupyter notebook inversion_testing_permafrost.ipynb
```
2. 🚧 WORK IN PROGRESS 🚧

## Files and folders

### Files:
**[inversion_testing_permafrost.ipynb](inversion_testing_permafrost.ipynb)** - Original Jupyter Notebook used to test simpeg functionality for both the forward and inversion steps. This file has more comments and instructions explaining what each step is doing. This is a **good place to start** to understand the layout of the project before running the `forward_sim_permafrost.py` and `inverse_model_permafrost.py` from command line, or editing the code yourself. 

**[forward_sim_permafrost.py](forward_sim_permafrost.py)** - 🚧 WORK IN PROGRESS CHECK BACK SOON. 🚧

**[inverse_model_permafrost.py](inverse_model_permafrost.py)** - 🚧 WORK IN PROGRESS CHECK BACK SOON. 🚧

**[functions.py](functions.py)** - contains all functions used in the other .py files.

**[environment.yml](environment.yml)** - environment file containing all dependencies needed to run this project.

**[.gitignore](.gitignore)** - plain text file to selectively save/upload files to github.

**[LICENSE](LICENSE)** - M.I.T. License file (see [License](#license)).

**[test_linear_prob.py](test_linear_prob.py)** - test file originally used for testing linear problem code, in future will contain tests for the files in this project, but currently outdated.

### Folders:
**data** - contains the *.dat DC resistivity data files in res2dinv format that are used to define the survey layout and carry out the inversions.

**.github/workflows** - contains files to carry out automatic testing of code when uploaded to github

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
