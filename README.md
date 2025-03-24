# Ground-based DC resistivity for permafrost

Author: _Cara James_

## Overview

My project code for EOSC 556 geophysical inversion class March/April 2025. 

This project explores using DC resistivity data from the Canadian Arctic to investigate subsurface permafrost characteristics using 2.5d forward simulations and inversions using various simpeg functions (https://github.com/simpeg/simpeg).

## Table of Contents
- [Installation](#installation)
- [Getting Started](#gettingstarted)
- [Files and folders](#filesandfolders)
- [License](#license)

## Installation
1. Clone the repo
```bash
git clone https://github.com/CaraBGJames/eosc-556-project.git
```
2. Create environment from included yml file
```bash
conda env create -f environment.yml
```
3. Activate environment
```bash
conda activate eosc-556-project
```

## Getting started
🚧 WORK IN PROGRESS CHECK BACK SOON 🚧


## Files and folders

### Files:
**[inversion_testing.ipynb](inversion_testing.ipynb)** - Original Jupyter Notebook used to test simpeg functionality for both the forward and inversion steps. This file has more comments and instructions explaining what each step is doing. This is a **good place to start** to understand the layout of the project before running the `forward_sim_permafrost.py` and `inverse_model_permafrost.py` from command line, or editing the code yourself. This notebook uses the inv_dcr_2d_files data from the simpeg tutorial.

**[simpeg_fwd_dcr_2d.ipynb](simpeg_fwd_dcr_2d.ipynb)** - Complete 2.5d forward simulation tutorial from Simpeg (unedited).\
 https://simpeg.xyz/user-tutorials/fwd-dcr-2d.

**[simpeg_fwd_dcr_2d.ipynb](simpeg_fwd_dcr_2d.ipynb)** - Complete 2.5d  inversion tutorial from Simpeg (unedited).\
https://simpeg.xyz/user-tutorials/inv-dcr-2d 

**[forward_sim_permafrost.py](forward_sim_permafrost.py)** - 🚧 WORK IN PROGRESS CHECK BACK SOON. 🚧

**[inverse_model_permafrost.py](forward_sim_permafrost.py)** - 🚧 WORK IN PROGRESS CHECK BACK SOON. 🚧

**[functions.py](functions.py)** - contains all functions used in the other .py files.

**[environment.yml](environment.yml)** - environment file containing all dependencies needed to run this project.

**[.gitignore](.gitignore)** - plain text file to selectively save/upload files to github.

**[LICENSE.md](LICENSE.md)** - M.I.T. License file (see [License](#license)).

**[test_linear_prob.py](test_linear_prob.py)** - test file originally used for testing linear problem code, in future will contain tests for the files in this project, but currently outdated.

### Folders:
**data** - contains the *.dat DC resistivity data files in res2dinv format that are used to define the survey layout and carry out the inversions.

**inv_dcr_2d_files** - contains files used in Simpeg user tutorial.


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
