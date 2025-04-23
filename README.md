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
If you like, you can open the jupyter notebook, follow along and run cells to see how it works.
```bash
jupyter notebook inversion_testing_permafrost.ipynb
```

Alternatively, you can just run the invert_synth_data.py or invert_real_data.py scripts directly from your terminal using
```bash
python invert_real_data.py
```

Enjoy!

## Files and folders

### Files:
**[testing_synthetic_inversion.ipynb](testing_synthetic_inversion.ipynb)** - Original Jupyter Notebook used to test simpeg functionality for both the forward and inversion steps. This file has more comments and instructions explaining what each step is doing. This is a **good place to start** to understand the layout of the project before running the `invert_synth_data.py` and `invert_real_data.py` from command line, or editing the code yourself. However, please note that this notebook is not the latest and neatest version of the code.

**[invert_synth_data.py](invert_synth_data.py)** - Python file that when called reads the survey layout from the supplied res2dat file, builds a model (defined in utils.py), forward synthesized the data from the model, then adds noise and inverts this 'synthetic' observed data to test my inversion parameters. Final version.

**[invert_real_data.py](invert_real_data.py)** - Python file that reads in the res2dat survey layout and observed data and inverts for subsurface structure. Final version.

**[utils.py](utils.py)** - contains all functions used in the other .py files.

**[environment.yml](environment.yml)** - environment file containing all dependencies needed to run this project.

**[.gitignore](.gitignore)** - plain text file to selectively save/upload files to github.

**[LICENSE](LICENSE)** - M.I.T. License file (see [License](#license)).

**[test_linear_prob.py](test_linear_prob.py)** - test file originally used for testing linear problem code, the goal was it would contain tests for the files in this project, but currently outdated.

### Folders:
**data** - contains the *.dat DC resistivity data files in res2dinv format that are used to define the survey layout and carry out the inversions.

**.github/workflows** - contains files to carry out automatic testing of code when uploaded to github

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
