# 867-team-gore

# Generative models for heterogeneous materials

Implementations of generative models for structure generation of heterogeneous materials.

## Dependencies
* numpy
* scipy
* scikit-image
* imageio
* pillow
* pytorch
* torchvision
* pytorch-lightning


## Setup
This code is written in python and is built on top of the pytorch library. You will need to have these installed
in order to use the code. We recommend using a package management system such as `conda` or `poetry` to setup a
virtual environment for this project.


We provide a script to create such an environment using `conda` in the developer folder. This script will install 
all required dependencies. After cloning the repository please follow the following steps to setup the environment:

1. Open a terminal
2. Navigate to the developer folder with your lpdgen clone

   `> cd path_to_lpdgen_clone\lpdgen\developer`

3. The setup_environment scripts accept the location of your conda installation
   as a command line arguments. See following instructions for examples.

** Windows users **

4. Run the script. The script accepts two command line arguments a) path to your conda install
   and b) device option which is either `cpu` or `gpu`.

```
> setup_environments.bat <path_to_conda_install> <device_option>
```

For example

```
> setup_environments.bat "C:\Users\vvenkate\Software\anaconda" cpu
```

** Linux/Mac users

4. Run the script.  The script accepts two command line arguments a) path to your conda install 
   and b) device option which is either `cpu` or `gpu`
   
```
> sh setup_environments.sh <path_to_conda> <device_option>
```

For example

```
> setup_environments.sh ~/Applications/anaconda cpu
```


Once the environment is set up, you can install either a developer or user version of library into 
your environment. The developer install will track changes you might make to the lpdgen code, versus 
the user version does not.

1. Activate the environment
2. Navigate to your clone of the lpdgen repository
3. Run the command for your installation type:

   **Developers** - `pip install -e .`

   **Users** - `pip install .`

   The developer installation will ensure that any changes that you make to 
   the lpdgen code will automatically be picked up by conda. Note that this
   is accomplished by linking the package within the conda environment to your
   local copy of lpdgen. If you move your folder, you will need to run this
   command again.


## Training Data
High-resolution CT scan data of porous media rocks is publicly available via
the [Digital Rocks Project](https://www.digitalrocksportal.org/). 


## Contributing
We would love contributions from the user community to `lpdgen`. The user community
is what makes a great tool. Please take a look at our open issues to see where we can
use the help. If you have new ideas, please do open an issue documenting the idea.
We use a simple `feature-branch-workflow` model. You will need direct access to the repo.
Please contact the maintainers for access.


## Acknowledgements
* Pytorch tutorial on [DCGAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

