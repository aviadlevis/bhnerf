# bhnerf
Gravitationally Lensed Black Hole Emission Tomography using [Neural Radiance Fields (NeRF)](https://www.matthewtancik.com/nerf).

Geodesics (photon trajectories) are computed using [kgeo](https://github.com/achael/kgeo). This raytracing implementation of null geodesics in the Kerr metric uses the formalism of [Gralla and Lupsasca 2019](https://arxiv.org/abs/1910.12881). 


Installation
---


Start a conda virtual environment and add channels
```
conda config --add channels conda-forge
conda create -n jax python=3.9 numpy==1.23.1
conda activate jax
```
Install requirements 
```
pip install -r requirements.txt
```
Install [`xarray`](http://xarray.pydata.org/) and its dependencies
```
conda install -c conda-forge xarray dask netCDF4 bottleneck
```

Clone and install bhnerf with the [kgeo](https://github.com/achael/kgeo) submodule
```
git clone --recurse-submodules https://github.com/aviadlevis/bhnerf.git
cd bhnerf/kgeo
pip install .
cd ../ 
pip install .
```

Install [`eht-imaging`](https://achael.github.io/eht-imaging/)
```
conda install -c conda-forge pynfft requests scikit-image
git clone https://github.com/achael/eht-imaging.git
cd eht-imaging
pip install .
cd ../
```

Getting Started
----
The easiest way to get started is through the jupyter notebooks in the `tutorials` directory.
These notebooks cover both the synthetic data generation (forward) and emission estimation (inverse) methods and procedures. Furthermore, 
basic utility and visualization methods are introduced.


&copy; Aviad Levis, California Institute of Technology, 2022.
