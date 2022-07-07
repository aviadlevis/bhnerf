# bhnerf
Gravitationally Lensed Black Hole Emission Tomography using [Neural Radiance Fields (NeRF)](https://www.matthewtancik.com/nerf).

Geodesics (photon trajectories) are computed using [kgeo](https://github.com/achael/kgeo). This raytracing implementation of null geodesics in the Kerr metric uses the formalism of [Gralla and Lupsasca 2019](https://arxiv.org/abs/1910.12881). 


Installation
---

Clone bhnerf repository with the [kgeo](https://github.com/achael/kgeo) submodule
```
git clone --recurse-submodules https://github.com/aviadlevis/bhnerf.git
cd bhnerf
```

Start a conda virtual environment and add channels
```
conda create -n bhnerf
conda activate bhnerf
```
If not added already add `conda-forge` and `anaconda` channels
```
conda config --add channels conda-forge
conda config --add channels anaconda
```
Install requirements 
```
conda install --file requirements.txt
```
Install [`xarray`](http://xarray.pydata.org/) and its dependencies
```
conda install -c conda-forge xarray dask netCDF4 bottleneck
```
Install [`jax`](https://github.com/google/jax)
```
pip install --upgrade pip
pip install --upgrade jax jaxlib>=0.1.69+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
Note that in the line above the number next to cude should be replaced with version of the existing CUDA installation (see [GitHub issue](https://github.com/google/jax/issues/5231)), for example CUDA10.1 --> cuda101. You can find your CUDA version using `nvcc --version`

Install [`eht-imaging`](https://achael.github.io/eht-imaging/)
```
conda install -c conda-forge pynfft requests scikit-image
git clone https://github.com/achael/eht-imaging.git
cd eht-imaging
pip install .
cd ../
``` 

Install `bhnerf`
```
pip install .
```
Note currently [`kgeo`](https://github.com/achael/kgeo) requires an experimental [`scipy`](https://scipy.github.io/devdocs/release.1.8.0.html) version which has [elliptic integrals](https://scipy.github.io/devdocs/reference/special.html#module-scipy.special) implemented
```
pip install scipy==1.8.0rc4
```

Getting Started
----
The easiest way to get started is through the jupyter notebooks in the `tutorials` directory.
These notebooks cover both the synthetic data generation (forward) and emission estimation (inverse) methods and procedures. Furthermore, 
basic utility and visualization methods are introduced.


&copy; Aviad Levis, California Institute of Technology, 2022.
