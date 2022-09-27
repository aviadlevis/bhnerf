import numpy as np
import xarray as xr
import functools
import math
import jax.numpy as jnp
import matplotlib.pyplot as plt

mse = lambda true, est: float(np.mean((true - est)**2))

psnr = lambda true, est: float(10.0 * np.log10(np.max(true)**2 / mse(true, est)))

normalize = lambda vector: vector / np.sqrt(np.dot(vector, vector))

def linspace_xr(num, start=-0.5, stop=0.5, endpoint=True, units='unitless'):
    """
    Return a DataArray with coordinates spaced over a specified interval in N-dimensions.

    Parameters
    ----------
    num: int or tuple
        Number of grid points in 1D (x) or 2D (x, y) or 3D (x, y, z). 
    start: float
        starting grid point (included in the grid)
    stop: float
        ending grid point (optionally included in the grid)
    endpoint: bool
        Optionally include the stop points in the grid.
    units: str, default='unitless'
        Store the units of the underlying grid.

    Returns
    -------
    grid: xr.DataArray
        A DataArray with coordinates linearly spaced over the desired interval
    """
    dimensions = ['x', 'y', 'z']
    num = np.atleast_1d(num)
    coords = {}
    for i, n in enumerate(num):
        coord = np.linspace(start, stop, n, endpoint=endpoint)
        coords[dimensions[i]] = coord
    grid = xr.Dataset(coords=coords)
    for dim in grid.dims:
        grid[dim].attrs.update(units=units)
    return grid

def gaussian_xr(resolution, center, std,  fov=(1.0, 'unitless'), std_clip=np.inf):
    """
    Generate a Gaussian image as xarray.DataArray.

    Parameters
    ----------
    resolution: int or nd-array,
            Number of (x,y,z)-axis grid points.
    center: int or nd-array,
        Center of the gaussian in coordinates ('x', 'y', 'z')
    std: (stdx, stdy, stdz), or float,
        Gaussian standard deviation in x,y,z directions. If scalar specified isotropic std is used.
    fov: (float, str), default=(1.0, 'unitless')
        Field of view and units. Default is unitless 1.0.
    std_clip: float, default=np.inf
        Clip after this number of standard deviations

    Returns
    -------
    emission: xr.DataArray,
        A DataArray with Gaussian emission.
    """
    if np.isscalar(std): std = (std, std, std)
    if len(resolution) != len(center): raise AttributeError('resolution and center should have same length {} != {}'.format(
        len(resolution), len(center)))
    grid = linspace_xr(resolution, start=-fov[0]/2.0, stop=fov[0]/2.0, units=fov[1])
    if 'x' in grid.dims and 'y' in grid.dims and 'z' in grid.dims:
        data = np.exp(-0.5*( ((grid.x - center[0])/std[0])**2 + ((grid.y - center[1])/std[1])**2 + ((grid.z - center[2])/std[2])**2 ))
        dims = ['x', 'y', 'z']
    elif 'x' in grid.dims and 'y' in grid.dims:
        data = np.exp(-0.5*( ((grid.y - center[1])/std[1])**2 + ((grid.x - center[0])/std[0])**2 ))
        dims = ['y', 'x']
    else:
        raise AttributeError

    threshold = np.exp(-0.5 * std_clip ** 2)
    emission = xr.DataArray(
        name='emission',
        data=data.where(data > threshold).fillna(0.0),
        coords=grid.coords,
        dims=dims,
        attrs={
            'fov': fov,
            'std': std,
            'center': center,
            'std_clip': std_clip
        })
    return emission

def rotation_matrix(axis, angle, use_jax=False):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis
    
    Parameters
    ----------
    axis: list or np.array, dim=3
        Axis of rotation
    angle: float or numpy array of floats,
        Angle of rotation in radians
    use_jax: bool, default=False
        Compuatations using jax.
        
    Returns
    -------
    rotation_matrix: np.array(shape=(3,3,...)),
        A rotation matrix. If angle is a numpy array additional dimensions are stacked at the end.
        
    References
    ----------
    [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
    [2] https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    _np = jnp if use_jax else np
    
    axis = _np.array(axis)
    axis = axis / _np.sqrt(_np.dot(axis, axis))
    
    a = _np.cos(angle / 2.0)
    b, c, d = _np.stack([-ax * _np.sin(angle / 2.0) for ax in axis])
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return _np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def spherical_coords_to_rotation_axis(theta, phi):
    """
    Transform the spherical coordinates into a rotation axis and angle
    
    Parameters
    ----------
    theta: float,
        zenith angle (rad)
    phi: float,
        azimuth angle (rad)
        
    Returns
    -------
    rot_axis: 3-vector,
        Rotation axis.
    rot_angle: float, 
        Rotation angle about the rot_axis.
    """
    z_axis = np.array([0, 0, 1])
    r_vector = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    rot_axis_prime = np.cross(r_vector, z_axis)
    rot_matrix = rotation_matrix(rot_axis_prime,  np.pi/2)
    rot_axis = np.matmul(rot_matrix, r_vector)
    rot_angle = phi
    return rot_axis, rot_angle

def world_to_image_coords(coords, fov, npix, use_jax=False):
    _np = jnp if use_jax else np
    image_coords = []
    for i in range(coords.shape[-1]):
        image_coords.append((coords[...,i] + fov[i]/2.0) / fov[i] * (npix[i] - 1))
    image_coords = _np.stack(image_coords, axis=-1)
    return image_coords

def intensity_to_nchw(intensity, cmap='viridis', gamma=0.5):
    """
    Utility function to converent a grayscale image to NCHW image (for tensorboard logging).
       N: number of images in the batch
       C: number of channels of the image (ex: 3 for RGB, 1 for grayscale...)
       H: height of the image
       W: width of the image

    Parameters
    ----------
    intensity: array,
         Grayscale intensity image.
    cmap : str, default='viridis'
        A registered colormap name used to map scalar data to colors.
    gamma: float, default=0.5
        Gamma correction term
        
    Returns
    -------
    nchw_images: array, 
        Array of images.
    """
    cm = plt.get_cmap(cmap)
    norm_images = ( (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity)) )**gamma
    nchw_images = np.moveaxis(cm(norm_images)[...,:3], (0, 1, 2, 3), (3, 2, 0, 1))
    return nchw_images

def anti_aliasing_filter(image_plane, window):
    """
    Anti-aliasing flitering / blurring
    
    Parameters
    ----------
    image_plane: np.array,
        2D image or 3D movie (frames are in the first index)
    window: np.array
        2D image used for anti-aliasing filtering
    
    Returns
    -------
    image_plane: np.array,
        2D image or 3D movie (frames are in the first index)
    """
    fourier = jnp.fft.fft2(jnp.fft.ifftshift(image_plane, axes=(-2, -1))) * jnp.fft.fft2(jnp.fft.ifftshift(window))
    image_plane = jnp.fft.ifftshift(jnp.fft.ifft2(fourier), axes=(-2, -1)).real
    return image_plane

def expand_dims(x, ndim, axis=0, use_jax=False):
    _np = jnp if use_jax else np
    for i in range(ndim-_np.array(x).ndim):
        x = _np.expand_dims(x, axis=min(axis, _np.array(x).ndim))
    return x