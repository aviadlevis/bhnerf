import utils
import numpy as np
import scipy as sp
import xarray as xr
import skimage.transform
from inspect import signature
import jax.numpy as jnp

def gaussian_2d(ny, nx, std, fwhm=None, fov=(1.0, 'unitless'), center=(0,0), std_clip=np.inf):
    """
    Gaussian image.

    Parameters
    ----------
    ny, nx: int,
            Number of (y/x)-axis grid points.
    std: float, 
        Gaussian standard deviation. Used if fwhm is not specified.
    fwhm: float, optional,
        Gaussian full width half max. Overrides the std parameter.
    fov: (float, str), default=(1.0, 'unitless')
        Field of view and units. Default is unitless 1.0.
    center: tuple, default=(0,0)
        Center of the gaussian in the image coordinates ('y', 'x')
    std_clip: float, default=np.inf
        Clip after this number of standard deviations

    Returns
    -------
    image: xr.DataArray,
        An image DataArray with dimensions ['y', 'x'].
    """
    if fwhm is None:
        fwhm = 2 * np.sqrt(2 * np.log(2)) * std
    else:
        std = fwhm / (2 * np.sqrt(2 * np.log(2)))

    grid = utils.linspace_2d((ny, nx), (-fov[0] / 2.0, -fov[0] / 2.0), (fov[0] / 2.0, fov[0] / 2.0), units=fov[1])

    data = np.exp(-0.5*( (grid.y - center[0])**2 + (grid.x - center[1])**2 ) / std**2)
    threshold = np.exp(-0.5 * std_clip ** 2)
    image = xr.DataArray(
        name='image',
        data=np.array(data.where(data > threshold).fillna(0.0), dtype=np.float64, order='C'),
        coords=grid.coords,
        dims=['y', 'x'],
        attrs={
            'fov': fov,
            'std': std,
            'fwhm': fwhm,
            'center': center,
            'std_clip': std_clip
        })
    return image

def gaussian_3d(nx, ny, nz, std,  fov=(1.0, 'unitless'), center=(0,0,0), std_clip=np.inf):
    """
    Gaussian image.

    Parameters
    ----------
    nx, ny, nz: int,
            Number of (x/y/z)-axis grid points.
    std: (stdx, stdy, stdz), or float,
        Gaussian standard deviation in x,y,z directions. If scalar specified isotropic std is used.
    fov: (float, str), default=(1.0, 'unitless')
        Field of view and units. Default is unitless 1.0.
    center: tuple, default=(0,0,0)
        Center of the gaussian in the image coordinates ('x', 'y', 'z')
    std_clip: float, default=np.inf
        Clip after this number of standard deviations

    Returns
    -------
    image: xr.DataArray,
        An image DataArray with dimensions ['x', 'y', 'z'].
    """
    if np.isscalar(std): std = (std, std, std)
    if len(std) != 3: raise AttributeError('std should be either a scalar or or length 3')
    
    start = (-fov[0] / 2.0, -fov[0] / 2.0, -fov[0] / 2.0)
    stop = (fov[0] / 2.0, fov[0] / 2.0, fov[0] / 2.0)
    grid = utils.linspace_3d((nx, ny, nz), start, stop, units=fov[1])
    
    data = np.exp(-0.5*( ((grid.x - center[0])/std[0])**2 + ((grid.y - center[1])/std[1])**2 + ((grid.z - center[2])/std[2])**2 ))
    threshold = np.exp(-0.5 * std_clip ** 2)
    image = xr.DataArray(
        name='emission',
        data=data.where(data > threshold).fillna(0.0),
        coords=grid.coords,
        dims=['x', 'y', 'z'],
        attrs={
            'fov': fov,
            'std': std,
            'center': center,
            'std_clip': std_clip
        })
    return image

def generate_hotspots_2d(ny, nx, theta, orbit_radius, std, r_isco=3, fov=(10.0, 'GM/c^2'), std_clip=3, normalize=True):
    
    if np.any(orbit_radius < r_isco):
        raise AttributeError('hotspot center ({}) is is within r_isco: {}'.format(orbit_radius, r_isco))
        
    thetas = np.atleast_1d(theta)
    orbit_radii = np.atleast_1d(orbit_radius)
    stds = np.atleast_1d(std)
    initial_frame = np.zeros([ny, nx])
    
    for theta, orbit_radius, std in zip(thetas, orbit_radii, stds):
        x, y = orbit_radius * np.array([np.cos(theta), np.sin(theta)])
        hotspot = gaussian_2d(ny, nx, std, fov=fov, center=(y, x), std_clip=std_clip)
        initial_frame += hotspot
    if normalize:
        initial_frame /= initial_frame.max()
        
    return initial_frame

def generate_hotspots_3d(nx, ny, nz, theta, phi, orbit_radius, std, r_isco=3, fov=(10.0, 'GM/c^2'), std_clip=3, normalize=True):
    
    if np.any(orbit_radius < r_isco):
        raise AttributeError('hotspot center ({}) is is within r_isco: {}'.format(orbit_radius, r_isco))
        
    thetas = np.atleast_1d(theta)
    phis = np.atleast_1d(phi)
    orbit_radii = np.atleast_1d(orbit_radius)
    stds = np.atleast_1d(std)
    initial_frame = np.zeros([nx, ny, nz])
    
    for theta, phi, orbit_radius, std in zip(thetas, phis, orbit_radii, stds):
        x, y, z = orbit_radius * np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
        hotspot = gaussian_3d(nx, ny, nz, std, fov=fov, center=(x, y, z), std_clip=std_clip)
        initial_frame += hotspot
    if normalize:
        initial_frame /= initial_frame.max()
        
    return initial_frame
    
def generate_orbit_2d(initial_frame, nt, velocity_field):
    movie = []
    fovx = (initial_frame.x.max() - initial_frame.x.min()).data
    fovy = (initial_frame.y.max() - initial_frame.y.min()).data
    npix = (initial_frame.y.size, initial_frame.x.size)
    y, x = np.meshgrid(initial_frame.y, initial_frame.x, indexing='ij')
    for t in np.linspace(0.0, 1.0, nt):
        warped_coords = velocity_warp_2d(x, y, t, velocity_field)
        image_coords = np.moveaxis(utils.world_to_image_coords(warped_coords, fov=(fovy, fovx), npix=npix), -1, 0)
        frame_data = skimage.transform.warp(initial_frame, image_coords, mode='constant', cval=0.0)
        frame = xr.DataArray(frame_data, dims=['y', 'x'], coords={'t': t, 'y': initial_frame.y, 'x':initial_frame.x})
        movie.append(frame.expand_dims('t'))
    movie = xr.concat(movie, dim='t')
    return movie

def generate_orbit_3d(initial_frame, nt, velocity_field, rot_axis):
    movie = []
    fovx = (initial_frame.x.max() - initial_frame.x.min()).data
    fovy = (initial_frame.y.max() - initial_frame.y.min()).data
    fovz = (initial_frame.z.max() - initial_frame.z.min()).data
    npix = (initial_frame.x.size, initial_frame.y.size, initial_frame.z.size)
    x, y, z = np.meshgrid(initial_frame.x, initial_frame.y, initial_frame.z, indexing='ij')
    for t in np.linspace(0.0, 1.0, nt):
        warped_coords = velocity_warp_3d(x, y, z, t, velocity_field, rot_axis)
        image_coords = np.moveaxis(utils.world_to_image_coords(warped_coords, fov=(fovx, fovy, fovz), npix=npix), -1, 0)
        frame_data = skimage.transform.warp(initial_frame, image_coords, mode='constant', cval=0.0, order=3)
        frame = xr.DataArray(frame_data, dims=['x', 'y', 'z'], 
                             coords={'t': t, 'x':initial_frame.x, 'y': initial_frame.y, 'z': initial_frame.z})
        movie.append(frame.expand_dims('t'))
    movie = xr.concat(movie, dim='t')
    return movie



def rotate_3d(volume, axis, angle):
    output = generate_orbit_3d(volume, 2, angle/(2*np.pi), axis).isel(t=1)
    return output

def velocity_warp_2d(x, y, t, velocity_field, use_jax=False):

    _np = jnp if use_jax else np
    
    radius = _np.sqrt(x**2 + y**2) 
    theta = _np.arctan2(y, x)
    
    if _np.isscalar(velocity_field):
        velocity = velocity_field
        
    elif callable(velocity_field):
        args = {}
        params = signature(velocity_field).parameters.keys()
        if ('radius' in params): args['radius'] = radius
        if ('theta' in params): args['theta'] = theta
        if ('r' in params): args['r'] = radius
        if ('x' in params): args['x'] = x
        if ('y' in params): args['y'] = y
        velocity = velocity_field(**args)
        
    elif isinstance(velocity_field, xr.DataArray):
        velocity = velocity_field.interp(x=xr.DataArray(x).fillna(0.0), y=xr.DataArray(y).fillna(0.0)).data
  
    # Fill NaNs with zeros
    velocity = _np.where(_np.isfinite(velocity), velocity, _np.zeros_like(velocity))
    theta_rot = theta - 2 * _np.pi * t * velocity

    x_rot = radius * _np.cos(theta_rot)
    y_rot = radius * _np.sin(theta_rot)
    warped_coords = _np.stack([y_rot, x_rot], axis=-1)
    return warped_coords

def velocity_warp_3d(x, y, z, t, velocity_field, rot_axis, tstart=0.0, tstop=1.0, use_jax=False):

    _np = jnp if use_jax else np
    
    radius = _np.sqrt(x**2 + y**2 + z**2) 
    t_unitless = (t - tstart) / (tstop - tstart)
    
    if _np.isscalar(velocity_field):
        velocity = velocity_field

    elif callable(velocity_field):
        args = {}
        params = signature(velocity_field).parameters.keys()
        if ('radius' in params): args['radius'] = radius
        if ('theta' in params): args['theta'] = theta
        if ('r' in params): args['r'] = radius
        if ('x' in params): args['x'] = x
        if ('y' in params): args['y'] = y
        velocity = velocity_field(**args)
        
    # Fill NaNs with zeros
    velocity = _np.where(_np.isfinite(velocity), velocity, _np.zeros_like(velocity))
    theta_rot = 2 * _np.pi * t_unitless * velocity
    rot_matrix = utils.rotation_matrix(rot_axis, theta_rot, use_jax=use_jax)
    coords = _np.stack((x, y, z))
    if (rot_matrix.ndim == 2):
        warped_coords = np.einsum('ij,{}'.format('jklm'[:coords.ndim]), rot_matrix, coords)
    else:
        warped_coords = _np.sum(rot_matrix * coords, axis=1)

    warped_coords = _np.moveaxis(warped_coords, 0, -1)
    return warped_coords

def integrate_rays(medium, sensor, dim='geo'):
    sensor = sensor.fillna(-1e9)
    coords = {'x': sensor.x, 'y': sensor.y}
    if 'z' in medium.dims:
        coords['z'] = sensor.z
    inteporlated_values = medium.interp(coords)
    pixels = (inteporlated_values.fillna(0.0) * sensor.deltas).sum(dim)
    return pixels