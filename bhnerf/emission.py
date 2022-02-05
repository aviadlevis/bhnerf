from bhnerf import utils
import numpy as np
import scipy as sp
import xarray as xr
import skimage.transform
from inspect import signature
import jax.numpy as jnp

def generate_hotspot_xr(resolution, rot_axis, rot_angle, orbit_radius, std, r_isco, fov=(10.0, 'GM/c^2'), std_clip=np.inf, normalize=True):
    """
    Generate an emission hotspot as a Gaussian xarray.DataArray.

    Parameters
    ----------
    resolution: int or nd-array,
        Number of (x,y,z)-axis grid points.
    rot_axis: 3d array/list/tuple,
        The orbit rotation axis along which
    rot_angle: float, 
        The angle along the (2d) circular orbit. 
    orbit_radius: float, 
        Radius of the orbit.
    std: (stdx, stdy, stdz), or float,
        Gaussian standard deviation in x,y,z directions. If scalar specified isotropic std is used.
    r_isco: float,
        Radius of the inner most stable circular orbit (ISCO). 
    fov: (float, str), default=(1.0, 'unitless')
        Field of view and units. Default is unitless 1.0.
    std_clip: float, default=np.inf
        Clip after this number of standard deviations
    normalize: bool, default=True,
        If True, normalize the maximum flux to 1.0. 
        
    Returns
    -------
    emission: xr.DataArray,
        A DataArray with Gaussian emission.
    """
    if orbit_radius < r_isco:
        raise AttributeError('hotspot center ({}) is is within r_isco: {}'.format(orbit_radius, r_isco))
    center_2d = orbit_radius * np.array([np.cos(rot_angle), np.sin(rot_angle)])
    if len(resolution) == 2:
        center = center_2d
    else:
        rot_axis = np.array(rot_axis) 
        rot_axis = rot_axis / np.sqrt(np.sum(rot_axis**2))
        z_axis = np.array([0, 0, 1])
        rot_axis_prime = np.cross(z_axis, rot_axis)
        rot_angle_prime = np.arccos(np.dot(rot_axis, z_axis))
        rot_matrix = utils.rotation_matrix(rot_axis_prime, rot_angle_prime)
        center = np.matmul(rot_matrix, np.append(center_2d, 0.0))
        
    emission = utils.gaussian_xr(resolution, center, std, fov=fov, std_clip=std_clip)
    if normalize: emission /= emission.max()
    
    return emission

def generate_orbit(initial_frame, nt, velocity_field, rot_axis, tstart=0.0, tstop=1.0):
    """
    Generate an orbit using a velocity field and initial frame (2D/3D)
    
    Parameters
    ----------
    initial_frame: xr.DataArray
        A DataArray with initial emission.
    nt: int, 
        Number of frames.
    velocity_field: float or function, 
        Either a homogeneous velocity field (float) or a function of "r"/"radius".
    rot_axis: 3d vector, 
        Rotation axis.
    tstart: float, default=0.0
        Start time of frame zero.
    tstop: float, default=1.0
        Stop time of frame nt.
    
    Returns
    -------
    movie: xr.DataArray,
        A movie of a temporally propogated emission field.
    """
    movie = []
    if initial_frame.ndim == 2:
        dims = ['y', 'x']
    elif initial_frame.ndim == 3: 
        dims = ['x', 'y', 'z']
    else:
        raise AttributeError('number of dimensions of initial_frame ({}) not supported'.format(initial_frame.ndim))

    fov = [(initial_frame[dim].max() - initial_frame[dim].min()).data for dim in dims]
    npix = [initial_frame[dim].size for dim in dims]
    grid = np.meshgrid(*[initial_frame[dim] for dim in dims], indexing='ij')
    for t in np.linspace(tstart, tstop, nt):
        warped_coords = velocity_warp(grid, t, velocity_field, rot_axis, tstart, tstop)
        image_coords = np.moveaxis(utils.world_to_image_coords(warped_coords, fov=fov, npix=npix), -1, 0)
        frame_data = skimage.transform.warp(initial_frame, image_coords, mode='constant', cval=0.0, order=3)
        coords = dict([('t', t)] + [(dim, initial_frame[dim]) for dim in dims])
        frame = xr.DataArray(frame_data, dims=dims, coords=coords)
        movie.append(frame.expand_dims('t'))
    movie = xr.concat(movie, dim='t')
    return movie

def velocity_warp(grid, t, velocity_field, rot_axis, tstart=0.0, tstop=1.0, use_jax=False):
    """
    Generate an coordinate transoform for the velocity warp.
    
    Parameters
    ----------
    grid: list of np arrays
        A list of arrays with grid coordinates
    t: float, 
        time.
    velocity_field: float or function, 
        Either a homogeneous velocity field (float) or a function of "r"/"radius".
    rot_axis: 3d vector, 
        Rotation axis.
    tstart: float, default=0.0
        Start time of frame zero.
    tstop: float, default=1.0
        Stop time of frame nt.
    use_jax: bool, default=False,
        Using jax enables GPU accelerated computing.
        
    Returns
    -------
    warped_coords: np.array,
        An array with the new coordinates for the warp transformation.
    """
    _np = jnp if use_jax else np
    radius = _np.sqrt(_np.sum(_np.array([dim**2 for dim in grid]), axis=0))
    t_unitless = (t - tstart) / (tstop - tstart)
    
    if _np.isscalar(velocity_field):
        velocity = velocity_field

    elif callable(velocity_field):
        args = {}
        params = signature(velocity_field).parameters.keys()
        if ('radius' in params): args['radius'] = radius
        if ('theta' in params): args['theta'] = theta
        if ('r' in params): args['r'] = radius
        velocity = velocity_field(**args)
        
    # Fill NaNs with zeros
    velocity = _np.where(_np.isfinite(velocity), velocity, _np.zeros_like(velocity))
    theta_rot = 2 * _np.pi * t_unitless * velocity
    
    if len(grid) == 2:
        inv_rot_matrix = _np.array([[_np.cos(theta_rot), -_np.sin(theta_rot)], 
                                    [_np.sin(theta_rot), _np.cos(theta_rot)]])
    elif len(grid) == 3:
        inv_rot_matrix = utils.rotation_matrix(rot_axis, -theta_rot, use_jax=use_jax)
    else:
        raise AttributeError('grid dimensions not supported')
        
    coords = _np.stack(grid)
    if (inv_rot_matrix.ndim == 2):
        warped_coords = _np.einsum('ij,{}'.format('jklm'[:coords.ndim]), inv_rot_matrix, coords)
    else:
        warped_coords = _np.sum(inv_rot_matrix * coords, axis=1)
    warped_coords = _np.moveaxis(warped_coords, 0, -1)
    
    return warped_coords

def integrate_rays(emission, sensor, dim='geo'):
    """
    Integrate emission over rays to get sensor pixel values.
    
    Parameters
    ----------
    emission: xr.DataArray,
        A DataArray with emission values.
    sensor: xr.Dataset
        A Dataset with ray geomeries for integration. 
    dim: str, default='geo'
        Name of the dimension along which integration is carried out. 
        The default corresponds to the dimension name used when generating a sensor.

    Returns
    -------
    pixels: xr.DataArray,
        An DataArray with pixel values (e.g. fluxes).
    """
    sensor = sensor.fillna(-1e9)
    coords = {'x': sensor.x, 'y': sensor.y}
    if 'z' in emission.dims:
        coords['z'] = sensor.z
    inteporlated_values = emission.interp(coords)
    pixels = (inteporlated_values.fillna(0.0) * sensor.deltas).sum(dim)
    return pixels

def zero_unsupervised_emission(emission, coords, rmin=0, rmax=np.Inf):
    """
    Zero emission that is not within the supervision region
    
    Parameters
    ----------
    emission: np.array
        3D array with emission values
    coords: list of np.arrays
        Spatial coordinate arrays each shaped like emission
    rmin: float, default=0
        Zero values at radii < rmin
    rmax: float, default=np.inf
        Zero values at radii > rmax
        
    Returns
    -------
    emission: np.array
        3D array with zeroed down emission values
    """
    emission = np.squeeze(emission)
    r = np.sum([np.squeeze(x)**2 for x in coords], axis=0)
    emission = jnp.where(r < rmin**2, jnp.zeros_like(emission), emission)
    emission = jnp.where(r > rmax**2, jnp.zeros_like(emission), emission)
    return emission