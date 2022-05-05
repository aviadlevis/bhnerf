from bhnerf import utils
from bhnerf import constants as consts
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
        if np.sqrt(np.sum(rot_axis_prime**2)) < 1e-5: rot_axis_prime = z_axis 
        rot_angle_prime = np.arccos(np.dot(rot_axis, z_axis))
        rot_matrix = utils.rotation_matrix(rot_axis_prime, rot_angle_prime)
        center = np.matmul(rot_matrix, np.append(center_2d, 0.0))
        
    emission = utils.gaussian_xr(resolution, center, std, fov=fov, std_clip=std_clip)
    if normalize: emission /= emission.max()
    
    return emission

def generate_orbit(initial_frame, nt, angular_velocity, rot_axis, tstart, tstop):
    """
    Generate an orbit using a velocity field and initial frame (2D/3D)
    
    Parameters
    ----------
    initial_frame: xr.DataArray
        A DataArray with initial emission.
    nt: int, 
        Number of frames.
    angular_velocity: float or function, 
        Either a homogeneous velocity field (float) or a function of "r"/"radius".
    rot_axis: 3d vector, 
        Rotation axis.
    tstart: float, default=0.0
        Start time of frame zero.
    
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
        warped_coords = velocity_warp(grid, t, angular_velocity, rot_axis, tstart)
        image_coords = np.moveaxis(utils.world_to_image_coords(warped_coords, fov=fov, npix=npix), -1, 0)
        frame_data = skimage.transform.warp(initial_frame, image_coords, mode='constant', cval=0.0, order=3)
        coords = dict([('t', t)] + [(dim, initial_frame[dim]) for dim in dims])
        frame = xr.DataArray(frame_data, dims=dims, coords=coords)
        movie.append(frame.expand_dims('t'))
    movie = xr.concat(movie, dim='t')
    return movie

def velocity_warp(grid, t, angular_velocity, rot_axis, tstart=0.0, use_jax=False):
    """
    Generate an coordinate transoform for the velocity warp.
    
    Parameters
    ----------
    grid: list of np arrays
        A list of arrays with grid coordinates
    t: float, 
        time.
    angular_velocity: float or function, 
        Either a homogeneous velocity field (float) or a function of "r"/"radius".
    rot_axis: 3d vector, 
        Rotation axis.
    tstart: float, default=0.0
        Start time of frame zero.
    use_jax: bool, default=False,
        Using jax enables GPU accelerated computing.
        
    Returns
    -------
    warped_coords: np.array,
        An array with the new coordinates for the warp transformation.
    """
    _np = jnp if use_jax else np
    radius = _np.sqrt(_np.sum(_np.array([dim**2 for dim in grid]), axis=0) + _np.finfo(_np.float32).eps)
    t_relative = (t - tstart)
    # assert t_relative >= 0, 'time t ({}) is before start time ({})'.format(t, tstart)
    
    if _np.isscalar(angular_velocity):
        velocity = angular_velocity

    elif callable(angular_velocity):
        args = {}
        params = signature(angular_velocity).parameters.keys()
        if ('radius' in params): args['radius'] = radius
        if ('theta' in params): args['theta'] = theta
        if ('r' in params): args['r'] = radius
        velocity = angular_velocity(**args)
        
    # Fill NaNs with zeros
    velocity = _np.where(_np.isfinite(velocity), velocity, _np.zeros_like(velocity))
    theta_rot = t_relative * velocity
    
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

def integrate_rays(emission, sensor, doppler_factor=1.0, dim='geo'):
    """
    Integrate emission over rays to get sensor pixel values.
    
    Parameters
    ----------
    emission: xr.DataArray,
        A DataArray with emission values.
    sensor: xr.Dataset
        A Dataset with ray geomeries for integration. 
    doppler_factor: np.array, default=1.0
        Multiply emission by a doppler factor (relativistic beaming) 
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
    inteporlated_values = emission.interp(coords) * doppler_factor
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

def doppler_factor(sensor, rot_axis, angular_velocity):
    """
    Compute the Doppler factor due to relativistic beaming
    
    Parameters
    ----------
    sensor: xr.Dataset
        A Dataset with ray geomeries for integration. 
    rot_axis: 3d vector, 
        Rotation axis of the azimuthal velocity field.
    angular_velocity: float or function, 
        Either a homogeneous velocity field (float) or a function of "r"/"radius".
        
    Returns
    -------
    doppler_factor: np.array
        Array with doppler multiplicative factor.
    """
    
    sensor = sensor.fillna(-1e9)
    coords = {'x': sensor.x, 'y': sensor.y}
    if 'z' in sensor.variables:
        coords['z'] = sensor.z
    radius = sensor.r

    if np.isscalar(angular_velocity):
        linear_velocity = radius * angular_velocity

    elif callable(angular_velocity):
        args = {}
        params = signature(angular_velocity).parameters.keys()
        if ('radius' in params): args['radius'] = radius
        if ('theta' in params): args['theta'] = theta
        if ('r' in params): args['r'] = radius
        linear_velocity = radius * angular_velocity(**args)

    linear_velocity = linear_velocity.fillna(0.0)
    linear_velocity *= (consts.G*consts.sgra.mass/consts.c**2) / 3600.0 # unit conversion to [m/s]
    beta = linear_velocity / consts.c

    # Emission velocity direction vector
    r_vector = np.stack([sensor.x, sensor.y, np.zeros_like(sensor.z)])
    r_vector = r_vector / (np.linalg.norm(r_vector, axis=0) + 1e-8)
    e_vx, e_vy, e_vz = np.cross(rot_axis, r_vector, axis=0)
    doppler_cosine = -sensor.vx*e_vx - sensor.vy*e_vy - sensor.vz*e_vz

    doppler_factor = np.sqrt(1 - beta**2) / (1 - doppler_cosine * beta)
    
    return doppler_factor

def load_sensor(spin, inclination, ngeo=100, num_alpha=64, num_beta=64, distance=1000.0, max_r=6.0):
    """
    Compute sensor rays for a given spin and inclination
    
    Parameters
    ----------
    spin: float
        normalized spin value in range [0,1]
    inclination: float, 
        inclination angle in [rad] in range [0, pi/2]
    ngeo: int, default=100
        Number of points along a ray
    num_alpha: int, default=64,
        Number of pixels in the vertical direction
    num_beta: int, default=64,
        Number of pixels in the horizontal direction   
    distance: float, default=1000.0
        Distance to observer
    max_r: float, default=6.0, 
        maximum radius for integration (in units of GM/c^2)
    
    Returns
    -------
    sensor: xr.Dataset
        A dataset specifying ray pahts for the sensor.
        
    Notes
    -----
    units are in GM/c^2
    """
    from kgeo import raytrace_ana
    
    alpha, beta = np.meshgrid(np.linspace(-8.0, 8.0, num_alpha), np.linspace(-8.0, 8.0, num_beta))
    image_coords = [alpha.ravel(), beta.ravel()]
    observer_coords = [0, distance, inclination, 0]
    sensor = raytrace_ana(spin, observer_coords, image_coords, ngeo+1, plotdata=False).get_dataset()
    sensor = sensor.isel(geo=range(ngeo))
    sensor.attrs.update(num_alpha=num_alpha, num_beta=num_beta)

    # Path direction vector
    vx, vy, vz = vx/piecwise_dist, vy/piecwise_dist, vz/piecwise_dist
    vx = xr.concat((xr.zeros_like(sensor.x.isel(geo=0)), vx), dim='geo').fillna(0.0)
    vy = xr.concat((xr.zeros_like(sensor.y.isel(geo=0)), vy), dim='geo').fillna(0.0)
    vz = xr.concat((xr.zeros_like(sensor.z.isel(geo=0)), vz), dim='geo').fillna(0.0)
    sensor = sensor.assign(vx=vx, vy=vy, vz=vz)

    piecwise_dist = xr.concat((xr.zeros_like(sensor.x.isel(geo=0)), piecwise_dist), dim='geo').fillna(0.0)
    sensor = sensor.assign(deltas=piecwise_dist)
    
    for key, data_var in sensor.data_vars.items():
        if data_var.dims == sensor.r.dims:
            sensor[key] = data_var.where(sensor.r < max_r)
    sensor = sensor.fillna(0.0)
    
    return sensor