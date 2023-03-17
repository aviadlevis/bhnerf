from bhnerf import utils
from bhnerf import constants as consts
from bhnerf import kgeo
import numpy as np
import xarray as xr
import jax.numpy as jnp
from astropy import units
import scipy.ndimage

def generate_hotspot_xr(resolution, rot_axis, rot_angle, orbit_radius, std, r_isco, fov, std_clip=np.inf, normalize=True):
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
    if normalize: emission /= emission.integrate(['x','y','z'])
    emission.attrs.update(
        rot_axis=rot_axis
    )
    return emission

def generate_tube_xr(resolution, rot_axis, phi_start, phi_end, orbit_radius, std, r_isco, fov, std_clip=np.inf, normalize=True):
    """
    Generate an emission tube with a Gaussian profile as an xarray.DataArray.

    Parameters
    ----------
    resolution: int or nd-array,
        Number of (x,y,z)-axis grid points.
    rot_axis: 3d array/list/tuple,
        The orbit rotation axis along which
    phi_start: float, 
        The start angle for the tube.
    phi_end: float, 
        The end angle for the tube.
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
        A DataArray with emission.
    """
    if orbit_radius < r_isco:
        raise AttributeError('hotspot center ({}) is is within r_isco: {}'.format(orbit_radius, r_isco))

    rot_axis = np.array(rot_axis) 
    rot_axis = rot_axis / np.sqrt(np.sum(rot_axis**2))
    z_axis = np.array([0, 0, 1])
    rot_axis_prime = np.cross(z_axis, rot_axis)
    if np.sqrt(np.sum(rot_axis_prime**2)) < 1e-5: rot_axis_prime = z_axis 
    rot_angle_prime = np.arccos(np.dot(rot_axis, z_axis))
    rot_matrix = utils.rotation_matrix(rot_axis_prime, rot_angle_prime)
        
    emission = 0
    angles = np.arange(phi_start, phi_end, 0.015)
    for phi in angles:
        center_2d = orbit_radius * np.array([np.cos(phi), np.sin(phi)]) 
        center = np.matmul(rot_matrix, np.append(center_2d, 0.0))
        emission += utils.gaussian_xr(resolution, center, std, fov=fov, std_clip=std_clip)
    if normalize: emission /= emission.integrate(['x','y','z'])
    emission.attrs.update(
        rot_axis=rot_axis,
        phi_start=phi_start, 
        phi_end=phi_end
    )
    return emission
def equatorial_ring(geos, mbar):
    """
    Equatorial ring of emission
    
    Parameters
    ----------
    geos: xr.Dataset
        Dataset with Kerr geodesics
    mbar: int,
        Order of the photon ring (number of loops). Direct emission = 0.
        
    Returns
    -------
    emission: xr.DataArray,
        A DataArray with emission set to 1.0 along the equatorial ring coordinates.
    """
    emission = xr.zeros_like(geos.mino)
    mino_times = [kgeo.equatorial_lensing.r_equatorial(float(geos.spin), np.inf, float(geos.inc), mbar, float(alpha), float(beta))[1] 
                  for alpha, beta in zip(geos.alpha, geos.beta)]
    mino_times = np.concatenate(mino_times)[:,None]
    geo_indices = np.abs(geos.mino - mino_times).argmin('geo')
    emission.loc[{'geo': geo_indices}] = 1.0
    return emission

def velocity_warp_coords(coords, Omega, t_frames, t_start_obs, t_geos, t_injection, rot_axis=[0,0,1], M=consts.sgra_mass, t_units=None, use_jax=False):
    """
    Generate an coordinate transoform for the velocity warp.
    
    Parameters
    ----------
    coords: list of np arrays
        A list of arrays with grid coordinates
    Omega: array, 
        Angular velocity array sampled along the coords points.
    t_frames: array, 
        Array of time for each image frame with astropy.units
    t_start_obs: astropy.Quantity, default=None
        Start time for observations, if None t_frames[0] is assumed to be start time.
    t_geos: array, 
        Time along each geodesic (ray). This is used to account for slow light (light travels at finite velocity).
    t_injection: float, 
        Time of hotspot injection in M units.
    rot_axis: array, default=[0, 0, 1]
        Currently only equitorial plane rotation is supported
    M: astropy.Quantity, default=constants.sgra_mass,
        Mass of the black hole used to convert frame times to space-time times in units of M
    t_units: astropy.units, default=None,
        Time units. If None units are taken from t_frames.
    use_jax: bool, default=False,
        Using jax enables GPU accelerated computing.
        
    Returns
    -------
    warped_coords: array,
        An array with the new coordinates for the warp transformation.
    """
    _np = jnp if use_jax else np
    coords = _np.array(coords)
    Omega = _np.array(Omega)
    
    if isinstance(t_start_obs, units.Quantity):
        t_units = t_start_obs.unit
        t_start_obs = t_start_obs.value
    
    GM_c3 = 1.0  
    if t_units is not None:
        GM_c3 = consts.GM_c3(M).to(t_units).value

    if isinstance(t_frames, units.Quantity):
        t_frames = t_frames.to(t_units).value
    t_frames = _np.array(t_frames)

    if (_np.isscalar(Omega) or Omega.ndim == 0):
        Omega = utils.expand_dims(Omega, coords.ndim-1, axis=-1, use_jax=use_jax)

    # Extend the dimensions of `t_frames` and `coords' for an array of times 
    if not (t_frames.ndim == 0):
        coords = utils.expand_dims(coords, coords.ndim + t_frames.ndim, 1, use_jax)
        t_frames = utils.expand_dims(t_frames, t_frames.ndim + Omega.ndim, -1, use_jax)

    # Convert time units to grid units
    t_geos = (t_frames - t_start_obs)/GM_c3 + _np.array(t_geos)
    t_M = t_geos - t_injection
    
    # Insert nans for angles before the injection time
    theta_rot = _np.array(t_M * Omega)
    theta_rot = _np.where(t_M < 0.0, _np.full_like(theta_rot, fill_value=np.nan), theta_rot)

    inv_rot_matrix = utils.rotation_matrix(rot_axis, -theta_rot, use_jax=use_jax)
        
    warped_coords = _np.sum(inv_rot_matrix * coords, axis=1)
    warped_coords = _np.moveaxis(warped_coords, 0, -1)
    return warped_coords

def interpolate_coords(emission, coords):
    """
    Interpolate 3D emission field along the given coordinates
    
    Parameters
    ----------
    emission_0: np.array
        3D array with emission values
    coords: [x, y, z], 
        A list of coordinate arrays.
        
    Returns
    -------
    interpolated_data: array, 
        An array with emission values interpolated onto the coordinates.
    """
    fov = [(emission[dim].max() - emission[dim].min()).data for dim in emission.dims]
    npix = [emission[dim].size for dim in emission.dims]
    image_coords = np.moveaxis(utils.world_to_image_coords(coords, fov=fov, npix=npix), -1, 0)
    interpolated_data = scipy.ndimage.map_coordinates(emission, image_coords, order=1, cval=0.)
    return interpolated_data

def image_plane_dynamics(emission_0, geos, Omega, t_frames, t_injection, J=1.0, t_start_obs=None, slow_light=True, doppler=True, rot_axis=[0,0,1], M=consts.sgra_mass):
    """
    Compute the image-plane dynamics (movie) for a given initial emission and geodesics.
    
    Parameters
    ----------
    emission_0: np.array
        3D array with emission values
    geos: xr.Dataset
        A dataset specifying geodesics (ray trajectories) ending at the image plane.
    Omega: xr.DataArray
        A dataarray specifying the keplerian velocity field
    t_frames: array, 
        Array of time for each image frame with astropy.units
    t_injection: float, 
        Time of hotspot injection in M units.
    J: np.array(shape=(3,...)), default=None,
        Stokes polarization factors on the geodesic grid. None means no magnetic fields (non-polarized emission).
    t_start_obs: astropy.Quantity, default=None
        Start time for observations, if None t_frames[0] is assumed to be start time.
    doppler: bool, default=True
        Modeling doppler boosting.
    slow_light: bool, default=True
        Modeling the time it takes for the propogation of light.
    rot_axis: array, default=[0, 0, 1]
        Currently only equitorial plane rotation is supported
    M: astropy.Quantity, default=constants.sgra_mass,
        Mass of the black hole used to convert frame times to space-time times in units of M
        
    Returns
    -------
    images: np.array
        A movie array with image-plane frames. Polarization components are given along axis=1.
    """
    t_geos = geos.t if slow_light else 0.0
    warped_coords = velocity_warp_coords(
        coords=[geos.x, geos.y, geos.z],
        Omega=Omega, 
        t_frames=t_frames, 
        t_start_obs=np.atleast_1d(t_frames)[0] if t_start_obs is None else t_start_obs, 
        t_geos=t_geos, 
        t_injection=t_injection, 
        rot_axis=rot_axis, 
        M=M  
    )
    
    g = 1.0
    if doppler:
        umu = kgeo.azimuthal_velocity_vector(geos, Omega)
        g = kgeo.doppler_factor(geos, umu)
    
    if emission_0.ndim == 3:
        emission = interpolate_coords(emission_0, warped_coords)
    
    # If emission_0 is already a movie
    elif emission_0.ndim == 4:
        emission = []
        for t in range(emission_0.shape[0]):
            emission.append(interpolate_coords(emission_0[t], warped_coords))
        emission = np.array(emission)
        
    # Use magnetic fields for polarized synchrotron radiation
    if not np.isscalar(J):
        J = utils.expand_dims(J, emission.ndim+1, 0)
        emission = J * utils.expand_dims(emission, emission.ndim+1, 1)
        emission = np.squeeze(emission)
    images = kgeo.radiative_trasfer(emission, np.array(g), np.array(geos.dtau), np.array(geos.Sigma))
    return images

def propogate_flatspace_emission(emission_0, Omega_3D, t_frames, rot_axis=[0,0,1], M=consts.sgra_mass):
    """
    Compute the 3D movie for a given initial emission in flat-space.
    
    Parameters
    ----------
    emission_0: np.array
        3D array with emission values
    Omega_3D: xr.DataArray
        A dataarray specifying the keplerian velocity field in flat-space 3D coords (NOT GEODESICS)
    t_frames: array, 
        Array of time for each image frame with astropy.units
    rot_axis: array, default=[0, 0, 1]
        Currently only equitorial plane rotation is supported
    M: astropy.Quantity, default=constants.sgra_mass,
        Mass of the black hole used to convert frame times to space-time times in units of M
        
    Returns
    -------
    images: np.array
        A movie array with image-plane frames. Polarization components are given along axis=1.
    """
    x, y, z = np.meshgrid(emission_0.x, emission_0.y, emission_0.z, indexing='ij')
    warped_coords = velocity_warp_coords(
        coords=[x, y, z],
        Omega=Omega_3D,
        t_frames=t_frames,
        t_start_obs=np.atleast_1d(t_frames)[0], 
        t_geos=0,
        t_injection=0,
        rot_axis=rot_axis,
        M=M
    )
    emission_t = interpolate_coords(emission_0, warped_coords)
    return emission_t

def fill_unsupervised_emission(emission, coords, rmin=0, rmax=np.Inf, z_width=2.0, fill_value=0.0, use_jax=False):
    """
    Fill emission that is not within the supervision region
    
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
    z_width: float, default=2,
        Maximum width of the disk (M units) 
    fill_value: float, default=0.0
        Fill value is default to zero 
    use_jax: bool, default=False,
        Using jax enables GPU accelerated computing.
        
    Returns
    -------
    emission: np.array
        3D array with emission values filled in
    """
    _np = jnp if use_jax else np
    r_sq = _np.sum(_np.array([_np.squeeze(x)**2 for x in coords]), axis=0)
    emission = _np.where(r_sq < rmin**2, _np.full_like(emission, fill_value=fill_value), emission)
    emission = _np.where(r_sq > rmax**2, _np.full_like(emission, fill_value=fill_value), emission)
    emission = _np.where(_np.abs(coords[2]) > z_width, _np.full_like(emission, fill_value=fill_value), emission)
    return emission

def grf_to_image_plane(grf, geos, Omega, J, diameter_M, alpha=2.0, H_r=0.075):
    fov_M = float(geos.alpha[-1]-geos.alpha[0])
    gaussian = utils.gaussian_xr([grf.y.size, grf.x.size], [0,0], std=diameter_M/2.355, fov=(fov_M, 'M')).data
    movie = np.exp(alpha*grf) * gaussian
    
    # Expand the 2D grf into 3D
    emission = utils.expand_3d(movie, fov_z=fov_M, H_r=H_r)
    emission.coords.update(utils.linspace_xr(emission.shape[1:], -fov_M/2, fov_M/2))
    image_plane = image_plane_dynamics(emission, geos, Omega, 0.0, 0.0, J, slow_light=False)
    return image_plane

def normalize_stokes(movie, I_flux, P_flux, V_flux=None):
    dolp = np.sqrt(np.sum(movie[:,1:].sum(axis=(-1,-2))**2, axis=1)).mean()
    movie[:,0]  *= I_flux / movie[:,0].sum(axis=(-1,-2)).mean()
    movie[:,1:3] *= P_flux / dolp
    if V_flux is not None:
        movie[:,3]  *= V_flux / movie[:,3].sum(axis=(-1,-2)).mean()
    return movie

def rotate_evpa(stokes, angle, axis=0):
    if stokes.shape[axis] == 2:
        p = np.exp(2j*angle) * (np.take(stokes, 0, axis) + 1j * np.take(stokes, 1, axis))
        stokes_rot = np.stack([p.real, p.imag], axis=axis)
    elif stokes.shape[axis] == 3:
        p = np.exp(2j*angle) * (np.take(stokes, 1, axis) + 1j * np.take(stokes, 2, axis))
        stokes_rot = np.stack([np.take(stokes, 0, axis), p.real, p.imag], axis=axis)
    elif stokes.shape[axis] == 4:
        p = np.exp(2j*angle) * (np.take(stokes, 1, axis) + 1j * np.take(stokes, 2, axis))
        stokes_rot = np.stack([np.take(stokes, 0, axis), p.real, p.imag, np.take(stokes, 3, axis)], axis=axis)
    else:
        raise AttributeError('Shape of stokes vector along axis={} not supported'.format(axis))
    return stokes_rot