from bhnerf import utils
from bhnerf import constants as consts
import numpy as np
import xarray as xr
import jax.numpy as jnp
from astropy import units
import scipy.ndimage

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
    emission.attrs.update(rot_axis=rot_axis)
    return emission

def kerr_geodesics(spin, inclination, alpha_range, beta_range, ngeo=100, 
                   num_alpha=64, num_beta=64, distance=1000.0, E=1.0, M=1.0, verbose=False):
    """
    Compute Kerr geodesics for a given spin and inclination
    
    Parameters
    ----------
    spin: float
        normalized spin value in range [0,1]
    inclination: float, 
        inclination angle in [rad] in range [0, pi/2]
    alpha_range: tuple,
        Vertical extent (M)
    beta_range: tuple,
        Horizontal extent (M)
    ngeo: int, default=100
        Number of points along a ray
    num_alpha: int, default=64,
        Number of pixels in the vertical direction
    num_beta: int, default=64,
        Number of pixels in the horizontal direction   
    distance: float, default=1000.0
        Distance to observer
    E: float, default=1.0, 
        Photon energy at infinity 
    M: float, default=1.0, 
        Black hole mass, taken as 1 (G=c=M=1)
        
    Returns
    -------
    geos: xr.Dataset
        A dataset specifying geodesics (ray trajectories) ending at the image plane.
        
    Notes
    -----
    Overleaf notes: https://www.overleaf.com/project/60ff0ece5aa4f90d07f2a417
    units are in GM/c^2
    """
    from kgeo import raytrace_ana
    
    alpha, beta = np.meshgrid(np.linspace(*alpha_range, num_alpha), np.linspace(*beta_range, num_beta), indexing='ij')
    image_coords = [alpha.ravel(), beta.ravel()]
    
    observer_coords = [0, distance, inclination, 0]
    geos = raytrace_ana(spin, observer_coords, image_coords, ngeo, plotdata=False, verbose=verbose)
    geos = geos.get_dataset(num_alpha, num_beta, E, M)

    return geos

def wave_vector(geos): 
    """
    Compute the wave (photon momentum)  `k`
    
    Parameters
    ----------
    geos: xr.Dataset, 
        A dataset with the geodesic information and auxilary variables 
    
    Returns
    -------
    k_mu: xr.DataArray
        A dataArray with the wave 4-vector: [k_t, k_r, k_th, k_ph]
    """
    # Plus-minus sign set according to angular (theta) and radial turning points 
    pm_r = np.sign(np.gradient(geos.r, axis=-1) / np.gradient(geos.affine, axis=-1))
    pm_th = np.sign(np.gradient(geos.theta, axis=-1) / np.gradient(geos.affine, axis=-1))

    k_t  = -geos.E
    k_r  = geos.E * np.sqrt(geos.R.clip(min=0)) * pm_r / geos.Delta
    k_th = geos.E * np.sqrt(geos.Theta.clip(min=0)) * pm_th
    k_ph = geos.E * geos.lam
    k_mu = xr.concat([k_t, k_r, k_th, k_ph], dim='mu')
    return k_mu
    
def spacetime_metric(geos):
    """
    Spacetime metric g_{munu} (Boyer-Linquist coordinates)
    
    Parameters
    ----------
    geos: xr.Dataset, 
        A dataset with the geodesic information and auxilary variables 
    
    Returns
    -------
    g_munu: xr.Dataset
        A dataset with the spacetime metric (non-zero components) 
        
    Notes
    -----
    g_munu is symetric (g_munu = g_numu) 
    """
    g_munu = xr.Dataset({
        'g_tt': -(1 - 2*geos.M*geos.r / geos.Sigma), 
        'g_rr': geos.Sigma / geos.Delta,
        'g_thth': geos.Sigma,
        'g_phph': geos.Xi*np.sin(geos.theta)**2 / geos.Sigma, 
        'g_tph': -2*geos.M*geos.spin*geos.r*np.sin(geos.theta)**2 / geos.Sigma
    })
    return g_munu

def spacetime_inv_metric(geos):
    """
    Inverse spacetime metric g^{munu} (Boyer-Linquist coordinates)
    
    Parameters
    ----------
    geos: xr.Dataset, 
        A dataset with the geodesic information and auxilary variables 
    
    Returns
    -------
    gmunu: xr.Dataset
        A dataset with the *inverse* spacetime metric (non-zero components) 
        
    Notes
    -----
    gmunu is symetric (gmunu = gnumu) 
    """
    gmunu = xr.Dataset({
        'gtt': -geos.Xi / (geos.Delta * geos.Sigma),
        'grr': geos.Delta / geos.Sigma,
        'gthth': 1 / geos.Sigma,
        'gphph': (geos.Delta - geos.spin**2 * np.sin(geos.theta)**2) / 
                 (geos.Delta * geos.Sigma * np.sin(geos.theta)**2), 
        'gtph': -2*geos.M*geos.spin*geos.r / (geos.Delta * geos.Sigma)
    })
    return gmunu

def raise_or_lower_indices(g, u):
    """
    Change contravarient to covarient vectors and vice-versa
    Lower indices: u_mu = g_munu * u^nu
    Raise indices: u^mu = g^munu * u_mu
    
    Parameters
    ----------
    g: xr.Dataset, 
        A dataset with non-zero spacetime metric components
    u: xr.DataArray, 
        A 4-vector dataarray with mu coordinates.
    """
    if 'g_tt' in g:
        g_tt = g.g_tt
    elif 'gtt' in g:
        g_tt = g.gtt 
    else:
        raise AttributeError('spacetime metric has no `tt` component')

    if 'g_rr' in g:
        g_rr = g.g_rr
    elif 'grr' in g:
        g_rr = g.grr 
    else:
        raise AttributeError('spacetime metric has no `rr` component')

    if 'g_thth' in g:
        g_thth = g.g_thth
    elif 'gthth' in g:
        g_thth = g.gthth 
    else:
        raise AttributeError('spacetime metric has no `thth` component')

    if 'g_phph' in g:
        g_phph = g.g_phph
    elif 'gphph' in g:
        g_phph = g.gphph 
    else:
        raise AttributeError('spacetime metric has no `phph` component')
    
    if 'g_tph' in g:
        g_tph = g.g_tph
    elif 'gtph' in g:
        g_tph = g.gtph 
    else:
        raise AttributeError('spacetime metric has no `tph` component')
        
    u_prime = xr.concat([
        g_tt * u.sel(mu=0) + g_tph * u.sel(mu=3),
        g_rr * u.sel(mu=1),
        g_thth * u.sel(mu=2),
        g_phph * u.sel(mu=3) + g_tph * u.sel(mu=0)
    ], dim='mu')
    return u_prime

def azimuthal_velocity_vector(geos, Omega):
    """
    Compute azimuthal velocity umu 4-vector on points sampled along geodesics
    
    Parameters
    ----------
    geos: xr.Dataset
        Dataset with Kerr geodesics (see: `kerr_geodesics` for more details)
    Omega: array, 
        An array with angular velocity specified along the geodesics coordinates
    
    Returns
    -------
    umu: xr.DataArray
        array with contravarient velocity 4 vector (index up) 
    """
    g_munu = spacetime_metric(geos)
    
    # 4-velocity vector
    ut = 1 / np.sqrt(-(g_munu.g_tt + 2*Omega*g_munu.g_tph + g_munu.g_phph*Omega**2))
    ur = xr.DataArray(0)
    uth = xr.DataArray(0)
    uph = ut * Omega
    umu = xr.concat([ut, ur, uth, uph], dim='mu', coords='minimal')
    return umu

def doppler_factor(geos, umu, fillna=0.0):
    """
    Compute Doppler factor as dot product of wave 4-vectors with the velocity 4-vecotr
    
    Parameters
    ----------
    geos: xr.Dataset
        Dataset with Kerr geodesics (see: `kerr_geodesics` for more details)
    umu: xr.DataArray
        array with contravarient velocity 4 vector (index up)
    fillna: float or False or None, 
        If float fill nans with float else if False leave nans
        
    Returns
    -------
    g: xr.DataArray,
        Doppler boosting factor sampled along the geodesics
    """
    g_munu = spacetime_metric(geos)
    k_mu = wave_vector(geos)
    g = geos.E / -(k_mu * umu).sum('mu', skipna=False)
    if not ((isinstance(fillna, bool) and fillna == False) or fillna is None):
        g = g.fillna(fillna)
    return g

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

def radiative_trasfer(emission, g, dtau, Sigma, use_jax=False):
    """
    Integrate emission over rays to get sensor pixel values.

    Parameters
    ----------
    emission: array,
        An array with emission values.
    g: array, 
        doppler boosting factor, 
    dtau: array, 
        mino time differential
    Sigma: array, 
    use_jax: bool, default=False,
        Using jax enables GPU accelerated computing.
        
    Returns
    -------
    radiance: array, 
        An array with pixel flux values.
    """
    g = utils.expand_dims(g, emission.ndim, use_jax=use_jax)
    dtau = utils.expand_dims(dtau, emission.ndim, use_jax=use_jax)
    Sigma = utils.expand_dims(Sigma, emission.ndim, use_jax=use_jax)
    fluxes = (g**2 * emission * dtau * Sigma).sum(axis=-1)
    return fluxes

def image_plane_dynamics(emission_0, geos, Omega, t_frames, t_injection, t_start_obs=None, rot_axis=[0,0,1], M=consts.sgra_mass):
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
    t_start_obs: astropy.Quantity, default=None
        Start time for observations, if None t_frames[0] is assumed to be start time.
    rot_axis: array, default=[0, 0, 1]
        Currently only equitorial plane rotation is supported
    M: astropy.Quantity, default=constants.sgra_mass,
        Mass of the black hole used to convert frame times to space-time times in units of M
        
    Returns
    -------
    movie: np.array
        A movie array with image-plane frames
    """
    warped_coords = velocity_warp_coords(
        coords=[geos.x, geos.y, geos.z],
        Omega=Omega, 
        t_frames=t_frames, 
        t_start_obs=t_frames[0] if t_start_obs is None else t_start_obs, 
        t_geos=geos.t, 
        t_injection=t_injection, 
        rot_axis=rot_axis, 
        M=M  
    )
    emission = interpolate_coords(emission_0, warped_coords)
    umu = azimuthal_velocity_vector(geos, Omega)
    g = doppler_factor(geos, umu)
    movie = radiative_trasfer(emission, np.array(g), np.array(geos.dtau), np.array(geos.Sigma))
    return movie

def fill_unsupervised_emission(emission, coords, rmin=0, rmax=np.Inf, fill_value=0.0, use_jax=False):
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
    return emission