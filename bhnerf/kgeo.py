from kgeo import *
import xarray as xr
import numpy as np
from bhnerf import utils

def image_plane_geos(spin, inclination, alpha_range, beta_range, ngeo=100, 
                     num_alpha=64, num_beta=64, distance=1000.0, E=1.0, M=1.0, verbose=False):
    """
    Compute Kerr geodesics for the entire image plane 
    
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
    
    alpha, beta = np.meshgrid(np.linspace(*alpha_range, num_alpha), np.linspace(*beta_range, num_beta), indexing='ij')
    image_coords = [alpha.ravel(), beta.ravel()]
    
    observer_coords = [0, distance, inclination, 0]
    geos = raytrace_ana(spin, observer_coords, image_coords, ngeo, plotdata=False, verbose=verbose)
    geos = geos.get_dataset(num_alpha, num_beta, E, M)
    return geos

def transform_coordinates(v, tetrad, contraction):
    """
    Tranform coordinate frame of v using tetrad.
    
    Parameters
    ----------
    v: xr.DataArray or np.array, 
        A 4-vector along last dimension (shape=(...,4))
    tetrad: array(shape=(..., 4, 4)), default=None,
        Optional coordinate transformation matrix. If None, the global frame coordinates are used.
    contraction: str='lower', 'upper'
        The index upon which to contract the matrix multiplication.
        
    Returns
    -------
    v_prime: np.array(shape=(...,4))
        Transformed vector coordinates.
    """
    v = np.array(v)
    if contraction == 'upper':
        tetrad = np.moveaxis(tetrad, [-2,-1], [-1, -2])
    elif contraction != 'lower':
        raise AttributeError('contraction can be either "upper" or "lower"')
    v_prime = np.squeeze(np.matmul(tetrad, v[...,None]), axis=-1)
    return v_prime
    
def wave_vector(geos): 
    """
    Compute the wave (photon momentum)  `k`
    
    Parameters
    ----------
    geos: xr.Dataset, 
        A dataset with the geodesic information and auxilary variables 

    Returns
    -------
    k_mu: np.array(shape=(...,4))
         First dimensions match geos dimensions.
         Last dimension is the spherical coordinate 4-vector: [k_t, k_r, k_th, k_ph]
    """
    # Plus-minus sign set according to angular (theta) and radial turning points 
    pm_r = np.sign(np.gradient(geos.r, axis=-1) / np.gradient(geos.affine, axis=-1))
    pm_th = np.sign(np.gradient(geos.theta, axis=-1) / np.gradient(geos.affine, axis=-1))

    # Global frame wave vector
    k_t  = -geos.E
    k_r  = geos.E * np.sqrt(geos.R.clip(min=0)) * pm_r / geos.Delta
    k_th = geos.E * np.sqrt(geos.Theta.clip(min=0)) * pm_th
    k_ph = geos.E * geos.lam
    k_mu = xr.concat([k_t, k_r, k_th, k_ph], dim='mu').transpose(...,'mu')
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
        'tt': -(1 - 2*geos.M*geos.r / geos.Sigma), 
        'rr': geos.Sigma / geos.Delta,
        'thth': geos.Sigma,
        'phph': geos.Xi*np.sin(geos.theta)**2 / geos.Sigma, 
        'tph': -2*geos.M*geos.spin*geos.r*np.sin(geos.theta)**2 / geos.Sigma
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
        'tt': -geos.Xi / (geos.Delta * geos.Sigma),
        'rr': geos.Delta / geos.Sigma,
        'thth': 1 / geos.Sigma,
        'phph': (geos.Delta - geos.spin**2 * np.sin(geos.theta)**2) / 
                 (geos.Delta * geos.Sigma * np.sin(geos.theta)**2), 
        'tph': -2*geos.M*geos.spin*geos.r / (geos.Delta * geos.Sigma)
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
        
    Returns
    -------
    u_prime: xr.DataArray, 
        Raised or lowered indices depending on the metric g.
    """
    u_prime = xr.concat([
        g.tt * u.sel(mu=0) + g.tph * u.sel(mu=3),
        g.rr * u.sel(mu=1),
        g.thth * u.sel(mu=2),
        g.phph * u.sel(mu=3) + g.tph * u.sel(mu=0)
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
    ut = 1 / np.sqrt(-(g_munu.tt + 2*Omega*g_munu.tph + g_munu.phph*Omega**2))
    ur = xr.DataArray(0)
    uth = xr.DataArray(0)
    uph = ut * Omega
    umu = xr.concat([ut, ur, uth, uph], dim='mu', coords='minimal')
    return umu

def doppler_factor(geos, umu, fillna=0.0):
    """
    Compute Doppler factor as dot product of wave 4-vectors with the velocity 4-vector
    
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

def magnetic_field(geos, b_r, b_th, b_ph):
    """
    A spherical-coordinate magnetic field sampled on the geodesic coordinates 
    
    Parameters
    ----------
    geos: xr.Dataset
        Dataset with Kerr geodesics (see: `kerr_geodesics` for more details)
    b_r, b_th, b_ph: floats or arrays,
        magnetic field components in (r, theta, phi) 
        float defines a constant magnetic field component along the geodesics. 
        An array should have the same dimensions as geos.
        
    Returns
    -------
    b: xr.DataArray,
        Spherical-coordinate magnetic field sampled along the geodesics. component dim='mu'.
    """
    if np.isscalar(b_r): b_r = xr.full_like(geos.mino, fill_value=b_r)
    if np.isscalar(b_th): b_th = xr.full_like(geos.mino, fill_value=b_th)
    if np.isscalar(b_ph): b_ph = xr.full_like(geos.mino, fill_value=b_ph)
    b = xr.concat([b_r, b_th, b_ph], dim='mu').transpose(...,'mu')
    return b

def fluid_frame_tetrad(geos, umu):
    """
    Transformation tetrad: transforms vectors (e.g. wave-vector/magnetic field) to a (local co-moving) fluid reference frame.
    
    Parameters
    ----------
    geos: xr.Dataset
        Dataset with Kerr geodesics (see: `kerr_geodesics` for more details)
    umu: xr.DataArray
        array with contravarient velocity 4 vector (index up)
    
    Returns 
    -------
    e_mu: array(shape=(...,4,4))
        Tetrad matrix for transforming 4-vactors onto the moving reference frame (umu)
        
    Notes
    -----
    Eqs (62)-(68) in the notes: https://www.overleaf.com/project/60ff0ece5aa4f90d07f2a417 
    """
    # lower indices of velocity vector 
    g_munu = spacetime_metric(geos)
    u_mu = raise_or_lower_indices(g_munu, umu)
    u_mumu = u_mu * umu
    
    # Normalization factors
    N_r = np.sqrt(-g_munu.rr * (u_mumu.sel(mu=0) + u_mumu.sel(mu=3)) * (1 + u_mumu.sel(mu=2)))
    N_th = np.sqrt(g_munu.thth * (1 + u_mumu.sel(mu=2)))
    N_ph = np.sqrt(-(u_mumu.sel(mu=0) + u_mumu.sel(mu=3)) * geos.Delta * np.sin(geos.theta)**2 )
    
    # Compute tetrad
    e_t = -umu
    e_r = xr.concat([u_mu.sel(mu=1) * umu.sel(mu=0), -(u_mumu.sel(mu=0) + u_mumu.sel(mu=3)), xr.DataArray(0), u_mu.sel(mu=1) * umu.sel(mu=3)], dim='mu', coords='minimal') / N_r
    e_th = xr.concat([u_mu.sel(mu=2) * umu.sel(mu=0), u_mu.sel(mu=2) * umu.sel(mu=1), 1 + u_mumu.sel(mu=2), u_mu.sel(mu=2) * umu.sel(mu=3)], dim='mu', coords='minimal') / N_th
    e_ph = xr.concat([u_mu.sel(mu=3), xr.DataArray(0), xr.DataArray(0), -u_mu.sel(mu=0)], dim='mu', coords='minimal') / N_ph
    e_mu = np.moveaxis(np.stack([e_t, e_r, e_th, e_ph], axis=1), [0, 1], [-2, -1])
    return e_mu

def zamo_frame_tetrad(geos, beta, chi):
    """
    Transformation tetrad: transforms vectors to the ZAMO frame
    
    Parameters
    ----------
    geos: xr.Dataset
        Dataset with Kerr geodesics (see: `kerr_geodesics` for more details)
    beta: float, 
        Boosting speed in [0, 1). beta=0 corresponds to the ZAMO frame. 
    chi: float, 
        Boosting angle [rad] units.
    
    Returns 
    -------
    e_mu: array(shape=(...,4,4))
        Tetrad matrix for transforming 4-vactors onto the moving reference frame (umu)
        
    Notes
    -----
    Eq (A4) in Gelles2021: https://arxiv.org/pdf/2105.09440.pdf
    Taking +1/geos.r in e_th (instead of minus) to have the RH coordinate system work with cross-product. 
    Our coordinate system's theta points down (r, theta, phi). Gelles' points up (r, phi, -theta).
    """
    gamma = 1 / np.sqrt(1-beta**2)
    e_t = xr.concat([
        (gamma / geos.r) * np.sqrt(geos.Xi / geos.Delta), 
        (beta * gamma * np.cos(chi) / geos.r) * np.sqrt(geos.Delta),
        xr.DataArray(0),
        (gamma  * geos.omega / geos.r) * np.sqrt(geos.Xi / geos.Delta) + geos.r * beta * gamma * np.sin(chi) / np.sqrt(geos.Xi)
    ], dim='mu', coords='minimal')
    
    e_r = xr.concat([
        (beta * gamma * np.cos(chi) / geos.r) * np.sqrt(geos.Xi / geos.Delta), 
        ((1 + (gamma-1)*np.cos(chi)**2) / geos.r) * np.sqrt(geos.Delta),
        xr.DataArray(0),
        beta*gamma*geos.omega*np.cos(chi)/ geos.r * np.sqrt(geos.Xi / geos.Delta) + \
        geos.r*(gamma-1)*np.cos(chi)*np.sin(chi) / np.sqrt(geos.Xi)
    ], dim='mu', coords='minimal')
    e_th = xr.concat([xr.DataArray(0), xr.DataArray(0), 1/geos.r, xr.DataArray(0)], dim='mu', coords='minimal')
    e_ph = xr.concat([
        (beta * gamma * np.sin(chi) / geos.r) * np.sqrt(geos.Xi / geos.Delta), 
        ((gamma-1)*np.cos(chi)*np.sin(chi) / geos.r) * np.sqrt(geos.Delta),
        xr.DataArray(0),
        (beta*geos.omega*np.sin(chi)*(gamma / geos.r) * np.sqrt(geos.Xi / geos.Delta) + \
         geos.r*((gamma-1)*np.sin(chi)**2 + 1) / np.sqrt(geos.Xi))
    ], dim='mu', coords='minimal')
    e_mu = np.moveaxis(np.stack([e_t, e_r, e_th, e_ph], axis=1), [0, 1], [-2, -1])
    return e_mu

def zamo_frame_velocity(geos, beta, chi):
    """
    ZAMO frame parameterization for the velocity
    
    Parameters
    ----------
    geos: xr.Dataset
        Dataset with Kerr geodesics (see: `kerr_geodesics` for more details)
    beta: float, 
        Boosting speed in [0, 1). beta=0 corresponds to the ZAMO frame. 
    chi: float, 
        Boosting angle [rad] units.
    
    Returns
    -------
    umu: xr.DataArray,
        Velocity field (covarient/index down) sampled along the geodesics. component dim='mu'.
        
    Notes
    -----
    Gelles2021 et al: https://arxiv.org/pdf/2105.09440.pdf 
    """
    gamma = 1 / np.sqrt(1-beta**2)
    ut = (gamma / geos.r) * np.sqrt(geos.Xi / geos.Delta)
    ur = (beta * gamma * np.cos(chi) / geos.r) * np.sqrt(geos.Delta)
    uth = xr.DataArray(0)
    uph = ut * geos.omega + geos.r * beta * gamma * np.sin(chi) / np.sqrt(geos.Xi)
    umu = xr.concat([ut, ur, uth, uph], dim='mu', coords='minimal')
    return umu

def parallel_transport(geos, umu, g, b, Q_factor=0.2, spectral_index=1):
    """
    Parallel transport stokes vector J = [I, Q, U] to the observer screen. 
    Locally emitted polarization has U=0 (before paralell transport).
    
    Parameters
    ----------
    geos: xr.Dataset
        A dataset specifying geodesics (ray trajectories).
    umu: xr.DataArray
        array with contravarient velocity 4 vector (index up)
    g: array, 
        doppler boosting factor.
    b: xr.DataArray,
        Spherical-coordinate magnetic field sampled along the geodesics. component dim='mu'.
    Q_factor: float, default=0.2, 
        Scaling of Q with respect to I. Q_factor < 1.0
    spectral_index: int, default=1,
        Spectral index used for EHT frequencies.
        
    Returns
    -------
    J: np.array(shape=(3,...)),
        Stokes vector scaling factors including parallel transport (I, Q, U)
    
    Notes
    -----
    Currently doesnt support V component
    """
    if Q_factor > 1.0 or Q_factor < 0.0: raise AttributeError('Q_factor should be in [0,1]')
        
    # Compute f (EVPA) in local fluid frame
    # Compute local cross product of wave vector with the magnetic field
    e_mu = fluid_frame_tetrad(geos, umu)
    k_mu = wave_vector(geos)
    k_mu_prime =  transform_coordinates(k_mu, e_mu, 'upper')[...,1:] # Remove time component
    k_mag = np.sqrt(np.sum(k_mu_prime**2, axis=-1))
    f_local = np.cross(k_mu_prime, b, axis=-1) / k_mag[...,None]
    
    # Transform local evpa vector f to global coordinates. 
    # Pad time component with zeros (last axis) and right multiply by the tetrad.
    pad_width = [(0,0)]*(f_local.ndim-1) + [(1,0)]
    f_global = np.pad(f_local, pad_width)
    f_global = transform_coordinates(f_global, e_mu, 'lower')
    ft, fr, fth, fph  = f_global[...,0], f_global[...,1], f_global[...,2], f_global[...,3]

    # Compute emissivity scalings which depend on:
    #    - the magnetic field magnitude and pitch angle: b_mag, sin(theta_b)
    #    - doppler factor: g. 
    #    - spectral index
    b_mag = np.array(np.sqrt((b**2).sum('mu')))
    sin_th_b = np.sqrt((f_local**2).sum(axis=-1)) / np.sqrt((k_mu_prime**2).sum(axis=-1))
    I = g**spectral_index * b_mag**(spectral_index+1) * sin_th_b**(spectral_index+1)
    Q = Q_factor * I
    U = np.zeros_like(Q)
    emitted_qu = np.stack([Q, U], axis=-1)[...,None]
    
    # Compute Penrose-Walker complex constant kappa and extract the rotation angle chi2 for parallel transport to observer screen
    # Expressions from Himwich2020: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.084020
    gmunu = spacetime_inv_metric(geos)
    kmu = raise_or_lower_indices(gmunu, k_mu)
    A = (kmu.sel(mu=0)*fr - kmu.sel(mu=1)*ft) + geos.spin*np.sin(geos.theta)**2 * (kmu.sel(mu=1)*fph - kmu.sel(mu=3)*fr)
    B = ((geos.r**2 + geos.spin**2) * (kmu.sel(mu=3)*fth - kmu.sel(mu=2)*fph) - geos.spin*(kmu.sel(mu=0)*fth - kmu.sel(mu=2)*ft)) * np.sin(geos.theta)
    kappa = (geos.r - 1j*geos.spin * np.cos(geos.theta)) * (A - 1j*B)
    mu = -(geos.alpha + geos.spin*np.sin(geos.inc))
    chi2 = np.angle( ((geos.beta + 1j*mu)*kappa.conj()) / ((geos.beta - 1j*mu)*kappa) )
    rot_matrix = np.moveaxis(np.array([
        [np.cos(chi2), -np.sin(chi2)],
        [np.sin(chi2), np.cos(chi2)]
    ]), [0, 1], [-2, -1])
    J_i = np.array(I)[None,...]
    J_qu = np.moveaxis(np.squeeze(np.matmul(rot_matrix, emitted_qu), axis=-1), -1, 0)
    J = np.concatenate([J_i, J_qu])
    return J

def parallel_transport_zamo(geos, beta_v, chi, g, b, Q_factor=0.2, spectral_index=1):
    """
    Parallel transport stokes vector J = [I, Q, U] to the observer screen. 
    Locally emitted polarization has U=0 (before paralell transport).
    
    Parameters
    ----------
    geos: xr.Dataset
        A dataset specifying geodesics (ray trajectories).
    umu: xr.DataArray
        array with contravarient velocity 4 vector (index up)
    g: array, 
        doppler boosting factor.
    b: xr.DataArray,
        Spherical-coordinate magnetic field sampled along the geodesics. component dim='mu'.
    Q_factor: float, default=0.2, 
        Scaling of Q with respect to I. Q_factor < 1.0
    spectral_index: int, default=1,
        Spectral index used for EHT frequencies.
        
    Returns
    -------
    J: np.array(shape=(3,...)),
        Stokes vector scaling factors including parallel transport (I, Q, U)
    
    Notes
    -----
    Currently doesnt support V component
    """
    if Q_factor > 1.0 or Q_factor < 0.0: raise AttributeError('Q_factor should be in [0,1]')
        
    # Compute f (EVPA) in local fluid frame
    # Compute local cross product of wave vector with the magnetic field
    e_mu = zamo_frame_tetrad(geos, beta_v, chi)
    k_mu = wave_vector(geos)
    k_mu_prime =  transform_coordinates(k_mu, e_mu, 'upper')[...,1:] # Remove time component
    k_mag = np.sqrt(np.sum(k_mu_prime**2, axis=-1))
    f_local = np.cross(k_mu_prime, b, axis=-1) / k_mag[...,None]
    
    # Transform local evpa vector f to global coordinates. 
    # Pad time component with zeros and right multiply by the tetrad.
    f_global = np.pad(f_local, pad_width=((0,0), (0,0), (1,0)))
    f_global = transform_coordinates(f_global, e_mu, 'lower')
    ft, fr, fth, fph  = f_global[...,0], f_global[...,1], f_global[...,2], f_global[...,3]

    # Compute emissivity scalings which depend on:
    #    - the magnetic field magnitude and pitch angle: b_mag, sin(theta_b)
    #    - doppler factor: g. 
    #    - spectral index
    b_mag = np.array(np.sqrt((b**2).sum('mu')))
    sin_th_b = np.sqrt((f_local**2).sum(axis=-1)) / np.sqrt((k_mu_prime**2).sum(axis=-1))
    I = g**spectral_index * b_mag**(spectral_index+1) * sin_th_b**(spectral_index+1)
    Q = Q_factor * I
    U = np.zeros_like(Q)
    emitted_qu = np.stack([Q, U], axis=-1)[...,None]
    
    # Compute Penrose-Walker complex constant kappa and extract the rotation angle chi2 for parallel transport to observer screen
    # Expressions from Himwich2020: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.084020
    gmunu = spacetime_inv_metric(geos)
    kmu = raise_or_lower_indices(gmunu, k_mu)
    A = (kmu.sel(mu=0)*fr - kmu.sel(mu=1)*ft) + geos.spin*np.sin(geos.theta)**2 * (kmu.sel(mu=1)*fph - kmu.sel(mu=3)*fr)
    B = ((geos.r**2 + geos.spin**2) * (kmu.sel(mu=3)*fth - kmu.sel(mu=2)*fph) - geos.spin*(kmu.sel(mu=0)*fth - kmu.sel(mu=2)*ft)) * np.sin(geos.theta)
    kappa = (geos.r - 1j*geos.spin * np.cos(geos.theta)) * (A - 1j*B)
    mu = -(geos.alpha + geos.spin*np.sin(geos.inc))
    chi2 = np.angle( ((geos.beta + 1j*mu)*kappa.conj()) / ((geos.beta - 1j*mu)*kappa) )
    rot_matrix = np.moveaxis(np.array([
        [np.cos(chi2), -np.sin(chi2)],
        [np.sin(chi2), np.cos(chi2)]
    ]), [0, 1], [-2, -1])
    J_i = np.array(I)[None,...]
    J_qu = np.moveaxis(np.squeeze(np.matmul(rot_matrix, emitted_qu), axis=-1), -1, 0)
    J = np.concatenate([J_i, J_qu])
    return J

def radiative_trasfer(emission, g, dtau, Sigma, use_jax=False):
    """
    Integrate emission over rays to get sensor pixel values.

    Parameters
    ----------
    emission: array,
        An array with emission values.
    J: np.array(shape=(3,...)),
        Stokes vector scaling factors including parallel transport (I, Q, U)
    g: array, 
        doppler boosting factor, 
    dtau: array, 
        mino time differential
    Sigma: array, 
    use_jax: bool, default=False,
        Using jax enables GPU accelerated computing.
        
    Returns
    -------
    stokes: array, 
        Image plane array with stokes vector values.
    """
    g = utils.expand_dims(g, emission.ndim, use_jax=use_jax)
    dtau = utils.expand_dims(dtau, emission.ndim, use_jax=use_jax)
    Sigma = utils.expand_dims(Sigma, emission.ndim, use_jax=use_jax)
    stokes = (g**2 * emission * dtau * Sigma).sum(axis=-1)
    return stokes

