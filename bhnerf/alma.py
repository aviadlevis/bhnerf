import bhnerf
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib import tzip
from astropy import units
import os
import pandas as pd

def preprocess_data(data_path, window_size, I_hs_mean, P_sha, chi_sha, de_rot_angle, t_start=9.33, t_end=11.05):
    # Load data and temporally average 
    # "loopy priod" between t0=9:20 UTC -- t7=11:03 [Weilgus et al. 2022]
    alma_lc = pd.read_csv(data_path, index_col=0)
    alma_lc_loops = alma_lc.loc[np.bitwise_and(alma_lc['time']>=t_start, alma_lc['time']<=t_end)]
    alma_lc_means = alma_lc_loops.rolling(window_size).mean().loc[::window_size].dropna()

    # drop points averaged between scans
    alma_lc_means = alma_lc_means.where(alma_lc_means['time'].diff().fillna(0.0) < 160/3600).dropna()
    t_frames = alma_lc_means['time'].values * units.hr

    # Remove a constant Q,U shadow (accretion disk polarization) and de-rotate Faraday rotation
    # Set a prior on the hotspot intensity as a Gaussian
    qu_sha = P_sha * np.array([np.cos(2*np.deg2rad(chi_sha)), np.sin(2*np.deg2rad(chi_sha))])
    target = bhnerf.emission.rotate_evpa(np.array(alma_lc_means[['Q','U']]) - qu_sha, np.deg2rad(de_rot_angle), axis=1)
    target = np.pad(target, ([0,0], [1,0]), constant_values=I_hs_mean)
    return target, t_frames
    
def image_plane_model(inc, spin, params, rot_angle=0.0, randomize_subpixel_rays=False):
    rot_sign={'cw': -1, 'ccw': 1}
    
    # Local variable assignment 
    num_alpha, num_beta, fov_M, z_width = params['num_alpha'], params['num_beta'], params['fov_M'], params['z_width']
    Q_frac, b_consts, Omega_dir = params['Q_frac'], params['b_consts'], params['Omega_dir']
    rmin = float(bhnerf.constants.isco_pro(spin)) if params['rmin'] == 'ISCO' else params['rmin']
    rmax = fov_M / 2
    
    # Compute geodesics paths
    geos = bhnerf.kgeo.image_plane_geos(
        spin, inc, 
        num_alpha=num_alpha, 
        num_beta=num_beta, 
        alpha_range=[-fov_M/2, fov_M/2],
        beta_range=[-fov_M/2, fov_M/2],
        randomize_subpixel_rays=randomize_subpixel_rays
    )
    geos = geos.fillna(0.0)
    
     # Keplerian velocity and Doppler boosting
    Omega = rot_sign[Omega_dir] * np.sqrt(geos.M) / (geos.r**(3/2) + geos.spin * np.sqrt(geos.M))
    umu = bhnerf.kgeo.azimuthal_velocity_vector(geos, Omega)
    g = bhnerf.kgeo.doppler_factor(geos, umu)
    
    # Magnitude normalized magnetic field in fluid-frame
    b = bhnerf.kgeo.magnetic_field_fluid_frame(geos, umu, **b_consts)
    domain = np.bitwise_and(np.bitwise_and(np.abs(geos.z) < z_width, geos.r > rmin), geos.r < rmax)
    b_mean = np.sqrt(np.sum(b[domain]**2, axis=-1)).mean()
    b /= b_mean
    
    # Polarized emission factors (including parallel transport)
    J = np.nan_to_num(bhnerf.kgeo.parallel_transport(geos, umu, g, b, Q_frac=Q_frac, V_frac=0), 0.0)
    J = bhnerf.emission.rotate_evpa(J, rot_angle)

    return geos, Omega, J

def get_raytracing_args(inc, spin, params, stokes=['I','Q','U'], rot_angle=0.0, num_subpixel_rays=1):
    raytracing_args = []
    J_inds = [['I', 'Q', 'U'].index(s) for s in stokes]
    
    if num_subpixel_rays == 1:
        randomize_rays = False
        ray_iteration = range(num_subpixel_rays)
    else:
        randomize_rays = True
        ray_iteration = tqdm(range(num_subpixel_rays), leave=False, desc='subrays')
        
    for i in ray_iteration:
        geos, Omega, J = image_plane_model(inc, spin, params, rot_angle, randomize_rays)
        t_injection = -float(geos.r_o + params['fov_M']/4)
        args = bhnerf.network.raytracing_args(geos, Omega, t_injection, params['t_start_obs']*units.hr, J[J_inds])
        raytracing_args.append(args)
    return raytracing_args

def chi2_lightcurves(raytracing_args, checkpoint_dir, t, data, sigma=1.0, rmin=0.0, rmax=np.inf, batchsize=20):
    image_plane = bhnerf.network.image_plane_checkpoint(raytracing_args, checkpoint_dir, t, rmin, rmax, batchsize)
    chi2 = np.sum(((image_plane.sum(axis=(-1,-2)) - data) / sigma)**2) / len(t)
    return chi2

def chi2_df(inclinations, spins, seeds, params, checkpoint_fmt, t, data, stokes=['I','Q','U'], sigma=1.0, rot_angle=0.0, num_subpixel_rays=1):

    import pandas as pd
    
    inclinations, spins = np.atleast_1d(inclinations), np.atleast_1d(spins)
    if len(inclinations)==1 and len(spins)>1:
        indices = spins
        index_name = 'spin'
        inclinations = np.full_like(spins, inclinations)
    elif len(inclinations)>1 and len(spins)==1:
        indices = inclinations
        index_name = 'inc'
        spins = np.full_like(inclinations, spins)
    elif len(inclinations)>1 and len(spins)>1:
        raise AttributeError('not implemented')
    
    inc_prev = spin_pref = np.nan
    data_fit = np.full((len(indices), len(seeds)), fill_value=np.nan)
    for i, (inc, spin) in enumerate(tzip(inclinations, spins, desc=index_name)):
        for j, seed in enumerate(tqdm(seeds, desc='seed', leave=False)):
            checkpoint_dir = checkpoint_fmt.format(indices[i], seed)
            if os.path.exists(os.path.join(checkpoint_dir, 'checkpoint_50000')):
                if (inc_prev != inc) or (spin_prev != spin):
                    raytracing_args = bhnerf.alma.get_raytracing_args(np.deg2rad(inc), spin, params, stokes, rot_angle, num_subpixel_rays)
                    inc_prev, spin_prev = inc, spin
                data_fit[i,j] = bhnerf.alma.chi2_lightcurves(raytracing_args, checkpoint_dir, t, data, sigma)

    data_fit_df = pd.DataFrame(data_fit, index=indices, columns=['seed 0', 'seed 1', 'seed 2', 'seed 3'])
    data_fit_df.index.name = index_name
    return data_fit_df

