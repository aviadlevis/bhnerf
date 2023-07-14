import bhnerf
import numpy as np

def image_plane_model(inc, spin, params, rot_angle=0.0, rot_sign={'cw': -1, 'ccw': 1}, randomize_subpixel_rays=False):
    
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
    J_rot = bhnerf.emission.rotate_evpa(J, rot_angle)

    return geos, Omega, J_rot