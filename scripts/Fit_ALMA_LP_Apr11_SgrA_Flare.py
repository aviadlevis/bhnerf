"""
This script fits the April 11 ALMA linear polarization of Sagittarius A*. 
The fit is of period of time where Q-U loops are appearent in the data directly after an X-ray flare [Wielgus et al. 2022].
"""
import bhnerf
from bhnerf.optimization import LogFn
from astropy import units
import jax
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
import argparse
import ruamel.yaml as yaml
import warnings
warnings.simplefilter("ignore")

def preprocess_data(data_path, window_size, I_hs_mean, P_sha, chi_sha, de_rot_angle):
    
    
    # Load data and temporally average 
    # "loopy priod" between t0 -- t7 [Weilgus et al. 2022]
    t0 = 9. + 20./60.                   # start time [UTC]
    t7 = t0 + 68./60. + 35./60.         # end time [UTC]
    alma_lc = pd.read_csv(data_path, index_col=0)
    alma_lc_loops = alma_lc.loc[np.bitwise_and(alma_lc['time']>=t0, alma_lc['time']<=t7)]
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inc', type=int, nargs='+',
                    help='Inclination angle (if one argument is passed). \n\
                          If two arguments are passed then the 39 points in the angle range of [4 80] is split: \n\
                             - first integer is num_blocks \n\
                             - second integer is index (starts at 0) \n\
                          example: \n\
                          `3 1` will split the 39 angles to 3 blocks of 13 angles and run the second block: 30-54 [deg].')
    parser.add_argument('--start_inc', type=float, 
                        help='Start after this angle.')
    parser.add_argument('--seeds', type=int, nargs='+',
                        help='Seeds for initializing networks weights.')
    parser.add_argument('--gpu', type=int, nargs='+',
                        help='GPU Device index (e.g. 1 2 3 for 3 GPUs)')
    parser.add_argument('--data_path', type=str, default='../data/Apr11_HI.dat',
                        help='Path to ALMA April 11, 2017 Data (HI band)')
    parser.add_argument('--config_path', type=str, default='Fit_ALMA_LP_Apr11_SgrA_Flare.yaml', 
                        help='Path to configuration YAML file')
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    config['cmdline'] = vars(args)
        
    # Save configuration files
    directory = Path(config['optimization']['checkpoint_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory.joinpath('config.yml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
        
    # Update number of GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args.gpu]) if args.gpu else ''
    print('running on devices: {}'.format(jax.devices()))
    return args, config

if __name__ == "__main__":
    
    basename = 'inc_{:.1f}.seed_{}_test123'
    
    args, config = parse_args()
    locals().update(config['preprocess'])
    locals().update(config['model'])
    locals().update(config['optimization'])
    
    # Preprocess the data
    target, t_frames = preprocess_data(**config['preprocess'])
    
    # Setup model for recovery
    rmax = fov_M / 2
    if rmin == 'ISCO': rmin = float(bhnerf.constants.isco_pro(spin))
    train_step = bhnerf.optimization.TrainStep.image(t_frames, target, sigma, dtype='lc')
    predictor = bhnerf.network.NeRF_Predictor(rmax, rmin, rmax, z_width, posenc_var=recovery_scale/fov_M)
    
    inc_grid = args.inc
    if len(inc_grid) > 1:
        angles = np.arange(4, 82, 2, dtype=float)
        inc_grid = np.array_split(angles, args.inc[0])[args.inc[1]]
    if args.start_inc:
        inc_grid = inc_grid[inc_grid >= args.start_inc]
    seeds = args.seeds if args.seeds else np.atleast_1d(hparams['seed'])
        
    for inclination in tqdm(inc_grid, desc='inc'):
        # Compute geodesics paths
        geos = bhnerf.kgeo.image_plane_geos(
            spin, np.deg2rad(inclination), 
            num_alpha=num_alpha, num_beta=num_beta, 
            alpha_range=[-fov_M/2, fov_M/2],
            beta_range=[-fov_M/2, fov_M/2]
        )
        geos = geos.fillna(0.0)

         # Keplerian velocity and Doppler boosting
        rot_sign = {'cw': -1, 'ccw': 1}
        Omega = rot_sign[Omega_dir] * np.sqrt(geos.M) / (geos.r**(3/2) + geos.spin * np.sqrt(geos.M))
        umu = bhnerf.kgeo.azimuthal_velocity_vector(geos, Omega)
        g = bhnerf.kgeo.doppler_factor(geos, umu)

        # Magnitude normalized magnetic field in fluid-frame
        b = bhnerf.kgeo.magnetic_field_fluid_frame(geos, umu, **b_consts)
        domain = np.bitwise_and(np.bitwise_and(np.abs(geos.z) < z_width, geos.r > rmin), geos.r < rmax)
        b_mean = np.sqrt(np.sum(b[domain]**2, axis=-1)).mean()
        b /= b_mean

        # Polarized emission factors (including parallel transport)
        de_rot_model = np.deg2rad(de_rot_angle + 20.0)
        J = np.nan_to_num(bhnerf.kgeo.parallel_transport(geos, umu, g, b, Q_frac=Q_frac, V_frac=0), 0.0)
        J_rot = bhnerf.emission.rotate_evpa(J, de_rot_model)

        t_injection = -float(geos.r_o + fov_M/4)
        raytracing_args = bhnerf.network.raytracing_args(geos, Omega, t_injection, t_frames[0], J_rot)
            
        for seed in tqdm(seeds, desc='seed'):
            runname = basename.format(inclination, seed)
            writer = bhnerf.optimization.SummaryWriter(logdir=os.path.join(log_dir, runname))
            log_fns = [
                LogFn(lambda opt: writer.add_scalar('log_loss/train', np.log10(np.mean(opt.loss)), global_step=opt.step)), 
                LogFn(lambda opt: writer.recovery_3d(fov_M)(opt), log_period=log_period),
                LogFn(lambda opt: writer.plot_lc_datafit(opt, target, ['I', 'Q', 'U'], t_frames, batchsize=20), log_period=log_period)
            ]
            
            hparams['seed'] = seed
            optimizer = bhnerf.optimization.Optimizer(
                hparams, predictor, raytracing_args, 
                save_period=save_period, 
                checkpoint_dir=os.path.join(checkpoint_dir, runname)
            )

            optimizer.run(batchsize, train_step, raytracing_args, log_fns=log_fns)
            writer.close()




