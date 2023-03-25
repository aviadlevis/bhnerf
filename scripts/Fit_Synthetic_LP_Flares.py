"""
This script fits synthetic linear polarization data of Sagittarius A*. 
"""
import bhnerf
from bhnerf.optimization import LogFn
from astropy import units
import jax
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
import argparse
import ruamel.yaml as yaml
import warnings
warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_path', type=str, 
                        help='Path to data configuration (.yaml) file. \n\
                              The yaml file containts paths to lightcurve data (.csv) and ground-truth flare volume (.nc).')
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
    parser.add_argument('--config_path', type=str, default='Fit_Synthetic_LP_Flares.yaml', 
                        help='Path to recovery configuration YAML file')
    args = parser.parse_args()
        
    # Update number of GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args.gpu]) if args.gpu else ''
    print('running on devices: {}'.format(jax.devices()))
    return args

if __name__ == "__main__":
    
    basename = 'inc_{:.1f}.seed_{}'
    
    args = parse_args()
    with open(args.yaml_path, 'r') as stream:
        simulation_params = yaml.load(stream, Loader=yaml.Loader)
    with open(args.config_path, 'r') as stream:
        recovery_params = yaml.load(stream, Loader=yaml.Loader)
    locals().update(simulation_params['model'])
    locals().update(recovery_params['model'])
    locals().update(recovery_params['optimization']) 
    
    rmax = fov_M / 2
    if rmin == 'ISCO': rmin = float(bhnerf.constants.isco_pro(spin))
    J_inds = [['I', 'Q', 'U'].index(s) for s in stokes]
    recovery_params['model'].update(rmax=rmax, rmin=rmin)
    
    data_path = Path(simulation_params['lightcurve_path'])
    lightcurves_df = pd.read_csv(data_path)
    
    # Save recovery simulation parameters 
    sim_name = simulation_params['name']
    recovery_dir = data_path.parent.joinpath('recovery/{}'.format(sim_name))
    recovery_dir.mkdir(parents=True, exist_ok=True)
    params = {'simulation': simulation_params, 'recovery': recovery_params}
    with open(recovery_dir.joinpath('params.yaml'), 'w') as file:
        yaml.dump(params, file, default_flow_style=False)
    
    # Load ground truth flare for comparison 
    flare_path = Path(simulation_params['flare_path'])
    emission_flare = xr.load_dataarray(flare_path)

    # Setup model for recovery
    target, t_frames = np.array(lightcurves_df[stokes]), np.array(lightcurves_df['t'])
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
        J = np.nan_to_num(bhnerf.kgeo.parallel_transport(geos, umu, g, b, Q_frac=Q_frac, V_frac=0), 0.0)[J_inds]
        
        t_injection = -float(geos.r_o + fov_M/4)
        raytracing_args = bhnerf.network.raytracing_args(geos, Omega, t_injection, t_start_obs*units.hr, J)
        
        for seed in tqdm(seeds, desc='seed'):
            runname = basename.format(inclination, seed)
            writer = bhnerf.optimization.SummaryWriter(logdir=recovery_dir.joinpath(runname))
            writer.add_images('emission/true', bhnerf.utils.intensity_to_nchw(emission_flare), global_step=0)
            log_fns = [
                LogFn(lambda opt: writer.add_scalar('log_loss/train', np.log10(np.mean(opt.loss)), global_step=opt.step)), 
                LogFn(lambda opt: writer.recovery_3d(fov_M, emission_true=emission_flare)(opt), log_period=log_period),
                LogFn(lambda opt: writer.plot_lc_datafit(opt, target, stokes, t_frames, batchsize=20), log_period=log_period)
            ]
            
            hparams['seed'] = seed
            optimizer = bhnerf.optimization.Optimizer(
                hparams, predictor, raytracing_args, 
                save_period=save_period, 
                checkpoint_dir=recovery_dir.joinpath(runname)
            )

            optimizer.run(batchsize, train_step, raytracing_args, log_fns=log_fns)
            writer.close()




