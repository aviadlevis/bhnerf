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
                        help='Path to ALMA April 11, 2017 Data (HI band) \n\
                              default=../data/Apr11_HI.dat')
    parser.add_argument('--config_path', type=str, default='Fit_ALMA_LP_Apr11_SgrA_Flare.yaml', 
                        help='Path to configuration YAML file\n\
                              default=Fit_ALMA_LP_Apr11_SgrA_Flare.yaml')
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
    
    basename = 'inc_{:.1f}.seed_{}'
    
    args, config = parse_args()
    locals().update(config['preprocess'])
    locals().update(config['model'])
    locals().update(config['optimization'])
    
    # Preprocess / split data in time into train/validation
    target, t_frames = bhnerf.alma.preprocess_data(**config['preprocess'])
    train_idx = t_frames <= t_start*units.hr + train_split*units.min
    val_idx = t_frames > t_start*units.hr + train_split*units.min
    data_train, data_val  = target[train_idx], target[val_idx] 
    t_train, t_val = t_frames[train_idx], t_frames[val_idx]
    
    # Setup model for recovery
    rmax = fov_M / 2
    if rmin == 'ISCO': rmin = float(bhnerf.constants.isco_pro(spin))
    train_step = bhnerf.optimization.TrainStep.image(t_train, data_train, sigma, dtype='lc')
    val_step = bhnerf.optimization.TrainStep.image(t_val, data_val, sigma, dtype='lc')
    predictor = bhnerf.network.NeRF_Predictor(rmax, rmin, rmax, z_width)
    rot_angle = np.deg2rad(de_rot_angle + 20.0)
    
    inc_grid = args.inc
    if len(inc_grid) > 1:
        angles = np.arange(4, 82, 2, dtype=float)
        inc_grid = np.array_split(angles, args.inc[0])[args.inc[1]]
    if args.start_inc:
        inc_grid = inc_grid[inc_grid >= args.start_inc]
    seeds = args.seeds if args.seeds else np.atleast_1d(hparams['seed'])
        
    for inclination in tqdm(inc_grid, desc='inc'):
        
        raytracing_args = bhnerf.alma.get_raytracing_args(np.deg2rad(inclination), spin, config['model'], rot_angle=rot_angle)

        for seed in tqdm(seeds, desc='seed'):
            
             # If already finished run --> skip iteration
            runname = basename.format(inclination, seed)
            if os.path.exists(os.path.join(checkpoint_dir, runname)):
                continue

            writer = bhnerf.optimization.SummaryWriter(logdir=os.path.join(log_dir, runname))
            log_fns = [
                LogFn(lambda opt: writer.add_scalar('log_loss/train', np.log10(np.mean(opt.loss)), global_step=opt.step)), 
                LogFn(lambda opt: writer.recovery_3d(fov_M)(opt), log_period=log_period),
                LogFn(lambda opt: writer.plot_lc_datafit(opt, 'training', train_step, data_train, ['I', 'Q', 'U'], t_train, batchsize=20), log_period=log_period),
                LogFn(lambda opt: writer.plot_lc_datafit(opt, 'validation', val_step, data_val, ['I', 'Q', 'U'], t_val, batchsize=20), log_period=log_period)
            ]
            
            hparams['seed'] = seed
            optimizer = bhnerf.optimization.Optimizer(
                hparams, predictor, raytracing_args, 
                save_period=save_period, 
                checkpoint_dir=os.path.join(checkpoint_dir, runname)
            )

            optimizer.run(batchsize, train_step, raytracing_args, log_fns=log_fns)
            writer.close()




