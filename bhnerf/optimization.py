import numpy as np
import jax, flax
from jax import numpy as jnp
import bhnerf
from bhnerf import kgeo
import os
from astropy import units   
from flax.training import checkpoints
from tqdm.auto import tqdm
import tensorboardX
import matplotlib.pyplot as plt


def total_movie_loss(batchsize, state, train_step, raytracing_args, return_frames=False):
    """
    This function chunks up the movie into frames which fit on GPUs and sums the total loss over all frames 
    
    Parameters
    ----------
    batchsize: int, 
        batchsize should be an integer factor of the number of GPUs used
    state: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    train_step: TrainStep, 
        A conatiner for parallel mapping of the training/testing function
    raytracing_args: OrderedDict, 
        dictionary with arguments used for ray tracing
    return_frames: bool, default=False, 
        Return the estimated movie frames

    Returns
    -------
    total_loss: float,
        The total loss over all frames.
    """
    # Split times according to batchsize and number of devices
    nt = train_step.args[0].num_frames
    
    if nt % jax.device_count():
        raise AttributeError('batch size should be an integer multiplication of the device number')
        
    nt_tilde = nt - nt % batchsize
    indices = np.array_split(np.arange(0, nt_tilde), nt_tilde/batchsize)

    nt_tilde1 = int(jax.device_count() * np.ceil(nt/jax.device_count()))
    indices.append(np.arange(nt_tilde, nt_tilde1) % nt)

    # Aggregate loss over all frames
    frames, total_loss = [], 0.0
    for inds in indices:
        if (inds.size == 0): break
        loss, state, images = train_step(state, raytracing_args, inds, update_state=False)
        total_loss += loss.sum()
        
        if return_frames:
            # non polarized 
            if np.isscalar(raytracing_args['J']): 
                frames.append(images.reshape(-1, *images.shape[-2:]))
            else:
                frames.append(images.reshape(-1, *images.shape[-3:]))
            
    output = total_loss / nt
    if return_frames:
        output = (output, np.concatenate(frames[:nt]))
        
    return output

class Optimizer(object):
    """
    Run a gradient descent optimization over the network parameters (3D emission) 
    
    Parameters
    ----------
    hparams: dict, 
        'num_iters': int, number of iterations, 
        'lr_init': float, initial learning rate, defualt=1e-4
        'lr_final': float, final learning rate, default=1e-6
        'lr_inject': float, learning rate for hotspot injection time, default=None
        'seed': random seed for NN initialization, default=1
    predictor: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    raytracing_args: OrderedDict, 
        dictionary with arguments used for ray tracing
    save_period: int, default=-1
        Save checkpoint every `save_period` iterations. Negative value means last iteration only.
    save_period: str, default=''
        Checkpoint directory. '' means no checkpoint saved
    keep: int, default=5,
        How many checkpoint history to keep.
    """
    def __init__(self, hparams, predictor, raytracing_args, save_period=-1, checkpoint_dir='', keep=5):
        self.step = 0 
        self.init_step = 0
        self.num_iters = hparams['num_iters']
        self.checkpoint_dir = checkpoint_dir
        self.save_period = self.num_iters if save_period<0 else save_period
        self.loss = np.inf
        self.keep = keep
        self.seed = hparams.get('seed', 1)
        
        params = predictor.init_params(raytracing_args, seed=self.seed)
        self.state = predictor.init_state(
            params=params, 
            num_iters=self.num_iters, 
            lr_init=hparams.get('lr_init', 1e-4),
            lr_final=hparams.get('lr_final', 1e-6),
            lr_inject=hparams.get('lr_inject', None),
            checkpoint_dir=self.checkpoint_dir
        )
        
        if checkpoint_dir != '':
            predictor.save_params(checkpoint_dir)
        
    def log(self):
        for log_fn in self.log_fns:
            log_fn(self)
            
    def save_checkpoint(self):
        if (self.checkpoint_dir != '') and ((self.step % self.save_period == 0) or (self.step ==  self.final_step)):
            current_state = jax.device_get(flax.jax_utils.unreplicate(self.state))
            checkpoints.save_checkpoint(self.checkpoint_dir, current_state, int(self.step), keep=self.keep)
        
    def run(self, batchsize, train_step, raytracing_args, log_fns=[]):
        
        self.init_step = flax.jax_utils.unreplicate(self.state.step) + 1
        self.final_step = self.init_step + self.num_iters
        self.log_fns = log_fns = np.atleast_1d(log_fns)
        self.train_step = train_step
        self.raytracing_args = raytracing_args
        
        try:
            for self.step in tqdm(range(self.init_step, self.final_step), desc='iteration'):
                batch_indices = train_step.args[0].sample(batchsize)
                self.loss, self.state, images = train_step(self.state, raytracing_args, indices=batch_indices)
                self.log()
                self.save_checkpoint()
                
        except KeyboardInterrupt:
            return

    @property
    def params(self):
        return jax.device_get(flax.jax_utils.unreplicate(self.state)).params
    
class TrainStep(object):
    
    def __init__(self, dtype, args, grad_pmap, test_pmap, scale):
    
        self.dtype = np.atleast_1d(dtype)
        self.args = np.atleast_1d(args)
        self.grad_pmap = np.atleast_1d(grad_pmap)
        self.test_pmap = np.atleast_1d(test_pmap)
        self.scale = np.atleast_1d(scale)
        
        if np.any([arg.t_units != units.hr for arg in self.args]):
            raise AttributeError('only hr units supported')
            
        assert self.dtype.size == self.args.size == self.test_pmap.size == \
            self.grad_pmap.size == self.scale.size, 'input list sizes are not equal'
        self.num_losses = self.dtype.size
                         

    def __call__(self, state, raytracing_args, indices, update_state=True):
        total_loss = 0.0
        total_images = 0.0
        raytracing_args = np.atleast_1d(raytracing_args)
        if update_state:
            call_fn = self.grad_pmap 
            raytracing_args = [raytracing_args[np.random.choice(len(raytracing_args))]]
        else:
            call_fn = self.test_pmap
        
        # Subpixel loop (stochastic subpixel selection for training)
        for rt_arg in raytracing_args:
            for i in range(self.num_losses):
                loss, state, images = call_fn[i](state, self.t_units, self.dtype[i], *self.args[i][indices], *rt_arg.values(), self.scale[i])
                total_loss += loss / len(raytracing_args)
                total_images += images / len(raytracing_args)
        return total_loss, state, total_images
    
    def __add__(self, other):
        dtype = np.append(self.dtype, other.dtype)
        args = np.append(self.args, other.args)
        grad_pmap = np.append(self.grad_pmap, other.grad_pmap)
        test_pmap = np.append(self.test_pmap, other.test_pmap)
        scale = np.append(self.scale, other.scale)
        return TrainStep(dtype, args, grad_pmap, test_pmap, scale)
    
    @classmethod
    def image(cls, t_frames, target, sigma=1.0, offset=0.0, scale=1.0, dtype='full'):
        """
        Construct a training step for image plane measurements
        
        Parameters
        ----------
        t_frames: array, 
            Array of time for each image frame with astropy.units. If no units are included defaults to Hours.
        target: np.array
            Target: first axis should have same length as t_frames
            For stokes concatenate along the second axis: np.stack([I, Q, U], axis=1)
        sigma: array,
            An array of standard deviations for each pixel
        dtype: str, default='full',
            Currently supports 'full' or 'lc' (lightcurve)
        """
        sigma = sigma * np.ones_like(target)
        offset = offset * np.ones_like(target)
        args = TemporalBatchedArgs(t_frames, [target, sigma, offset])
        grad_pmap = jax.pmap(bhnerf.network.gradient_step_image,
                             axis_name='batch', 
                             in_axes=(0, None, None, 0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None),
                             static_broadcasted_argnums=(1, 2))
        test_pmap = jax.pmap(bhnerf.network.test_image,
                     axis_name='batch', 
                     in_axes=(0, None, None, 0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None),
                     static_broadcasted_argnums=(1, 2))
        return cls(dtype, args, grad_pmap, test_pmap, scale)
    
    @classmethod
    def eht(cls, t_frames, obs, image_fov, image_size, chisqdata, pol='I', scale=1.0):
        """
        Construct a training step for eht measurements
        
        Parameters
        ----------
        t_frames: array, 
            Array of time for each image frame with astropy.units. If no units are included defaults to Hours.
        obs: ehtim.Obsdata
            ehtim Observation object
        image_fov: float,
            Image fov in radians
        image_size: int,
            Number of image pixels
        chisqdata: ehtim.imaging.imager_utils.chisqdata_<dtype>
            ehtim method to extract (target, sigma, A) for this observation.
        """
        from ehtim.image import make_square
        dtype = chisqdata.__name__.split('_')[-1]
        pol_types = ['I', 'Q', 'U'] 
        
        obs_frames = obs.split_obs(t_gather=(t_frames[-1]-t_frames[0]).to('s').value / (len(t_frames)+1))
        prior = make_square(obs, image_size, image_fov)
        
        target, sigma, A = [], [], [] 
        for p in np.atleast_1d(pol):
            if p not in pol_types: raise AttributeError('pol ({}) not in supported pol_types: {},{},{}'.format(p, *pol_types))
            target_p, sigma_p, A_p = [np.array(out) for out in zip(*[chisqdata(obs, prior, mask=[], pol=p) for obs in obs_frames])]
            target.append(target_p)
            sigma.append(sigma_p)
            A.append(A_p)
        target, sigma, A = np.squeeze(np.stack(target, axis=1)), np.squeeze(np.stack(sigma, axis=1)), np.squeeze(np.stack(A, axis=1))
        
        # ehtim cphases are in degrees and not radians
        if dtype == 'cphase':
            target, sigma = np.deg2rad(target), np.deg2rad(sigma)
        
        args = TemporalBatchedArgs(t_frames, [target, sigma, A])
        grad_pmap = jax.pmap(bhnerf.network.gradient_step_eht, 
                axis_name='batch', 
                in_axes=(0, None, None, 0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None), 
                static_broadcasted_argnums=(1, 2)) 
        
        test_pmap = jax.pmap(bhnerf.network.test_eht, 
                axis_name='batch', 
                in_axes=(0, None, None, 0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None), 
                static_broadcasted_argnums=(1, 2)) 
        
        return cls(dtype, args, grad_pmap, test_pmap, scale)

    @property
    def t_units(self):
        return self.args[0].t_units
        
class TemporalBatchedArgs(object):
    
    def __init__(self, t_frames, args=[]):
        self.t_frames = t_frames
        if not isinstance(args, list): args = [args]
        self.num_frames = len(t_frames)
        assert all([self.num_frames == arg.shape[0] for arg in args])  
        args.append(t_frames)
        self.args = args
        self.default_t_units = units.hr

    def sample(self, batchsize, replace=False):   
        batch = np.random.choice(range(self.num_frames), batchsize, replace=replace)
        return batch
    
    def __getitem__(self, key):
        batched_args = [shard(arg[key,...]) for arg in self.args]
        return batched_args
    
    @property
    def t_units(self):
        if isinstance(self.t_frames, units.Quantity): 
            return self.t_frames.unit
        else: 
            return self.default_t_units
    
    @property
    def t_start_obs(self):
        return self.t_frames[0]

class SummaryWriter(tensorboardX.SummaryWriter):
    def __init__(self, logdir=None, comment='', purge_step=None, max_queue=10, flush_secs=120,
                 filename_suffix='', write_to_disk=True, log_dir=None, **kwargs):
        super().__init__(logdir, comment, purge_step, max_queue, flush_secs,
                         filename_suffix, write_to_disk, log_dir, **kwargs)
    
    def recovery_3d(self, fov, vis_res=64, emission_true=None):
        # Grid for visualization (no interpolation is added for easy comparison)
        if emission_true is not None:
            vis_coords = np.array(np.meshgrid(np.linspace(emission_true.x[0], emission_true.x[-1], emission_true.shape[0]),
                                              np.linspace(emission_true.y[0], emission_true.y[-1], emission_true.shape[1]),
                                              np.linspace(emission_true.z[0], emission_true.z[-1], emission_true.shape[2]),
                                              indexing='ij'))
        else:
            grid_1d = np.linspace(-fov/2, fov/2, vis_res)
            vis_coords = np.array(np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij'))
            
        def log_fn(opt):
            params = jax.device_get(flax.jax_utils.unreplicate(opt.state.params))
            emission_grid = bhnerf.network.sample_3d_grid(opt.state.apply_fn, params, coords=vis_coords)
            volume_slices = bhnerf.utils.intensity_to_nchw(emission_grid)
            self.add_images('emission/estimate', volume_slices, dataformats='NCWH', global_step=opt.step)
            if emission_true is not None:
                self.add_scalar('emission/mse', bhnerf.utils.mse(emission_true.data, emission_grid), global_step=opt.step)
                self.add_scalar('emission/psnr', bhnerf.utils.psnr(emission_true.data, emission_grid), global_step=opt.step)
        
        return log_fn
    
    def plot_lc_datafit(self, opt, name, train_step, target, stokes, t_frames=None, batchsize=20):
        
        plt.style.use('default')
        raytracing_args = np.atleast_1d(opt.raytracing_args)
        raytracing_args = raytracing_args[np.random.choice(len(raytracing_args))]
        loss, movie = bhnerf.optimization.total_movie_loss(
            batchsize, opt.state, train_step, raytracing_args, return_frames=True
        )
        lc_est = movie.sum(axis=(-1,-2))
        axes = bhnerf.visualization.plot_stokes_lc(target, stokes, t_frames, label='True')
        axes = bhnerf.visualization.plot_stokes_lc(lc_est, stokes, t_frames, axes=axes, fmt='x', color='r', label='Estimate')
        
        for ax in axes:
            ax.legend()
            
        self.add_figure('lightcurve/{}'.format(name), plt.gcf(), global_step=opt.step)
        self.add_scalar('datafit/{}'.format(name), np.log10(np.mean(loss)), global_step=opt.step)
    
class LogFn(object):
    def __init__(self, log_fn, log_period=1):
        self.log_period = log_period
        self.log_fn = log_fn
        
    def __call__(self, optimizer):
        if self.log_period > 0:
            if (optimizer.step == 1) or ((optimizer.step % self.log_period) == 0):
                self.log_fn(optimizer)
            
            
def shard(xs):
    """Split data into shards for multiple devices along the first dimension."""
    return jax.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)

