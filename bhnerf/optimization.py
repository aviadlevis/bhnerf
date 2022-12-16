import numpy as np
import jax, flax
from jax import numpy as jnp
import bhnerf
from bhnerf import kgeo
import os
from astropy import units   
from flax.training import checkpoints

def run(runname, batchsize, num_iters, state, train_step, raytracing_args, rmin, rmax, emission_true=None, vis_res=64, log_period=100, save_period=1000):
    """
    Run a gradient descent optimization over the network parameters (3D emission) 
    
    Parameters
    ----------
    runname: str, 
        String used for logging and saving checkpoints.
    num_iters: int, 
        number of iterations, 
    batchsize: int, 
        should be an integer factor of the number of GPUs used
    state: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    train_step: TrainStep, 
        A conatiner for parallel mapping of the training function
    raytracing_args: OrderedDict, 
        dictionary with arguments used for ray tracing
    rmin: float, 
        The minim radius for visualization
    rmax: float, 
        The maximum radius for recovery
    emission_true: array, 
        A ground-truth array of 3D emission for evalutation metrics 
    vis_res: int, default=64
        Resolution (number of grid points in x,y,z) at which to visualize the contiuous 3D emission
    log_period: int, default=100
        TensorBoard logging every `log_period` iterations
    save_period: int, default=1000
        Save checkpoint every `save_period` iterations
        
    Returns
    -------
    current_state: flax.training.train_state.TrainState, 
        The training state holding the network parameters at the end of the optimization
    """
    
    from tensorboardX import SummaryWriter
    from tqdm.notebook import tqdm
    
    # Logging parameters
    checkpoint_dir = '../checkpoints/{}'.format(runname)
    logdir = '../runs/{}'.format(runname)
    
    # Grid for visualization (no interpolation is added for easy comparison)
    if emission_true is not None:
        vis_coords = np.array(np.meshgrid(np.linspace(emission_true.x[0], emission_true.x[-1], emission_true.shape[0]),
                                          np.linspace(emission_true.y[0], emission_true.y[-1], emission_true.shape[1]),
                                          np.linspace(emission_true.z[0], emission_true.z[-1], emission_true.shape[2]),
                                          indexing='ij'))
    else:
        grid_1d = np.linspace(-rmax, rmax, vis_res)
        vis_coords = np.array(np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij'))

    init_step = flax.jax_utils.unreplicate(state.step) + 1
    with SummaryWriter(logdir=logdir) as writer:
        if emission_true is not None: writer.add_images('emission/true', bhnerf.utils.intensity_to_nchw(emission_true), global_step=0)

        for i in tqdm(range(init_step, init_step + num_iters), desc='iteration'):
            batch_indices = train_step.args[0].sample(batchsize)
            loss, state, images = train_step(state, raytracing_args, indices=batch_indices)

            # Log the current state on TensorBoard
            writer.add_scalar('log_loss/train', np.log10(np.mean(loss)), global_step=i)
            if (i == 1) or ((i % log_period) == 0) or (i ==  init_step + num_iters):
                emission_grid = bhnerf.network.sample_3d_grid(state.apply_fn, state.params, rmin, rmax, coords=vis_coords)
                writer.add_images('emission/estimate', bhnerf.utils.intensity_to_nchw(emission_grid), global_step=i)
                # if 'lr_inject' in hparams.keys(): writer.add_scalar('t_injection', float(current_state.params['t_injection']), global_step=i)
                if emission_true is not None:
                    writer.add_scalar('emission/mse', bhnerf.utils.mse(emission_true.data, emission_grid), global_step=i)
                    writer.add_scalar('emission/psnr', bhnerf.utils.psnr(emission_true.data, emission_grid), global_step=i)

            # Save checkpoints occasionally
            if np.isscalar(save_period) and ((i % save_period == 0) or (i ==  init_step + num_iters)):
                current_state = jax.device_get(flax.jax_utils.unreplicate(state))
                checkpoints.save_checkpoint(checkpoint_dir, current_state, int(i), keep=5)
    return current_state

def loop_over_inclination(geos_base, inc_grid, b_consts, runname, hparams, predictor, train_step, rmax, t_injection,   
                          emission_true=None, vis_res=64, log_period=100):
    """
    Run a gradient descent optimization over the network parameters (3D emission) 
    
    Parameters
    ----------
    inc_grid, array,
        A grid of inclination values to loop over.
    b_consts: list, 
        Constants magnetic field components: b_consts = [b_r, b_th, b_ph]
    runname: str, 
        String used for logging and saving checkpoints.
    hparams: dict, 
        'num_iters': int, number of iterations, 
        'lr_init': float, initial learning rate,
        'lr_final': float, final learning rate, 
        'batchsize': int, should be an integer factor of the number of GPUs used
    predictor: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    train_step: TrainStep, 
        A conatiner for parallel mapping of the training function
    rmax: float, 
        The maximum radius for recovery
    t_injection: float, 
        Time of hotspot injection in M units.
    emission_true: array, 
        A ground-truth array of 3D emission for evalutation metrics 
    vis_res: int, default=64
        Resolution (number of grid points in x,y,z) at which to visualize the contiuous 3D emission
    log_period: int, default=100
        TensorBoard logging every `log_period` iterations
    """
    from tqdm.notebook import tqdm
    
    for inclination in tqdm(np.atleast_1d(inc_grid), desc='inc'):
        geos = kgeo.image_plane_geos(
            float(geos_base.spin), inclination, 
            num_alpha=geos_base.alpha.size, num_beta=geos_base.beta.size, 
            alpha_range=[float(geos_base.alpha[0]), float(geos_base.alpha[-1])],
            beta_range=[float(geos_base.beta[0]), float(geos_base.beta[-1])]
        )
        geos = geos.fillna(0.0)
        
        # Keplerian prograde velocity field
        Omega = np.sign(geos.spin + np.finfo(float).eps) * np.sqrt(geos.M) / (geos.r**(3/2) + geos.spin * np.sqrt(geos.M))
        umu = kgeo.azimuthal_velocity_vector(geos, Omega)
        g = kgeo.doppler_factor(geos, umu)
        b = kgeo.magnetic_field(geos, *b_consts) 
        J = np.nan_to_num(kgeo.parallel_transport(geos, umu, g, b), 0.0)

        runname_inc = runname + '.inc_{:.1f}'.format(np.rad2deg(inclination))
        run(runname_inc, hparams, predictor, train_step, geos, Omega, rmax, t_injection, J, emission_true, vis_res, log_period, hparams['num_iters'])

def total_movie_loss(checkpoint_dir, batchsize, state, train_step, raytracing_args, rmax, return_frames=False):
    """
    This function chunks up the movie into frames which fit on GPUs and sums the total loss over all frames 
    
    Parameters
    ----------
    checkpoint_dir: str, 
        Path to directory with latest checkpoint
    batchsize: int, 
        batchsize should be an integer factor of the number of GPUs used
    state: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    train_step: TrainStep, 
        A conatiner for parallel mapping of the training/testing function
    raytracing_args: OrderedDict, 
        dictionary with arguments used for ray tracing
    rmax: float, 
        The maximum radius for recovery
    return_frames: bool, default=False, 
        Return the estimated movie frames

    Returns
    -------
    total_loss: float,
        The total loss over all frames.
    """
    if not os.path.exists(checkpoint_dir): 
        raise AttributeError('checkpoint directory does not exist: {}'.format(checkpoint_dir))
    
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
            frames.append(images.reshape(-1, *images.shape[2:]))
            
    output = total_loss
    if return_frames:
        output = (total_loss, np.concatenate(frames[:nt]))
        
    return output


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
        call_fn = self.grad_pmap if update_state == True else self.test_pmap
        for i in range(self.num_losses):
            loss, state, images = call_fn[i](state, self.t_units, self.dtype[i], *self.args[i][indices], *raytracing_args.values(), self.scale[i])
            total_loss += loss
        return total_loss, state, images
    
    def __add__(self, other):
        dtype = np.append(self.dtype, other.dtype)
        args = np.append(self.args, other.args)
        grad_pmap = np.append(self.grad_pmap, other.grad_pmap)
        test_pmap = np.append(self.test_pmap, other.test_pmap)
        scale = np.append(self.scale, other.scale)
        return TrainStep(dtype, args, grad_pmap, test_pmap, scale)
    
    @classmethod
    def image(cls, t_frames, target, sigma=1.0, scale=1.0, dtype='full'):
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
        sigma = sigma * np.ones_like(target) if np.isscalar(sigma) else sigma
        args = TemporalBatchedArgs(t_frames, [target, sigma])
        grad_pmap = jax.pmap(bhnerf.network.gradient_step_image,
                             axis_name='batch', 
                             in_axes=(0, None, None, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None),
                             static_broadcasted_argnums=(1, 2))
        test_pmap = jax.pmap(bhnerf.network.test_image,
                     axis_name='batch', 
                     in_axes=(0, None, None, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None),
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
                in_axes=(0, None, None, 0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None), 
                static_broadcasted_argnums=(1, 2)) 
        
        test_pmap = jax.pmap(bhnerf.network.test_eht, 
                axis_name='batch', 
                in_axes=(0, None, None, 0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None), 
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
    
    
def shard(xs):
    """Split data into shards for multiple devices along the first dimension."""
    return jax.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)

def flattened_traversal(fn):
    def mask(data):
        flat = flax.traverse_util.flatten_dict(data)
        return flax.traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})
    return mask

