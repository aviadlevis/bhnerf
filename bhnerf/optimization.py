import numpy as np
import jax, flax
from jax import numpy as jnp
import bhnerf
from bhnerf import kgeo
import optax
import os
from flax.training import train_state
from flax.training import checkpoints
from astropy import units
    
def run(runname, hparams, predictor, train_pstep, geos, Omega, rmax, t_injection, batched_args, J=1.0, 
        emission_true=None, vis_res=64, log_period=100, save_period=1000):
    """
    Run a gradient descent optimization over the network parameters (3D emission) 
    
    Parameters
    ----------
    runname: str, 
        String used for logging and saving checkpoints.
    hparams: dict, 
        'num_iters': int, number of iterations, 
        'lr_init': float, initial learning rate,
        'lr_final': float, final learning rate, 
        'batchsize': int, should be an integer factor of the number of GPUs used
    predictor: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    train_pstep: jax.pmap, 
        A parallel mapping of the training function (e.g. train_step_image or train_step_eht)
    geos: xr.Dataset
        A dataset specifying geodesics (ray trajectories) ending at the image plane.
    Omega: xr.DataArray
        A dataarray specifying the keplerian velocity field
    rmax: float, 
        The maximum radius for recovery
    t_injection: float, 
        Time of hotspot injection in M units.
    batched_args: BatchedTemporalArgs instance,
        Arguments taken by the training_step function which should be batched along the first dimension. 
        See BatchedTemporalArgs classmethods for more information.
    J: np.array(shape=(3,...)), default=1.0
        Stokes vector scaling factors including parallel transport (I, Q, U). J=1.0 gives non-polarized emission.
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
    
    # Image rendering arguments
    t_units = batched_args.t_units
    coords = jnp.array([geos.x, geos.y, geos.z])
    umu = kgeo.azimuthal_velocity_vector(geos, Omega)
    g = jnp.array(kgeo.doppler_factor(geos, umu))
    Omega = jnp.array(Omega)
    dtau = jnp.array(geos.dtau)
    Sigma = jnp.array(geos.Sigma)
    t_start_obs = batched_args.t_start_obs
    t_geos = jnp.array(geos.t)
    rmin = geos.r.min().data
    rendering_args = [coords, Omega, J, g, dtau, Sigma, t_start_obs, t_geos, t_injection, rmin, rmax]
    
    non_jit_args = [t_units]
    if (train_pstep.__name__ != 'train_step_image') and (batched_args.dtype is not None):
        non_jit_args += [batched_args.dtype]
    
    # Grid for visualization (no interpolation is added for easy comparison)
    if emission_true is not None:
        vis_coords = np.array(np.meshgrid(np.linspace(emission_true.x[0], emission_true.x[-1], emission_true.shape[0]),
                                          np.linspace(emission_true.y[0], emission_true.y[-1], emission_true.shape[1]),
                                          np.linspace(emission_true.z[0], emission_true.z[-1], emission_true.shape[2]),
                                          indexing='ij'))
    else:
        grid_1d = np.linspace(-rmax, rmax, vis_res)
        vis_coords = np.array(np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij'))

    params = predictor.init(jax.random.PRNGKey(1), t_start_obs, t_units, coords, Omega, t_start_obs, t_geos, t_injection)['params']
    
    tx = optax.adam(learning_rate=optax.polynomial_schedule(hparams['lr_init'], hparams['lr_final'], 1, hparams['num_iters']))
    # tx = optax.adam(learning_rate=optax.polynomial_schedule(hparams['lr_init'], hparams['lr_final'], 1, hparams['num_iters']))
    if 'lr_inject' in hparams.keys():
        tx = optax.chain(
            optax.masked(optax.adam(learning_rate=hparams['lr_inject']), mask=flattened_traversal(lambda path, _: path[-1] == 't_injection')),
            optax.masked(tx, mask=flattened_traversal(lambda path, _: path[-1] != 't_injection')),
        )
        
    state = train_state.TrainState.create(apply_fn=predictor.apply, params=params.unfreeze(), tx=tx)  # TODO(pratul): this unfreeze feels sketchy

    # Restore saved checkpoint
    # if np.isscalar(save_period): state = checkpoints.restore_checkpoint(checkpoint_dir, state)
    init_step = 1 + state.step
    
    # Replicate state for multiple gpus
    state = flax.jax_utils.replicate(state)
    with SummaryWriter(logdir=logdir) as writer:
        if emission_true is not None: writer.add_images('emission/true', bhnerf.utils.intensity_to_nchw(emission_true), global_step=0)

        for i in tqdm(range(init_step, init_step + hparams['num_iters']), desc='iteration'):
            loss, state, images = train_pstep(state, *non_jit_args, *batched_args.sample(hparams['batchsize']), *rendering_args)

            # Log the current state on TensorBoard
            writer.add_scalar('log_loss/train', np.log10(np.mean(loss)), global_step=i)
            if (i == 1) or ((i % log_period) == 0) or (i ==  init_step + hparams['num_iters']):
                emission_grid = bhnerf.network.sample_3d_grid(state.apply_fn, state.params, rmin, rmax, coords=vis_coords)
                writer.add_images('emission/estimate', bhnerf.utils.intensity_to_nchw(emission_grid), global_step=i)
                if 'lr_inject' in hparams.keys(): writer.add_scalar('t_injection', float(current_state.params['t_injection']), global_step=i)
                if emission_true is not None:
                    writer.add_scalar('emission/mse', bhnerf.utils.mse(emission_true.data, emission_grid), global_step=i)
                    writer.add_scalar('emission/psnr', bhnerf.utils.psnr(emission_true.data, emission_grid), global_step=i)

            # Save checkpoints occasionally
            if np.isscalar(save_period) and ((i % save_period == 0) or (i ==  init_step + hparams['num_iters'])):
                current_state = jax.device_get(flax.jax_utils.unreplicate(state))
                checkpoints.save_checkpoint(checkpoint_dir, current_state, int(i), keep=5)
    return current_state

def total_movie_loss(runname, batchsize, predictor, test_pstep, target, t_frames, geos, rmax, 
                     t_injection, lr_inject=True, return_frames=False, batched_args=[], args=[]):
    """
    This function chunks up the movie into frames which fit on GPUs and sums the total loss over all frames 
    
    Parameters
    ----------
    runname: str, 
        String used for logging and saving checkpoints.
    batchsize: int, 
        Should be an integer factor of the number of GPUs used
    predictor: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    trest_pstep: jax.pmap, 
        A parallel mapping of the testing function (e.g. test_step_image or test_step_eht)
    target: array, 
        Target to fit the model to (image pixels or eht observations depending on train_pstep)
    t_frames: array, 
        Array of time for each image frame with astropy.units, 
        Time units for t_frames.
    geos: xr.Dataset
        A dataset specifying geodesics (ray trajectories) ending at the image plane.
    rmax: float, 
        The maximum radius for recovery
    t_injection: float, 
        Time of hotspot injection in M units.
    lr_inject: bool, default=True
        True for optimized injection time. 
    return_frames: bool, default=False, 
        Return the estimated movie frames
    batched_args: list,
        Additional arguments taken by the training_step function which should be batched along the first dimension (e.g. Fourier matrices)
    args: list,
        Additional arguments taken by the training_step function
        
    Returns
    -------
    total_loss: float,
        The total loss over all frames.
    """
        
    checkpoint_dir = os.path.join('../checkpoints', runname)
    if not os.path.exists(checkpoint_dir): 
        raise AttributeError('checkpoint directory does not exist: {}'.format(checkpoint_dir))
        
    # Image rendering arguments
    t_units = t_frames.unit
    coords = jnp.array([geos.x, geos.y, geos.z])
    Omega = jnp.array(geos.Omega)
    g = jnp.array(geos.g.fillna(0.0))
    dtau = jnp.array(geos.dtau)
    Sigma = jnp.array(geos.Sigma)
    t_start_obs = t_frames[0]
    t_geos = jnp.array(geos.t)
    rmin = geos.r.min().data
    rendering_args = [coords, Omega, g, dtau, Sigma, t_start_obs, t_geos, t_injection, rmin, rmax]
    
    params = predictor.init(jax.random.PRNGKey(1), t_frames[0], t_units, coords, Omega, t_start_obs, t_geos, t_injection)['params']
    tx = optax.adam(learning_rate=optax.polynomial_schedule(1, 1, 1, 1))
    if 'lr_inject':
        tx = optax.chain(
            optax.masked(optax.adam(learning_rate=0), mask=flattened_traversal(lambda path, _: path[-1] == 't_injection')),
            optax.masked(tx, mask=flattened_traversal(lambda path, _: path[-1] != 't_injection')),
        )
    state = train_state.TrainState.create(apply_fn=predictor.apply, params=params.unfreeze(), tx=tx)  # TODO(pratul): this unfreeze feels sketchy
    state = checkpoints.restore_checkpoint(checkpoint_dir, state)
    state = flax.jax_utils.replicate(state)
    
    # Split times according to batchsize
    nt = len(t_frames)
    nt_tilde = nt - nt % batchsize
    indices = np.array_split(np.arange(0, nt_tilde), nt_tilde/batchsize)
    indices.append(np.arange(nt-nt % batchsize, nt))

    # Aggregate loss over all t_frames
    total_loss = 0.0
    frames = []
    for inds in indices:
        if (inds.size == 0): break
        bargs = [shard(arg[inds, ...]) for arg in batched_args]
        loss, images = test_pstep(state, shard(target[inds, ...]), shard(t_frames[inds, ...]), t_units, *rendering_args, *bargs, *args)
        total_loss += float((batchsize // jax.device_count()) * loss.sum())
        if return_frames:
            frames.append(images.reshape(-1, geos.alpha.size, geos.beta.size))
            
    output = total_loss
    if return_frames:
        output = (total_loss, np.concatenate(frames))
        
    return output

class TemporalBatchedArgs(object):
    
    def __init__(self, t_frames, args=[], dtype=None):
        self.t_frames = t_frames
        if not isinstance(args, list): args = [args]
        self.num_frames = len(t_frames)
        assert all([self.num_frames == arg.shape[0] for arg in args])  
        args.append(t_frames)
        self.args = args
        self.default_t_units = units.hr
        self.dtype = dtype

    def sample(self, batchsize, replace=False):
        batch = np.random.choice(range(self.num_frames), batchsize, replace=replace)
        batched_args = [shard(arg[batch,...]) for arg in self.args]
        return batched_args
    
    @classmethod
    def train_step_image(cls, t_frames, image_plane):
        """
        batched arguments for training directly on image plane data.
        
        Parameters
        ----------
        t_frames: array, 
            Array of time for each image frame with astropy.units. If no units are included defaults to Hours.
        image_plane: np.array
            A movie array with image-plane frames (first axis should have same length as t_frames)
            For stokes concatenate movies along the second axis: np.stack([I, Q, U], axis=1)
        """
        return cls(t_frames, image_plane)
    
    @classmethod
    def train_step_eht(cls, t_frames, obs, image_fov, image_size, chisqdata, pol='I'):
        """
        batched arguments for training on eht observations.
        
        Parameters
        ----------
        t_frames: array, 
            Array of time for each image frame with astropy.units. If no units are included defaults to Hours.
        target: array, 
            Target measurement values to fit the model to (e.g. complex visibilties or closure phases). 
        sigma: array,
            An array of standard deviations for each measurement
        A: array,
            An array of discrete time fourier transform matrices for each frame time
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
            
        return cls(t_frames, [target, sigma, A], dtype)
    
    
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