import numpy as np
import jax, flax
from jax import numpy as jnp
import bhnerf
from bhnerf import kgeo
import optax
import os
from flax.training import checkpoints
from astropy import units   
    
def run(runname, hparams, predictor, train_step, geos, Omega, rmax, t_injection, J=1.0, emission_true=None, vis_res=64, log_period=100, save_period=1000):
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
    train_step: TrainStep, 
        A conatiner for parallel mapping of the training function
    geos: xr.Dataset
        A dataset specifying geodesics (ray trajectories) ending at the image plane.
    Omega: xr.DataArray
        A dataarray specifying the keplerian velocity field
    rmax: float, 
        The maximum radius for recovery
    t_injection: float, 
        Time of hotspot injection in M units.
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
    
    # Grid for visualization (no interpolation is added for easy comparison)
    if emission_true is not None:
        vis_coords = np.array(np.meshgrid(np.linspace(emission_true.x[0], emission_true.x[-1], emission_true.shape[0]),
                                          np.linspace(emission_true.y[0], emission_true.y[-1], emission_true.shape[1]),
                                          np.linspace(emission_true.z[0], emission_true.z[-1], emission_true.shape[2]),
                                          indexing='ij'))
    else:
        grid_1d = np.linspace(-rmax, rmax, vis_res)
        vis_coords = np.array(np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij'))

    state, init_step = train_step.preprocess(hparams, predictor, geos, Omega, rmax, t_injection, J, checkpoint_dir)
    rmin = geos.r.min().data
    
    with SummaryWriter(logdir=logdir) as writer:
        if emission_true is not None: writer.add_images('emission/true', bhnerf.utils.intensity_to_nchw(emission_true), global_step=0)

        for i in tqdm(range(init_step, init_step + hparams['num_iters']), desc='iteration'):
            loss, state, images = train_step(state, batchsize=hparams['batchsize'])

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

def total_movie_loss(runname, hparams, predictor, test_step, geos, Omega, rmax, t_injection, batched_args, J=1.0, 
                     return_frames=False):
    """
    This function chunks up the movie into frames which fit on GPUs and sums the total loss over all frames 
    
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
    test_step: TrainStep, 
        A conatiner for parallel mapping of the training function
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
    return_frames: bool, default=False, 
        Return the estimated movie frames

    Returns
    -------
    total_loss: float,
        The total loss over all frames.
    """
        
    checkpoint_dir = os.path.join('../checkpoints', runname)
    if not os.path.exists(checkpoint_dir): 
        raise AttributeError('checkpoint directory does not exist: {}'.format(checkpoint_dir))
    rendering_args, non_jit_args, state, init_step = preprocessing(hparams, predictor, test_pstep, geos, Omega, rmax, t_injection, batched_args, J, checkpoint_dir)
    rmin = geos.r.min().data
    
    # Split times according to batchsize
    batchsize = hparams['batchsize']
    nt = batched_args.num_frames
    nt_tilde = nt - nt % batchsize
    indices = np.array_split(np.arange(0, nt_tilde), nt_tilde/batchsize)
    indices.append(np.arange(nt-nt % batchsize, nt))

    # Aggregate loss over all frames
    frames, total_loss = [], 0.0
    for inds in indices:
        if (inds.size == 0): break
        loss, images = test_step(state, *non_jit_args, *batched_args[inds], *rendering_args)
        total_loss += float((batchsize // jax.device_count()) * loss.sum())
        if return_frames:
            frames.append(images.reshape(-1, geos.alpha.size, geos.beta.size))
            
    output = total_loss
    if return_frames:
        output = (total_loss, np.concatenate(frames))
        
    return output

class TrainStep(object):
    
    def __init__(self, dtype, args, grad_pmap, scale):
    
        self.dtype = np.atleast_1d(dtype)
        self.args = np.atleast_1d(args)
        self.grad_pmap = np.atleast_1d(grad_pmap)
        self.scale = np.atleast_1d(scale)
        
        if np.any([arg.t_units != units.hr for arg in self.args]):
            raise AttributeError('only hr units supported')
            
        assert self.dtype.size == self.args.size == \
            self.grad_pmap.size == self.scale.size, 'input list sizes are not equal'
        self.num_losses = self.dtype.size
                         
    def preprocess(self, hparams, predictor, geos, Omega, rmax, t_injection, J, checkpoint_dir):
        
        from flax.training import train_state
        
        # Image rendering arguments
        t_units = self.args[0].t_units
        coords = jnp.array([geos.x, geos.y, geos.z])
        umu = kgeo.azimuthal_velocity_vector(geos, Omega)
        g = jnp.array(kgeo.doppler_factor(geos, umu))
        Omega = jnp.array(Omega)
        dtau = jnp.array(geos.dtau)
        Sigma = jnp.array(geos.Sigma)
        t_start_obs = self.args[0].t_start_obs
        t_geos = jnp.array(geos.t)
        rmin = geos.r.min().data
        
        self.rendering_args = [coords, Omega, J, g, dtau, Sigma, t_start_obs, t_geos, t_injection, rmin, rmax]
        
        params = predictor.init(jax.random.PRNGKey(1), t_start_obs, t_units, coords, Omega, t_start_obs, t_geos, t_injection)['params']
        tx = optax.adam(learning_rate=optax.polynomial_schedule(hparams['lr_init'], hparams['lr_final'], 1, hparams['num_iters']))

        if 'lr_inject' in hparams.keys():
            tx = optax.chain(
                optax.masked(optax.adam(learning_rate=hparams['lr_inject']), mask=flattened_traversal(lambda path, _: path[-1] == 't_injection')),
                optax.masked(tx, mask=flattened_traversal(lambda path, _: path[-1] != 't_injection')),
            )

        state = train_state.TrainState.create(apply_fn=predictor.apply, params=params.unfreeze(), tx=tx)  # TODO(pratul): this unfreeze feels sketchy

        # Restore saved checkpoint
        state = checkpoints.restore_checkpoint(checkpoint_dir, state)
        init_step = 1 + state.step

        # Replicate state for multiple gpus
        state = flax.jax_utils.replicate(state)
        return state, init_step

    def __call__(self, state, batchsize):
        total_loss = 0.0
        for i in range(self.num_losses):
            loss, state, images = self.grad_pmap[i](state, self.args[0].t_units, self.dtype[i], *self.args[i].sample(batchsize), *self.rendering_args, self.scale[i])
            total_loss += loss
        return total_loss, state, images
    
    def __add__(self, other):
        dtype = np.append(self.dtype, other.dtype)
        args = np.append(self.args, other.args)
        grad_pmap = np.append(self.grad_pmap, other.grad_pmap)
        scale = np.append(self.scale, other.scale)
        return TrainStep(dtype, args, grad_pmap, scale)
    
    @classmethod
    def image(cls, t_frames, image_plane, scale=1.0, dtype='full'):
        """
        Construct a training step for image plane measurements
        
        Parameters
        ----------
        t_frames: array, 
            Array of time for each image frame with astropy.units. If no units are included defaults to Hours.
        image_plane: np.array
            A movie array with image-plane frames (first axis should have same length as t_frames)
            For stokes concatenate movies along the second axis: np.stack([I, Q, U], axis=1)
        """
        args = TemporalBatchedArgs(t_frames, image_plane)
        grad_pmap = jax.pmap(bhnerf.network.gradient_step_image,
                             axis_name='batch', 
                             in_axes=(0, None, None, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None),
                             static_broadcasted_argnums=(1, 2))
        
        return cls(dtype, args, grad_pmap, scale)
        
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
        
        return cls(dtype, args, grad_pmap, scale)

        
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
        return self[batch]
    
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

