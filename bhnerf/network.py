import flax
import jax
from flax import linen as nn
import numpy as np
from jax import numpy as jnp
from typing import Any, Callable
import functools
import bhnerf
from jax import jit
import optax

from flax.training import train_state
from flax.training import checkpoints

safe_sin = lambda x: jnp.sin(x % (100 * jnp.pi))

class MLP(nn.Module):
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    do_skip: bool = True
  
    @nn.compact
    def __call__(self, x):
        """A simple Multi-Layer Preceptron (MLP) network

        Parameters
        ----------
        x: jnp.ndarray(float32), 
            [batch_size * n_samples, feature], points.
        net_depth: int, 
            the depth of the first part of MLP.
        net_width: int, 
            the width of the first part of MLP.
        activation: function, 
            the activation function used in the MLP.
        out_channel: 
            int, the number of alpha_channels.
        do_skip: boolean, 
            whether or not to use a skip connection

        Returns
        -------
        out: jnp.ndarray(float32), 
            [batch_size * n_samples, out_channel].
        """
        dense_layer = functools.partial(
            nn.Dense, kernel_init=jax.nn.initializers.he_uniform())

        if self.do_skip:
            skip_layer = self.net_depth // 2

        inputs = x
        for i in range(self.net_depth):
            x = dense_layer(self.net_width)(x)
            x = self.activation(x)
            if self.do_skip:
                if i % skip_layer == 0 and i > 0:
                    x = jnp.concatenate([x, inputs], axis=-1)
        out = dense_layer(self.out_channel)(x)

        return out

def posenc(x, deg):
    """
    Concatenate `x` with a positional encoding of `x` with degree `deg`.
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Parameters
    ----------
    x: jnp.ndarray, 
        variables to be encoded. Note that x should be in [-pi, pi].
    deg: int, 
        the degree of the encoding.

    Returns
    -------
    encoded: jnp.ndarray, 
        encoded variables.
    """
    if deg == 0:
        return x
    scales = jnp.array([2**i for i in range(deg)])
    xb = jnp.reshape((x[..., None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1])
    four_feat = safe_sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    return jnp.concatenate([x] + [four_feat], axis=-1)

def shard(xs):
    """Split data into shards for multiple devices along the first dimension."""
    return jax.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)

def flattened_traversal(fn):
    def mask(data):
        flat = flax.traverse_util.flatten_dict(data)
        return flax.traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})
    return mask

class NeRF_Predictor(nn.Module):
    """
    Full function to predict emission at a time step.
    
    Parameters
    ----------
    posenc_deg: int, default=3
    net_depth: int, default=4
    net_width: int, default=128
    activation: Callable[..., Any], default=nn.relu
    out_channel: int default=1
    do_skip: bool, default=True
    """
    posenc_deg: int = 3
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    do_skip: bool = True
    
    @nn.compact
    def __call__(self, t_frames, t_units, coords, Omega, t_start_obs, t_geos, t_injection):
        """
        Sample emission on given coordinates at specified times assuming a velocity model (Omega)
        
        Parameters
        ----------
        t_frames: array, 
            Array of time for each image frame
        t_units: astropy.units, 
            Time units of t_frames.
        coords: list of arrays, 
            For 3D emission coords=[x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
            alpha, beta are image coordinates. These arrays contain the ray integration points
        Omega: array, 
            Angular velocity array sampled along the coords points
        t_start_obs: astropy.Quantity, default=None
            Start time for observations, if None t_frames[0] is assumed to be start time.
        t_geos: array, 
            Time along each geodesic (ray). This is used to account for slow light (light travels at finite velocity).
        t_injection: float, 
            Time of hotspot injection in M units.
        
        Returns
        -------
        emission: jnp.array,
            An array with the emission points
        """
        emission_MLP = MLP(self.net_depth, self.net_width, self.activation, self.out_channel, self.do_skip)
        def predict_emission(t_frames, t_units, coords, Omega, t_start_obs, t_geos, t_injection):
            warped_coords = bhnerf.emission.velocity_warp_coords(
                coords, Omega, t_frames, t_start_obs, t_geos, t_injection, t_units=t_units, use_jax=True
            )
            
            # Zero emission prior to injection time
            valid_inputs_mask = jnp.isfinite(warped_coords)
            net_input = jnp.where(valid_inputs_mask, warped_coords, jnp.zeros_like(warped_coords))
            net_output = emission_MLP(posenc(net_input, self.posenc_deg))
            emission = nn.sigmoid(net_output[..., 0] - 10.0)
            emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
            
            return emission
        
        t_injection_param = self.param('t_injection', lambda key, values: jnp.array(values, dtype=jnp.float32), t_injection)
        emission = predict_emission(t_frames, t_units, coords, Omega, t_start_obs, t_geos, t_injection_param)
        return emission
    
def loss_fn_image(params, predictor_fn, target, t_frames, t_units, coords, Omega, 
                  g, dtau, Sigma, t_start_obs, t_geos, t_injection, rmin, rmax):
    """
    An L2 loss function for image pixels

    Parameters
    ----------
    params: dict, 
        A dictionary with network parameters (from state.params)
    predictor_fn: nn.Module
        A coordinate-based neural net for predicting the emission values as a continuous function
    target: array, 
        Target images to fit the model to. 
    t_frames: array, 
        Array of time for each image frame
    t_units: astropy.units, 
        Time units for t_frames.
    coords: list of arrays, 
        For 3D emission coords=[x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
        alpha, beta are image coordinates. These arrays contain the ray integration points
    Omega: array, 
        Angular velocity array sampled along the coords points
    g: array, 
        doppler boosting factor, 
    dtau: array, 
        mino time differential
    Sigma: array, 
    t_start_obs: astropy.Quantity, default=None
        Start time for observations, if None t_frames[0] is assumed to be start time.
    t_geos: array, 
        Time along each geodesic (ray). This is used to account for slow light (light travels at finite velocity).
    t_injection: float, 
        Time of hotspot injection in M units.
    rmin: float, 
        The minimum radius for recovery
    rmax: float, 
        The maximum radius for recovery
        
    Returns
    -------
    loss: jnp.array,
        An array with loss values (the size is the number of GPUs)
    images: jnp.array
        An array of predicted images at different times (t_frames)
    """
    emission = predictor_fn({'params': params}, t_frames, t_units, coords, Omega, t_start_obs, t_geos, t_injection)
    emission = bhnerf.emission.fill_unsupervised_emission(emission, coords, rmin, rmax, use_jax=True)
    images = bhnerf.emission.radiative_trasfer(emission, g, dtau, Sigma, use_jax=True)
    loss = jnp.mean(jnp.abs(images - target)**2)
    return loss, [images]

def loss_fn_eht(params, predictor_fn, target, t_frames, t_units, coords, Omega, 
                g, dtau, Sigma, t_start_obs, t_geos, t_injection, rmin, rmax, dtft_mats, vis_sigma):
    """
    An chi-square loss function for EHT observations

    Parameters
    ----------
    params: dict, 
        A dictionary with network parameters (from state.params)
    predictor_fn: nn.Module
        A coordinate-based neural net for predicting the emission values as a continuous function
    target: array, 
        Target images to fit the model to. 
    t_frames: array, 
        Array of time for each image frame
    t_units: astropy.units, 
        Time units for t_frames.
    coords: list of arrays, 
        For 3D emission coords=[x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
        alpha, beta are image coordinates. These arrays contain the ray integration points
    Omega: array, 
        Angular velocity array sampled along the coords points
    g: array, 
        doppler boosting factor, 
    dtau: array, 
        mino time differential
    Sigma: array, 
    t_start_obs: astropy.Quantity, default=None
        Start time for observations, if None t_frames[0] is assumed to be start time.
    t_geos: array, 
        Time along each geodesic (ray). This is used to account for slow light (light travels at finite velocity).
    t_injection: float, 
        Time of hotspot injection in M units.
    rmin: float, 
        The minimum radius for recovery
    rmax: float, 
        The maximum radius for recovery
    dtft_mats: array,
        An array of discrete time fourier transform matrices for each frame time
    vis_sigma: array,
        An array of standard deviations for each visibility measurement
    
    Returns
    -------
    loss: jnp.array,
        An array with loss values (the size is the number of GPUs)
    images: jnp.array
        An array of predicted images at different times (t_frames)
        
    Notes
    -----
    Currently only supports complex visibilities
    """
    emission = predictor_fn({'params': params}, t_frames, t_units, coords, Omega, t_start_obs, t_geos, t_injection)
    emission = bhnerf.emission.fill_unsupervised_emission(emission, coords, rmin, rmax, use_jax=True)
    images = bhnerf.emission.radiative_trasfer(emission, g, dtau, Sigma, use_jax=True)
    visibilities = jnp.stack([jnp.matmul(ft, image.ravel()) for ft, image in zip(dtft_mats, images)])
    loss = jnp.mean((jnp.abs(visibilities - target)/vis_sigma)**2)
    return loss, [images]

@functools.partial(jit, static_argnums=(3))
def train_step_image(state, target, t_frames, t_units, coords, Omega, g, dtau, Sigma,
                     t_start_obs, t_geos, t_injection, rmin, rmax):
    """
    Training step function for fitting the image-plane directly
    
    Parameters
    ----------
    state: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    target: array, 
        Target images to fit the model to. 
    t_frames: array, 
        Array of time for each image frame
    t_units: astropy.units, 
        Time units for t_frames.
    coords: list of arrays, 
        For 3D emission coords=[x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
        alpha, beta are image coordinates. These arrays contain the ray integration points
    Omega: array, 
        Angular velocity array sampled along the coords points
    g: array, 
        doppler boosting factor, 
    dtau: array, 
        mino time differential
    Sigma: array, 
    t_start_obs: astropy.Quantity, default=None
        Start time for observations, if None t_frames[0] is assumed to be start time.
    t_geos: array, 
        Time along each geodesic (ray). This is used to account for slow light (light travels at finite velocity).
    t_injection: float, 
        Time of hotspot injection in M units.
    rmin: float, 
        The minimum radius for recovery
    rmax: float, 
        The maximum radius for recovery
    dtft_mats: array,
        An array of discrete time fourier transform matrices for each frame time
    vis_sigma: array,
        An array of standard deviations for each visibility measurement
    
    Returns
    -------
    loss: jnp.array,
        An array with loss values (the size is the number of GPUs)
    images: jnp.array
        An array of predicted images at different times (t_frames)
        
    Notes
    -----
    Arguments for jax.pmap:
        axis_name='batch',
        in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None, None, None)
        static_broadcasted_argnums=(3),
    """
    (loss, [images]), grads = jax.value_and_grad(loss_fn_image, argnums=(0), has_aux=True)(
        state.params, state.apply_fn, target, t_frames, t_units, coords, Omega, g, dtau, Sigma, 
        t_start_obs, t_geos, t_injection, rmin, rmax)
    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    return loss, state, images

@functools.partial(jit, static_argnums=(3))
def test_step_image(state, target, t_frames, t_units, coords, Omega, g, dtau, Sigma, 
                    t_start_obs, t_geos, t_injection, rmin, rmax):
    """
    Test step function for fitting the image-plane directly. 
    This function is identical to train_step_image except does not compute gradients or 
    updates the state
    
    Parameters
    ----------
    state: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    target: array, 
        Target images to fit the model to. 
    t_frames: array, 
        Array of time for each image frame
    t_units: astropy.units, 
        Time units for t_frames.
    coords: list of arrays, 
        For 3D emission coords=[x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
        alpha, beta are image coordinates. These arrays contain the ray integration points
    Omega: array, 
        Angular velocity array sampled along the coords points
    g: array, 
        doppler boosting factor, 
    dtau: array, 
        mino time differential
    Sigma: array, 
    t_start_obs: astropy.Quantity, default=None
        Start time for observations, if None t_frames[0] is assumed to be start time.
    t_geos: array, 
        Time along each geodesic (ray). This is used to account for slow light (light travels at finite velocity).
    t_injection: float, 
        Time of hotspot injection in M units.
    rmin: float, 
        The minimum radius for recovery
    rmax: float, 
        The maximum radius for recovery
    dtft_mats: array,
        An array of discrete time fourier transform matrices for each frame time
    vis_sigma: array,
        An array of standard deviations for each visibility measurement
    
    Returns
    -------
    loss: jnp.array,
        An array with loss values (the size is the number of GPUs)
    images: jnp.array
        An array of predicted images at different times (t_frames)
        
    Notes
    -----
    Arguments for jax.pmap:
        axis_name='batch',
        in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None, None, None)
        static_broadcasted_argnums=(3),
    """
    loss, [images] = loss_fn_image(
        state.params, state.apply_fn, target, t_frames, t_units, coords, Omega, g, dtau, Sigma, 
        t_start_obs, t_geos, t_injection, rmin, rmax)
    return loss, images

@functools.partial(jit, static_argnums=(3))
def train_step_eht(state, target, t_frames, t_units, coords, Omega, g, dtau, Sigma, t_start_obs, 
                   t_geos, t_injection, rmin, rmax, dtft_mats, vis_sigma):
    """
    Train step function for fitting eht observations
    This function computed gradients and updates the state. 
    Currently only supports complex visibilities.
    
    Parameters
    ----------
    state: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    target: array, 
        Target eht observations to fit the model to. 
    t_frames: array, 
        Array of time for each image frame
    t_units: astropy.units, 
        Time units for t_frames.
    coords: list of arrays, 
        For 3D emission coords=[x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
        alpha, beta are image coordinates. These arrays contain the ray integration points
    Omega: array, 
        Angular velocity array sampled along the coords points
    g: array, 
        doppler boosting factor, 
    dtau: array, 
        mino time differential
    Sigma: array, 
    t_start_obs: astropy.Quantity, default=None
        Start time for observations, if None t_frames[0] is assumed to be start time.
    t_geos: array, 
        Time along each geodesic (ray). This is used to account for slow light (light travels at finite velocity).
    t_injection: float, 
        Time of hotspot injection in M units.
    rmin: float, 
        The minimum radius for recovery
    rmax: float, 
        The maximum radius for recovery
    dtft_mats: array,
        An array of discrete time fourier transform matrices for each frame time
    vis_sigma: array,
        An array of standard deviations for each visibility measurement
    
    Returns
    -------
    loss: jnp.array,
        An array with loss values (the size is the number of GPUs)
    images: jnp.array
        An array of predicted images at different times (t_frames)
        
    Notes
    -----
    Arguments for jax.pmap:
        axis_name='batch',
        in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, 0, 0), 
        static_broadcasted_argnums=(3),
    """
    (loss, [images]), grads = jax.value_and_grad(loss_fn_eht, argnums=(0), has_aux=True)(
        state.params, state.apply_fn, target, t_frames, t_units, coords, Omega, g, dtau, Sigma, 
        t_start_obs, t_geos, t_injection, rmin, rmax, dtft_mats, vis_sigma)
    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    return loss, state, images


@functools.partial(jit, static_argnums=(3))
def test_step_eht(state, target, t_frames, t_units, coords, Omega, g, dtau, Sigma, 
                  t_start_obs, t_geos, t_injection, rmin, rmax, dtft_mats, vis_sigma):
    """
    Test step function for fitting eht observations
    This function is identical to train_step_image except does not compute gradients or 
    updates the state. Currently only supports complex visibilities.
    
    Parameters
    ----------
    state: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    target: array, 
        Target eht observations to fit the model to. 
    t_frames: array, 
        Array of time for each image frame
    t_units: astropy.units, 
        Time units for t_frames.
    coords: list of arrays, 
        For 3D emission coords=[x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
        alpha, beta are image coordinates. These arrays contain the ray integration points
    Omega: array, 
        Angular velocity array sampled along the coords points
    g: array, 
        doppler boosting factor, 
    dtau: array, 
        mino time differential
    Sigma: array, 
    t_start_obs: astropy.Quantity, default=None
        Start time for observations, if None t_frames[0] is assumed to be start time.
    t_geos: array, 
        Time along each geodesic (ray). This is used to account for slow light (light travels at finite velocity).
    t_injection: float, 
        Time of hotspot injection in M units.
    rmin: float, 
        The minimum radius for recovery
    rmax: float, 
        The maximum radius for recovery
    dtft_mats: array,
        An array of discrete time fourier transform matrices for each frame time
    vis_sigma: array,
        An array of standard deviations for each visibility measurement
    
    Returns
    -------
    loss: jnp.array,
        An array with loss values (the size is the number of GPUs)
    images: jnp.array
        An array of predicted images at different times (t_frames)
        
    Notes
    -----
    Arguments for jax.pmap:
        axis_name='batch',
        in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, 0, 0), 
        static_broadcasted_argnums=(3),
    """
    loss, [images] = loss_fn_eht(state.params, state.apply_fn, target, t_frames, t_units, coords, Omega, g, dtau, Sigma, 
                                 t_start_obs, t_geos, t_injection, rmin, rmax, dtft_mats, vis_sigma)
    return loss, images
    
def run_optimization(runname, hparams, predictor, train_pstep, target, t_frames, geos, rmax, t_injection,
                     batched_args=[], args=[], emission_true=None, vis_res=64, log_period=100, save_period=1000):
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
    batched_args: list,
        Additional arguments taken by the training_step function which should be batched along the first dimension (e.g. Fourier matrices)
    args: list,
        Additional arguments taken by the training_step function
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
    t_units = t_frames.unit
    coords = jnp.array([geos.x, geos.y, geos.z])
    Omega = jnp.array(geos.Omega)
    g = jnp.array(geos.g.fillna(0.0))
    dtau = jnp.array(geos.dtau)
    Sigma = jnp.array(geos.Sigma)
    t_start_obs = t_frames[0]
    t_geos = jnp.array(geos.t)
    t_injection = t_injection
    rmin = geos.r.min().data
    rendering_args = [coords, Omega, g, dtau, Sigma, t_start_obs, t_geos, t_injection, rmin, rmax]
    
    # Grid for visualization (no interpolation is added for easy comparison)
    if emission_true is not None:
        vis_coords = np.array(np.meshgrid(np.linspace(emission_true.x[0], emission_true.x[-1], emission_true.shape[0]),
                                          np.linspace(emission_true.y[0], emission_true.y[-1], emission_true.shape[1]),
                                          np.linspace(emission_true.z[0], emission_true.z[-1], emission_true.shape[2]),
                                          indexing='ij'))
    else:
        grid_1d = np.linspace(-rmax, rmax, vis_res)
        vis_coords = np.array(np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij'))

    params = predictor.init(jax.random.PRNGKey(1), t_frames[0], t_units, coords, Omega, t_start_obs, t_geos, t_injection)['params']
    
    tx = optax.adam(learning_rate=optax.polynomial_schedule(hparams['lr_init'], hparams['lr_final'], 1, hparams['num_iters']))
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
            batch = np.random.choice(range(len(t_frames)), hparams['batchsize'], replace=False)
            bargs = [shard(arg[batch, ...]) for arg in batched_args]
            loss, state, images = train_pstep(state, shard(target[batch, ...]), shard(t_frames[batch, ...]), t_units, *rendering_args, *bargs, *args)

            # Log the current state on TensorBoard
            writer.add_scalar('log_loss/train', np.log10(np.mean(loss)), global_step=i)
            if (i == 1) or ((i % log_period) == 0) or (i ==  init_step + hparams['num_iters']):
                current_state = jax.device_get(flax.jax_utils.unreplicate(state))
                emission_grid = state.apply_fn({'params': current_state.params}, t_frames[0], t_units, vis_coords, 0.0, t_start_obs, 0.0, 0.0)
                emission_grid = bhnerf.emission.fill_unsupervised_emission(emission_grid, vis_coords, rmin, rmax)
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