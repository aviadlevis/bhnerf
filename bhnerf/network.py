import flax
import jax
from flax import linen as nn
import numpy as np
from jax import numpy as jnp
from typing import Any, Callable
import functools
import bhnerf
from bhnerf import kgeo
from jax import jit
from flax.training import train_state, checkpoints
import optax
from pathlib import Path
import yaml

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

def integrated_posenc(x, x_cov, max_deg, min_deg=0):
    """
    Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].

    Parameters
    ----------
    x: jnp.ndarray, variables to be encoded. Should
      be in [-pi, pi]. 
    x_cov: jnp.ndarray, covariance matrices for `x`.
    max_deg: int, the max degree of the encoding.
    min_deg: int, the min degree of the encoding. default=0.

    Returns
    -------
    encoded: jnp.ndarray, encoded variables.
    """
    if jnp.isscalar(x_cov):
        x_cov = jnp.full_like(x, x_cov)
    scales = 2**jnp.arange(min_deg, max_deg)
    shape = list(x.shape[:-1]) + [-1]
    y = jnp.reshape(x[..., None, :] * scales[:, None], shape)
    y_var = jnp.reshape(x_cov[..., None, :] * scales[:, None]**2, shape)

    return expected_sin(
      jnp.concatenate([y, y + 0.5 * jnp.pi], axis=-1),
      jnp.concatenate([y_var] * 2, axis=-1))


def expected_sin(x, x_var):
    # When the variance is wide, shrink sin towards zero.
    y = jnp.exp(-0.5 * x_var) * jnp.sin(x)
    return y

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

class NeRF_Predictor(nn.Module):
    """
    Full function to predict emission at a time step.
    
    Parameters
    ----------
    scale: float, default=1.0       
        scale of the domain; scales the NN inputs
    rmin: float, default=0.0        
        minimum radius for recovery
    rmax: float, default=np.inf     
        maximum radius for recovery
    z_width: float, default=np.inf  
        maximum width of the disk (M units)
    posenc_deg: int, default=3
    posenc_var: float, default=2e-5 
        Corresponds to variance of uniform distribution variance with voxel width of ~1/64.
    net_depth: int, default=4
    net_width: int, default=128
    activation: Callable[..., Any], default=nn.relu
    out_channel: int default=1
    do_skip: bool, default=True
    """
    scale: float = 1.0
    rmin: float = 0.0
    rmax: float = np.inf
    z_width: float = np.inf 
    posenc_deg: int = 3
    posenc_var: float = 2e-5
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    do_skip: bool = True
    
    def init_params(self, raytracing_args, seed=1):
        raytracing_args = np.atleast_1d(raytracing_args)[0]
        params = self.init(jax.random.PRNGKey(seed), 
                           raytracing_args['t_start_obs'], 
                           raytracing_args['t_start_obs'].unit, 
                           raytracing_args['coords'], 
                           raytracing_args['Omega'], 
                           raytracing_args['t_start_obs'], 
                           raytracing_args['t_geos'], 
                           raytracing_args['t_injection'])['params']
        return params.unfreeze() # TODO(pratul): this unfreeze feels sketchy

    def init_state(self, params, num_iters=5000, lr_init=1e-4, lr_final=1e-6, lr_inject=None, checkpoint_dir=''):
        
        lr = optax.polynomial_schedule(lr_init, lr_final, 1, num_iters)
        tx = optax.adam(learning_rate=lr)
        
        if lr_inject:
            tx = optax.chain(
                optax.masked(optax.adam(learning_rate=lr_inject), mask=flattened_traversal(lambda path, _: path[-1] == 't_injection')),
                optax.masked(tx, mask=flattened_traversal(lambda path, _: path[-1] != 't_injection')),
            )
            
        state = train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)  
               
        # Restore saved checkpoint
        state = checkpoints.restore_checkpoint(checkpoint_dir, state)

        # Replicate state for multiple gpus
        state = flax.jax_utils.replicate(state)
        return state
    
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
            # net_output = emission_MLP(integrated_posenc(net_input / self.scale, self.posenc_var, self.posenc_deg))
            net_output = emission_MLP(posenc(net_input / self.scale, self.posenc_deg))
            emission = nn.sigmoid(net_output[..., 0] - 10.0)
            emission = bhnerf.emission.fill_unsupervised_emission(emission, coords, self.rmin, self.rmax, self.z_width, use_jax=True)
            emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
            return emission
        
        t_injection_param = self.param('t_injection', lambda key, values: jnp.array(values, dtype=jnp.float32), t_injection)
        emission = predict_emission(t_frames, t_units, coords, Omega, t_start_obs, t_geos, t_injection_param)
        return emission
    
    def save_params(self, directory, filename='NeRF_Predictor_params.yml'):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        params = self.__dict__.copy()
        params.pop('_state', None)
        params.pop('activation', None)
        with open(directory.joinpath(filename), 'w') as yaml_file:
            yaml.dump(params, yaml_file)
        
    @classmethod
    def from_yml(cls, directory, filename='NeRF_Predictor_params.yml'):
        params = yaml.safe_load(Path(directory).joinpath(filename).read_text())
        return cls(**params)

class GRID_Predictor(nn.Module):
    """
    Full function to predict emission at a time step.
    
    Parameters
    ----------
    scale: float, default=1.0       
        scale of the domain; scales the NN inputs
    rmin: float, default=0.0        
        minimum radius for recovery
    rmax: float, default=np.inf     
        maximum radius for recovery
    z_width: float, default=np.inf  
        maximum width of the disk (M units)
    grid_res: int, default=64
    """
    scale: float = 1.0
    rmin: float = 0.0
    rmax: float = np.inf
    z_width: float = np.inf 
    grid_res: int = 64 
    
    def init_params(self, raytracing_args, seed=1):
        params = self.init(jax.random.PRNGKey(seed), 
                           raytracing_args['t_start_obs'], 
                           raytracing_args['t_start_obs'].unit, 
                           raytracing_args['coords'], 
                           raytracing_args['Omega'], 
                           raytracing_args['t_start_obs'], 
                           raytracing_args['t_geos'], 
                           raytracing_args['t_injection'])['params']
        return params.unfreeze() # TODO(pratul): this unfreeze feels sketchy

    def init_state(self, params, num_iters=5000, lr_init=1e-4, lr_final=1e-6, lr_inject=None, checkpoint_dir=''):
        
        lr = optax.polynomial_schedule(lr_init, lr_final, 1, num_iters)
        tx = optax.adam(learning_rate=lr)
        
        if lr_inject:
            tx = optax.chain(
                optax.masked(optax.adam(learning_rate=lr_inject), mask=flattened_traversal(lambda path, _: path[-1] == 't_injection')),
                optax.masked(tx, mask=flattened_traversal(lambda path, _: path[-1] != 't_injection')),
            )
            
        state = train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)  
               
        # Restore saved checkpoint
        state = checkpoints.restore_checkpoint(checkpoint_dir, state)

        # Replicate state for multiple gpus
        state = flax.jax_utils.replicate(state)
        return state
    
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
        grid_init = lambda rng, shape: jnp.ones(shape) * -10.
        grid = self.param('grid', grid_init, (self.grid_res, self.grid_res, self.grid_res)) 
        def predict_emission(t_frames, t_units, coords, Omega, t_start_obs, t_geos, t_injection):
            warped_coords = bhnerf.emission.velocity_warp_coords(
                coords, Omega, t_frames, t_start_obs, t_geos, t_injection, t_units=t_units, use_jax=True
            )
            
            # Zero emission prior to injection time
            valid_inputs_mask = jnp.isfinite(warped_coords)
            net_input = jnp.where(valid_inputs_mask, warped_coords, jnp.zeros_like(warped_coords))
            net_input = jnp.moveaxis(net_input, -1, 0)
            net_input = (net_input + self.scale) / (2*self.scale) * (self.grid_res - 1.)
            net_output = jax.scipy.ndimage.map_coordinates(grid, net_input, order=1, cval=0.)
            emission = nn.sigmoid(net_output - 10.)
            emission = bhnerf.emission.fill_unsupervised_emission(emission, coords, self.rmin, self.rmax, self.z_width, use_jax=True)
            emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
            return emission
        
        t_injection_param = self.param('t_injection', lambda key, values: jnp.array(values, dtype=jnp.float32), t_injection)
        emission = predict_emission(t_frames, t_units, coords, Omega, t_start_obs, t_geos, t_injection_param)
        return emission
    
    def save_params(self, directory, filename='GRID_Predictor_params.yml'):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        params = self.__dict__.copy()
        params.pop('_state', None)
        params.pop('activation', None)
        with open(directory.joinpath(filename), 'w') as yaml_file:
            yaml.dump(params, yaml_file)
        
    @classmethod
    def from_yml(cls, directory, filename='GRID_Predictor_params.yml'):
        params = yaml.safe_load(Path(directory).joinpath(filename).read_text())
        return cls(**params)

    
def image_plane_prediction(params, predictor_fn, t_frames, coords, Omega, J,
                           g, dtau, Sigma, t_start_obs, t_geos, t_injection, t_units):
    """
    Predict image pixels from NeRF emission values

    Parameters
    ----------
    params: dict, 
        A dictionary with network parameters (from state.params)
    predictor_fn: nn.Module
        A coordinate-based neural net for predicting the emission values as a continuous function
    t_frames: array, 
        Array of time for each image frame
    coords: list of arrays, 
        For 3D emission coords=[x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
        alpha, beta are image coordinates. These arrays contain the ray integration points
    Omega: array, 
        Angular velocity array sampled along the coords points
    J: np.array(shape=(3,...)), 
        Stokes vector scaling factors including parallel transport (I, Q, U)
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
    t_units: astropy.units, 
        Time units for t_frames.
        
    Returns
    -------
    loss: jnp.array,
        An array with loss values (the size is the number of GPUs)
    images: jnp.array
        An array of predicted images at different times (t_frames)
    """
    emission = predictor_fn({'params': params}, t_frames, t_units, coords, Omega, t_start_obs, t_geos, t_injection)
    if not jnp.isscalar(J):
        J = bhnerf.utils.expand_dims(J, emission.ndim+1, 0, use_jax=True)
        emission = J * bhnerf.utils.expand_dims(emission, emission.ndim+1, 1, use_jax=True)
        emission = jnp.squeeze(emission)
    images = kgeo.radiative_trasfer(emission, g, dtau, Sigma, use_jax=True)
    return images

def loss_fn_image(params, predictor_fn, target, sigma, offset, t_frames, coords, Omega, J, g, dtau, 
                  Sigma, t_start_obs, t_geos, t_injection, scale, t_units, dtype):
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
    sigma: array,
        An array of standard deviations for each pixel
    offset: array,
        A bias or offset for each pixel
    t_frames: array, 
        Array of time for each image frame
    coords: list of arrays, 
        For 3D emission coords=[x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
        alpha, beta are image coordinates. These arrays contain the ray integration points
    Omega: array, 
        Angular velocity array sampled along the coords points
    J: np.array(shape=(3,...)), 
        Stokes vector scaling factors including parallel transport (I, Q, U)
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
    scale: float, 
        Scaling factor for the loss
    t_units: astropy.units, 
        Time units for t_frames.
    dtype: 'str',
        Datatype to compute the loss for ('full'/'lightcurve' etc..)
        
    Returns
    -------
    loss: jnp.array,
        An array with loss values (the size is the number of GPUs)
    images: jnp.array
        An array of predicted images at different times (t_frames)
    """
    images = image_plane_prediction(
        params, predictor_fn, t_frames, coords, Omega, J, g, dtau, Sigma, t_start_obs, t_geos, t_injection, t_units
    )
    if dtype == 'full':
        loss = jnp.sum(jnp.abs((images - target - offset)/sigma)**2)
    elif dtype == 'lc':
        lightcurve = images.sum(axis=(-1,-2))
        loss = jnp.sum(jnp.abs((lightcurve - target - offset)/sigma)**2)
    else:
        raise AttributeError('image dtype ({}) not supported'.format(dtype))
        
    return scale*loss, [images]

def loss_fn_eht(params, predictor_fn, target, sigma, A, t_frames, coords, Omega, J,
                g, dtau, Sigma, t_start_obs, t_geos, t_injection, scale, t_units, dtype):
    """
    An chi-square loss function for EHT observations

    Parameters
    ----------
    params: dict, 
        A dictionary with network parameters (from state.params)
    predictor_fn: nn.Module
        A coordinate-based neural net for predicting the emission values as a continuous function
    target: array, 
        Target measurement values to fit the model to. 
    A: array,
        An array of discrete time fourier transform matrices for each frame time
    sigma: array,
        An array of standard deviations for each measurement
    t_frames: array, 
        Array of time for each image frame
    coords: list of arrays, 
        For 3D emission coords=[x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
        alpha, beta are image coordinates. These arrays contain the ray integration points
    Omega: array, 
        Angular velocity array sampled along the coords points
    J: np.array(shape=(3,...)), 
        Stokes vector scaling factors including parallel transport (I, Q, U)
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
    scale: float, 
        Scaling factor for the loss
    t_units: astropy.units, 
        Time units for t_frames.
    dtype: 'str',
        Datatype to compute the loss for ('vis'/'cphase' etc..)
        
    Returns
    -------
    loss: jnp.array,
        An array with loss values (the size is the number of GPUs)
    images: jnp.array
        An array of predicted images at different times (t_frames)
    """
    images = image_plane_prediction(
        params, predictor_fn, t_frames, coords, Omega, J, g, dtau, Sigma, t_start_obs, t_geos, t_injection, t_units
    )
    
    # Reshape images to match A operations
    image_vectors = images.reshape(*images.shape[:-2], -1, 1)
    image_vectors = bhnerf.utils.expand_dims(image_vectors, A.ndim, axis=-3, use_jax=True)
    visibilities = jnp.squeeze(jnp.matmul(A, image_vectors), -1)
    if dtype == 'vis':
        if visibilities.ndim != target.ndim: 
            raise AttributeError('visibilities (ndim={}) should have same dimensions as target (ndim={}) for dtype={}'.format(visibilities.ndim, target.ndim, dtype))
        chisq = jnp.sum((jnp.abs(visibilities - target)/sigma)**2)
        
    elif dtype == 'amp':
        if visibilities.ndim != target.ndim: 
            raise AttributeError('visibilities (ndim={}) should have same dimensions as target (ndim={}) for dtype={}'.format(visibilities.ndim, target.ndim, dtype))
        chisq = jnp.sum(jnp.abs((jnp.abs(visibilities) - target)/sigma)**2)
        
    elif dtype == 'cphase':
        if visibilities.ndim != target.ndim+1: 
            raise AttributeError('visibilities (ndim={}) should have +1 dimensions as target (ndim={}) for dtype={}'.format(visibilities.ndim, target.ndim, dtype))
        clphase_samples = jnp.angle(jnp.product(visibilities, axis=-2))
        chisq = jnp.sum((1.0 - jnp.cos(target-clphase_samples))/(sigma**2))
        
    else: 
        raise AttributeError('eht dtype ({}) not supported'.format(dtype))
        
    return scale*chisq, [images]

@functools.partial(jit, static_argnums=(1, 2))
def gradient_step_image(state, t_units, dtype, target, sigma, offset, t_frames, coords, Omega, J, g, dtau, Sigma,
                        t_start_obs, t_geos, t_injection, scale):
    """
    Gradient step function for fitting the image-plane directly
    This function computed gradients and updates the state. 
    
    Parameters
    ----------
    state: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    t_units: astropy.units, 
        Time units for t_frames.
    dtype: 'str',
        Datatype to compute the loss for ('full'/'lc' etc..)
    target: array, 
        Target images to fit the model to. 
    sigma: array,
        An array of standard deviations for each pixel
    offset: array,
        A bias or offset for each pixel
    t_frames: array, 
        Array of time for each image frame
    coords: list of arrays, 
        For 3D emission coords=[x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
        alpha, beta are image coordinates. These arrays contain the ray integration points
    Omega: array, 
        Angular velocity array sampled along the coords points
    J: np.array(shape=(3,...)), 
        Stokes vector scaling factors including parallel transport (I, Q, U)
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
    scale: float, 
        Scaling factor for the loss
        
    Returns
    -------
    loss: jnp.array,
        An array with loss values (the size is the number of GPUs)
    images: jnp.array
        An array of predicted images at different times (t_frames)
    """
    (loss, [images]), grads = jax.value_and_grad(loss_fn_image, argnums=(0), has_aux=True)(
        state.params, state.apply_fn, target, sigma, offset, t_frames, coords, Omega, J, g, dtau, Sigma, 
        t_start_obs, t_geos, t_injection, scale, t_units, dtype)
    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    return loss, state, images

@functools.partial(jit, static_argnums=(1, 2))
def gradient_step_eht(state, t_units, dtype, target, sigma, A, t_frames, coords, Omega, J, g, dtau, 
                      Sigma, t_start_obs, t_geos, t_injection, scale):
    """
    Gradient step function for fitting eht observations
    This function computed gradients and updates the state. 
    
    Parameters
    ----------
    state: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    t_units: astropy.units, 
        Time units for t_frames.
    dtype: 'str',
        Datatype to compute the loss for ('vis'/'cphase' etc..)
    target: array, 
        Target measurement values to fit the model to. 
    A: array,
        An array of discrete time fourier transform matrices for each frame time
    sigma: array,
        An array of standard deviations for each measurement
    t_frames: array, 
        Array of time for each image frame
    dtype: 'str', default='vis'
        Datatype to compute the loss for ('vis'/'cphase' etc..)
    coords: list of arrays, 
        For 3D emission coords=[x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
        alpha, beta are image coordinates. These arrays contain the ray integration points
    Omega: array, 
        Angular velocity array sampled along the coords points
    J: np.array(shape=(3,...)), 
        Stokes vector scaling factors including parallel transport (I, Q, U)
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
    scale: float, 
        Scaling factor for the loss
        
    Returns
    -------
    loss: jnp.array,
        An array with loss values (the size is the number of GPUs)
    images: jnp.array
        An array of predicted images at different times (t_frames)
    """
    (loss, [images]), grads = jax.value_and_grad(loss_fn_eht, argnums=(0), has_aux=True)(
        state.params, state.apply_fn, target, sigma, A, t_frames, coords, Omega, J, g, dtau, Sigma, 
        t_start_obs, t_geos, t_injection, scale, t_units, dtype)
    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    return loss, state, images

@functools.partial(jit, static_argnums=(1, 2))
def test_image(state, t_units, dtype, target, sigma, offset, t_frames, coords, Omega, J, g, dtau, Sigma, 
                    t_start_obs, t_geos, t_injection, scale):
    """
    Test step function for fitting the image-plane directly. 
    This function is identical to train_step_image except does not compute gradients or 
    updates the state
    
    Parameters
    ----------
    state: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    t_units: astropy.units, 
        Time units for t_frames.
    dtype: 'str',
        Datatype to compute the loss for ('full'/'lc' etc..)
    target: array, 
        Target images to fit the model to. 
    sigma: array,
        An array of standard deviations for each pixel
    offset: array,
        A bias or offset for each pixel
    t_frames: array, 
        Array of time for each image frame
    coords: list of arrays, 
        For 3D emission coords=[x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
        alpha, beta are image coordinates. These arrays contain the ray integration points
    Omega: array, 
        Angular velocity array sampled along the coords points
    J: np.array(shape=(3,...)), 
        Stokes vector scaling factors including parallel transport (I, Q, U)
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
    scale: float, 
        Scaling factor for the loss
        
    Returns
    -------
    loss: jnp.array,
        An array with loss values (the size is the number of GPUs)
    images: jnp.array
        An array of predicted images at different times (t_frames)
    """
    loss, [images] = loss_fn_image(
        state.params, state.apply_fn, target, sigma, offset, t_frames, coords, Omega, J, g, dtau, Sigma, 
        t_start_obs, t_geos, t_injection, scale, t_units, dtype)      
    return loss, state, images

@functools.partial(jit, static_argnums=(1, 2))
def test_eht(state, t_units, dtype, target, sigma, A, t_frames, coords, Omega, J, g, dtau, Sigma, 
             t_start_obs, t_geos, t_injection, scale):
    """
    Test step function for fitting eht observations
    This function is identical to train_step_image except does not compute gradients or 
    updates the state. Currently only supports complex visibilities.
    
    Parameters
    ----------
    state: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    t_units: astropy.units, 
        Time units for t_frames.
    dtype: 'str',
        Datatype to compute the loss for ('vis'/'cphase' etc..)
    target: array, 
        Target measurement values to fit the model to. 
    A: array,
        An array of discrete time fourier transform matrices for each frame time
    sigma: array,
        An array of standard deviations for each measurement
    t_frames: array, 
        Array of time for each image frame
    coords: list of arrays, 
        For 3D emission coords=[x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
        alpha, beta are image coordinates. These arrays contain the ray integration points
    Omega: array, 
        Angular velocity array sampled along the coords points
    J: np.array(shape=(3,...)), 
        Stokes vector scaling factors including parallel transport (I, Q, U)
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
    scale: float, 
        Scaling factor for the loss
        
    Returns
    -------
    loss: jnp.array,
        An array with loss values (the size is the number of GPUs)
    images: jnp.array
        An array of predicted images at different times (t_frames)
    """
    loss, [images] = loss_fn_eht(state.params, state.apply_fn, target, sigma, A, t_frames, coords, Omega, J, g, dtau, Sigma, 
                                 t_start_obs, t_geos, t_injection, scale, t_units, dtype)
    return loss, state, images
       
def sample_3d_grid(apply_fn, params, fov=None, coords=None, resolution=64, chunk=-1): 
    """
    Parameters
    ----------
    apply_fn: nn.Module
        A coordinate-based neural net for predicting the emission values as a continuous function
    params: dict, 
        A dictionary with network parameters (from state.params)
    fov: float, default=None
        Field of view. If None then coords need to be provided.
    coords: array(shape=(3,npoints)), optional, 
        Array of grid coordinates (x, y, z). If not specified, fov and resolution are used to grid the domain.
    resolution: int, default=64
        Grid resolution along [x,y,z].
    """   
    try:
        params = jax.device_get(flax.jax_utils.unreplicate(params))
    except IndexError:
        params = jax.device_get(params)
    
    if (coords is None) and (fov is not None):
        grid_1d = np.linspace(-fov/2, fov/2, resolution)
        coords = np.array(np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij'))
    elif (coords is None):
        raise AttributeError('Either coords or fov+resolution must be provided')

    # Get the a grid values sampled from the neural network
    resolution = coords.shape[-1]
    chunk = resolution if chunk < 0 else chunk
    
    emission = []
    for c in range(resolution//chunk):
        coords_chunk = coords[:, c * chunk : (c + 1) * chunk, :, :]
        emission.append(apply_fn({'params': params}, 0.0, None, coords_chunk, 0.0, 0.0, 0.0, 0.0))
    emission = np.concatenate(emission, axis=0)
    return emission

def raytracing_args(geos, Omega, t_injection, t_start_obs, J=1.0):
    """
    Return a list (ordered) with the ray tracing (non-optimized) arguments . 
    
    Parameters
    ----------
    geos: xr.Dataset
        A dataset specifying geodesics (ray trajectories) ending at the image plane.
    Omega: xr.DataArray
        A dataarray specifying the keplerian velocity field
    t_injection: float, 
        Time of hotspot injection in M units.
    t_start_obs: astropy.Quantity, 
        Start time for observations
    J: np.array(shape=(3,...)), default=1.0
        Stokes vector scaling factors including parallel transport (I, Q, U). J=1.0 gives non-polarized emission.
        
    Returns
    -------
    raytracing_args: list.
        List of ray-tracing arguments (non-optimized)
    """
    from collections import OrderedDict
    
    coords = jnp.array([geos.x, geos.y, geos.z])
    umu = kgeo.azimuthal_velocity_vector(geos, Omega)
    g = jnp.array(kgeo.doppler_factor(geos, umu))
    Omega = jnp.array(Omega)
    dtau = jnp.array(geos.dtau)
    Sigma = jnp.array(geos.Sigma)
    t_geos = jnp.array(geos.t)

    raytracing_args = OrderedDict({
        'coords': coords, 
        'Omega': Omega, 
        'J': J, 
        'g': g, 
        'dtau': dtau, 
        'Sigma': Sigma, 
        't_start_obs': t_start_obs, 
        't_geos': t_geos, 
        't_injection': t_injection
    })

    return raytracing_args

def tv_reg(apply_fn, params, coords):
    """
    Calculates a proxy for TV regularization which involves evaluating
      
    Parameters
    ----------
    apply_fn: nn.Module
        A coordinate-based neural net for predicting the emission values as a continuous function
    params: dict, 
        A dictionary with network parameters (from state.params)
    coords: list of arrays, 
        For 3D emission coords=[x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
        alpha, beta are image coordinates. These arrays contain the ray integration points
          
    Returns
    ------- 
    reg_term: the value of the TV regularization cost of the MLP evaluated at points coords
    """
    def predict_eta(coords):
        return apply_fn({'params': params}, 0.0, None, coords, 0.0, 0.0, 0.0, 0.0)
    
    pred_eta_grad_fn = jax.vmap(jax.value_and_grad(predict_eta), in_axes=(0,))
    _, eta_grad = pred_eta_grad_fn(coords)
    reg_term = jnp.sum(jnp.absolute(eta_grad)) * lam
    
    return reg_term

def flattened_traversal(fn):
    def mask(data):
        flat = flax.traverse_util.flatten_dict(data)
        return flax.traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})
    return mask
