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
    
def loss_fn_image(params, predictor_fn, target, t_frames, coords, Omega, J,
                  g, dtau, Sigma, t_start_obs, t_geos, t_injection, rmin, rmax, t_units):
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
    rmin: float, 
        The minimum radius for recovery
    rmax: float, 
        The maximum radius for recovery
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
    emission = bhnerf.emission.fill_unsupervised_emission(emission, coords, rmin, rmax, use_jax=True)
    if not jnp.isscalar(J):
        J = bhnerf.utils.expand_dims(J, emission.ndim+1, 0, use_jax=True)
        emission = bhnerf.utils.expand_dims(emission, emission.ndim+1, 1, use_jax=True)
    images = kgeo.radiative_trasfer(emission*J, g, dtau, Sigma, use_jax=True)
    loss = jnp.mean(jnp.abs(images - target)**2)
    return loss, [images]

def loss_fn_eht(params, predictor_fn, target, sigma, A, t_frames, coords, Omega, J,
                g, dtau, Sigma, t_start_obs, t_geos, t_injection, rmin, rmax, t_units, dtype):
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
    rmin: float, 
        The minimum radius for recovery
    rmax: float, 
        The maximum radius for recovery
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
        
    Notes
    -----
    Currently only supports complex visibilities
    """
    emission = predictor_fn({'params': params}, t_frames, t_units, coords, Omega, t_start_obs, t_geos, t_injection)
    emission = J * bhnerf.emission.fill_unsupervised_emission(emission, coords, rmin, rmax, use_jax=True)
    images = kgeo.radiative_trasfer(emission, g, dtau, Sigma, use_jax=True)
    
    # Reshape images to match A operations
    image_vectors = images.reshape(images.shape[0], -1, 1)
    image_vectors = bhnerf.utils.expand_dims(image_vectors, A.ndim, axis=1, use_jax=True)
    visibilities = jnp.squeeze(jnp.matmul(A, image_vectors), -1)
    
    if dtype == 'vis':
        if A.ndim != 3: raise AttributeError('A matrix should have 3 dimensions for cphase dtype')
        chisq = jnp.mean((jnp.abs(visibilities - target)/sigma)**2)
    elif dtype == 'cphase':
        if A.ndim != 4: raise AttributeError('A matrix should have 4 dimensions for cphase dtype')
        clphase_samples = jnp.angle(jnp.product(visibilities, axis=1))
        chisq = jnp.mean((1.0 - jnp.cos(target-clphase_samples))/(sigma**2))
        
    return chisq, [images]

@functools.partial(jit, static_argnums=(1))
def train_step_image(state, t_units, target, t_frames, coords, Omega, J, g, dtau, Sigma,
                     t_start_obs, t_geos, t_injection, rmin, rmax):
    """
    Training step function for fitting the image-plane directly
    
    Parameters
    ----------
    state: flax.training.train_state.TrainState, 
        The training state holding the network parameters and apply_fn
    t_units: astropy.units, 
        Time units for t_frames.
    target: array, 
        Target images to fit the model to. 
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
        
    Notes
    -----
    Arguments for jax.pmap:
        axis_name='batch',
        in_axes=(0, None, 0, 0, None, None, None, None, None, None, None, None, None, None, None),
        static_broadcasted_argnums=(1),
    """
    (loss, [images]), grads = jax.value_and_grad(loss_fn_image, argnums=(0), has_aux=True)(
        state.params, state.apply_fn, target, t_frames, coords, Omega, J, g, dtau, Sigma, 
        t_start_obs, t_geos, t_injection, rmin, rmax, t_units)
    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    return loss, state, images

@functools.partial(jit, static_argnums=(1))
def test_step_image(state, t_units, target, t_frames, coords, Omega, J, g, dtau, Sigma, 
                    t_start_obs, t_geos, t_injection, rmin, rmax):
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
    target: array, 
        Target images to fit the model to. 
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
        
    Notes
    -----
    Arguments for jax.pmap:
        axis_name='batch',
        in_axes=(0, None, None, 0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None), 
        static_broadcasted_argnums=(1),
    """
    loss, [images] = loss_fn_image(
        state.params, state.apply_fn, target, t_frames, coords, Omega, J, g, dtau, Sigma, 
        t_start_obs, t_geos, t_injection, rmin, rmax, t_units)
    return loss, images

@functools.partial(jit, static_argnums=(1, 2))
def train_step_eht(state, t_units, dtype, target, sigma, A, t_frames, coords, Omega, J, g, dtau, Sigma, t_start_obs, 
                   t_geos, t_injection, rmin, rmax):
    """
    Train step function for fitting eht observations
    This function computed gradients and updates the state. 
    Currently only supports complex visibilities.
    
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
        
    Notes
    -----
    Arguments for jax.pmap:
        axis_name='batch',
        in_axes=(0, None, None, 0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None), 
        static_broadcasted_argnums=(1, 2))
    """
    (loss, [images]), grads = jax.value_and_grad(loss_fn_eht, argnums=(0), has_aux=True)(
        state.params, state.apply_fn, target, sigma, A, t_frames, coords, Omega, J, g, dtau, Sigma, 
        t_start_obs, t_geos, t_injection, rmin, rmax, t_units, dtype)
    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    return loss, state, images

@functools.partial(jit, static_argnums=(1, 2))
def test_step_eht(state, t_units, dtype, target, sigma, A, t_frames, coords, Omega, J, g, dtau, Sigma, 
                  t_start_obs, t_geos, t_injection, rmin, rmax):
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
        
    Notes
    -----
    Arguments for jax.pmap:
        axis_name='batch',
        in_axes=(0, None, None, 0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None), 
        static_broadcasted_argnums=(1, 2))
    """
    loss, [images] = loss_fn_eht(state.params, state.apply_fn, target, sigma, A, t_frames, coords, Omega, J, g, dtau, Sigma, 
                                 t_start_obs, t_geos, t_injection, rmin, rmax, t_units, dtype)
    return loss, images
    
def sample_3d_grid(state, t_frames, tstart, rmin=0.0, rmax=np.inf, Omega=0.0, fov=None, coords=None, resolution=64): 
    """
    Parameters
    ----------
    t_frames: array, 
        Array of time for each image frame
    t_start: astropy.Quantity, default=None
        Start time for observations.
    rmin: float, default=0
        Zero values at radii < rmin
    rmax: float, default=np.inf
        Zero values at radii > rmax,
    Omega: array or float, default=0.0 
        Angular velocity array sampled along the coords. If initial time is sampled Omega has no effect and could be 0.0.
    fov: float, default=None
        Field of view. If None then coords need to be provided.
    coords: array(shape=(3,npoints)), optional, 
        Array of grid coordinates (x, y, z). If not specified, fov and resolution are used to grid the domain.
    resolution: int, default=64
        Grid resolution along [x,y,z].
    """
    try:
        state = jax.device_get(flax.jax_utils.unreplicate(state))
    except IndexError:
        state = jax.device_get(state)
    
    if (coords is None) and (fov is not None):
        grid_1d = np.linspace(-fov/2, fov/2, resolution)
        coords = np.array(np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij'))
    elif (coords is None):
        raise AttributeError('Either coords or fov+resolution must be provided')
        
    if not all(np.atleast_1d(t_frames == tstart)) and (Omega==None):
        raise AttributeError('If t_frames != tstart, an angular velocity Omega!=0.0 needs to be provided')
        
    # Get the a grid values sampled from the neural network
    emission = state.apply_fn({'params': state.params}, t_frames, tstart.unit, coords, Omega, tstart, 0.0, 0.0)
    emission =  bhnerf.emission.fill_unsupervised_emission(emission, coords, rmin, rmax)
    
    return emission
    