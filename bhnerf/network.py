import flax
import jax
from flax import linen as nn
import numpy as np
from jax import numpy as jnp
from typing import Any, Callable
import functools
import bhnerf

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
    
class NeRF_RotationAxis(nn.Module):
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
    def __call__(self, coordinates, velocity, tstart, tstop, axis_init):
        """
        Parameters
        ----------
        coordinates: list of arrays, 
            1. For 3D emission coords=[t, x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
               alpha, beta are image coordinates. These arrays contain the ray integration points
            2. For 2D emission coords=[t, x, y] with each array shape=(nt, num_alpha, ngeo)
        """
        emission_MLP = MLP(self.net_depth, self.net_width, self.activation, self.out_channel, self.do_skip)
        
        def predict_emission(coordinates, velocity, axis, tstart, tstop):
            net_input = bhnerf.emission.velocity_warp(
                coordinates[1:], coordinates[0], velocity, axis, tstart, tstop, use_jax=True)
            valid_inputs_mask = jnp.isfinite(net_input)
            net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
            net_output = emission_MLP(posenc(net_input, self.posenc_deg))
            emission = nn.sigmoid(net_output[..., 0] - 10.)
            emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
            return emission
        
        axis = self.param('axis', lambda key, values: jnp.array(values, dtype=jnp.float32), axis_init)
        emission = predict_emission(coordinates, velocity, axis, tstart, tstop)

        return emission

class EmissionOperator(object):
    def __init__(self, predictor, kwargs):
        self.predictor = predictor
        self.kwargs = kwargs

    def __call__(self, params, coordinates):
        output = self.predictor.apply({'params': params}, coordinates, **self.kwargs)
        return output
    
    
class ImagePlaneOperator(object):
    def __init__(self, emission_op):
        """
        Image Plane operator takes in network parameters, coordinates to produce image pixels.
        
        Parameters
        ----------
        emission_op: EmissionOperater, 
            The emission_op samples emission from the neural network along specified grid points. 
        """
        self.emission_op = emission_op

    def __call__(self, params, coordinates, path_lengths):
        """
        Generate image-plane pixels for a given network (emission operator).
        
        Parameters
        ----------
        params, FrozenDict, 
            A dictionary with the neural network parameters.
        coordinates: list of arrays, 
            1. For 3D emission coords=[t, x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
               alpha, beta are image coordinates. These arrays contain the ray integration points
            2. For 2D emission coords=[t, x, y] with each array shape=(nt, num_alpha, ngeo)
        path_lengths: array,
            Path lengths for integration, shape=(nt, num_alpha, num_beta, ngeo)
        Returns
        -------
        images: array, 
            Image plane pixels.
        """
        emission = self.emission_op(params, coordinates)
        images = jnp.sum(emission * path_lengths, axis=-1)
        return images   
    
class VisibilityOperator(ImagePlaneOperator):
    def __init__(self, emission_op):
        """
        Visibility operator takes in network parameters, coordinates, dtft matrices 
        to produce visibility measurements.
        
        Parameters
        ----------
        emission_op: EmissionOperater, 
            The emission_op samples emission from the neural network along specified grid points. 
        """
        super().__init__(emission_op)

    def __call__(self, params, coordinates, path_lengths, dtft_matrices):
        """
        Generate visibility measurements for a given network state and frequency sampling pattern.
        
        Parameters
        ----------
        params, FrozenDict, 
            A dictionary with the neural network parameters.
        coordinates: list of arrays, 
            For 3D emission coords=[t, x, y, z] with each array shape=(nt, num_alpha, num_beta, ngeo)
            alpha, beta are image coordinates. These arrays contain the ray integration points
        path_lengths: array,
            Path lengths for integration, shape=(nt, num_alpha, num_beta, ngeo)
        dtft_matrices: array, shape=(nt, nfreq, npix)
            Here nfreq is the (maximum) number of frequencies sampled at a given observation time and npix=num_alpha*num_beta.
            This array contains all the dtft matrices. 
            
        Returns
        -------
        visibilities: array, shape=(nt, nfreq), 
            Complex visibilities at all observation times.
        images: array, shape=(nt, num_alpha, num_beta)
            Image plane measurements.
        """
        images = super().__call__(params, coordinates, path_lengths)
        visibilities = jnp.stack([jnp.matmul(ft, image.ravel()) for ft, image in zip(dtft_matrices, images)])
        return visibilities, images     

def shard(xs):
    """Split data into shards for multiple devices along the first dimension."""
    return jax.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)

def flattened_traversal(fn):
    def mask(data):
        flat = flax.traverse_util.flatten_dict(data)
        return flax.traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})
    return mask

def get_input_coords(sensor, t_array, ngeo=None, npix=None, batch='rays'):
    """
    Get input coordinates for optimization.
    
    Parameters
    ----------
    sensor: xarray.Dataset
        A dataset with dimensions (geo, pix) and variables: x, y, z, t, deltas. 
        These are the ray coordinates and deltas are the segments lengths used for integration.
    t_array: np.array,
        An array with the measurement times.
    ngeo: int, default=None
        Number of sample points along a ray. If None it is taken from the sensor attributes.
    npix: int, default=None
        Number of pixels/rays. If None it is taken from the sensor attributes.
    batch: 'rays' or 't', default='rays'
        'rays' used to batch (along first axis) single rays.
        't' used to batch whole images (needed for Fourier transform).
    
    Returns
    -------
    coordinates: dict,
        A dictionary with keys: t, x, y, z, d.
    """
    ngeo = ngeo if ngeo else sensor.geo.size
    npix = npix if npix else sensor.pix.size
    nt = len(t_array)
    
    if batch == 'rays':
        data_shape = [nt * npix, ngeo]
    elif batch == 't': 
        data_shape = [nt, sensor.num_alpha, sensor.num_beta, ngeo]
        
    if 't' in sensor:
        sensor = sensor.drop('t')
    interpolated_sensor = sensor.interp(geo=np.linspace(0, sensor.geo.size-1, ngeo),
                                        pix=np.linspace(0, sensor.pix.size-1, npix))
    interpolated_sensor = interpolated_sensor.expand_dims(t=range(nt))
    t = np.broadcast_to(t_array[:, None, None], [nt, npix, ngeo])
    coordinates = {
        't': t.reshape(*data_shape),
        'x': interpolated_sensor.x.data.reshape(*data_shape),
        'y': interpolated_sensor.y.data.reshape(*data_shape),
        'z': interpolated_sensor.z.data.reshape(*data_shape),
        'd': interpolated_sensor.deltas.data.reshape(*data_shape)
    }
    return coordinates

# import jax 
# from jax import jit, random
# from flax import linen as nn
# from typing import Any, Callable
# import functools
# import numpy as np
# import jax.numpy as jnp
# import jax.scipy as jsp
# import flax
# from typing import Sequence
# import bhnerf
# 
# 
# class MLP(nn.Module):
#     """A simple MLP."""
#     
#     net_depth: int = 4
#     net_width: int = 128
#     activation: Callable[..., Any] = nn.relu
#     out_channel: int = 1
#     do_skip: bool = True
#   
#     @nn.compact
#     def __call__(self, x):
#         """Multi-layer perception for nerf.
# 
#         Args:
#           x: jnp.ndarray(float32), [batch_size * n_samples, feature], points.
#           net_depth: int, the depth of the first part of MLP.
#           net_width: int, the width of the first part of MLP.
#           activation: function, the activation function used in the MLP.
#           out_channel: int, the number of alpha_channels.
#           do_skip: boolean, whether or not to use a skip connection
# 
#         Returns:
#           out: jnp.ndarray(float32), [batch_size * n_samples, out_channel].
#         """
#         dense_layer = functools.partial(
#             nn.Dense, kernel_init=jax.nn.initializers.he_uniform())
# 
#         if self.do_skip:
#             skip_layer = self.net_depth // 2
# 
#         inputs = x
#         for i in range(self.net_depth):
#             x = dense_layer(self.net_width)(x)
#             x = self.activation(x)
#             if self.do_skip:
#                 if i % skip_layer == 0 and i > 0:
#                     x = jnp.concatenate([x, inputs], axis=-1)
#         out = dense_layer(self.out_channel)(x)
# 
#         return out
# 
# class PREDICT_EMISSION_2D(nn.Module):
#     """Full function to predict emission at a time step."""
#     posenc_deg: int = 3
#     
#     @nn.compact
#     def __call__(self, x, y, t, v):
# 
#         emission_MLP = MLP()
# 
#         def predict_emission(x, y, t, v):
#             net_input = bhnerf.emission.velocity_warp_2d(x, y, t, v, use_jax=True)
#             valid_inputs_mask = jnp.isfinite(net_input)
#             net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
#             net_output = emission_MLP(posenc(net_input, self.posenc_deg))
#             emission = nn.sigmoid(net_output[..., 0])
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             return emission
# 
#         emission = predict_emission(x, y, t, v)
# 
#         return emission
#     
#     def lossfn(self, params, x, y, d, t, v, target, key):
#         emission = self.apply({'params': params}, x, y, t, v)
#         valid_d_mask = jnp.isfinite(d)
#         d = jnp.where(valid_d_mask, d, jnp.zeros_like(d))
#         rendering = jnp.sum(emission * d, axis=-1)
#         loss = jnp.mean((rendering - target)**2)
#         return loss, [emission, rendering]
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def train_step(self, v, i, x, y, d, t, target, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(state.params, x, y, d, t, v, target, new_key)
#         grads = jax.lax.pmean(grads, axis_name='batch')
#         state = state.apply_gradients(grads=grads)
#         loss, [emission, rendering] = vals
#         return loss, state, emission, rendering, new_key
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def eval_step(self, v, i, x, y, d, t, target, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(state.params, x, y, d, t, v, target, new_key)
#         loss, [emission, rendering] = vals
#         return loss, state, emission, rendering, new_key
# 
# class PREDICT_EMISSION_3D_GRID(nn.Module):
#     """Full function to predict emission with discrete grid at a time step."""
#     
#     grid_res = 64
#     fov = 10.
#     
#     @nn.compact
#     def __call__(self, x, y, z, t, v, axis):
#         
#         grid_init = lambda rng, shape: jnp.ones(shape) * -10.
# #         grid_init = lambda rng, shape: jnp.zeros(shape)
# 
#         grid = self.param('grid', grid_init, (self.grid_res, self.grid_res, self.grid_res)) 
# 
#         def predict_emission(x, y, z, t, v, axis):
#             net_input = bhnerf.emission.velocity_warp((x, y, z), t, v, axis, use_jax=True)
#             valid_inputs_mask = jnp.isfinite(net_input)
#             net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
#             net_input = jnp.moveaxis(net_input, -1, 0)
#             net_input = (net_input + self.fov / 2.) / self.fov * (self.grid_res - 1.)
#             net_output = jax.scipy.ndimage.map_coordinates(grid, net_input, order=1, cval=0.)
#             emission = nn.sigmoid(net_output)
# #             emission = net_output
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             return emission
# 
#         emission = predict_emission(x, y, z, t, v, axis)
# 
#         return emission
#     
#     def lossfn(self, params, x, y, z, d, t, v, axis, target, key):
#         emission = self.apply({'params': params}, x, y, z, t, v, axis)
#         valid_d_mask = jnp.isfinite(d)
#         d = jnp.where(valid_d_mask, d, jnp.zeros_like(d))
#         rendering = jnp.sum(emission * d, axis=-1)
#         loss = jnp.mean((rendering - target)**2)
#         return loss, [emission, rendering]
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def train_step(self, v, axis, i, x, y, z, d, t, target, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, v, axis, target, new_key)
#         grads = jax.lax.pmean(grads, axis_name='batch')
#         state = state.apply_gradients(grads=grads)
#         loss, [emission, rendering] = vals
#         return loss, state, emission, rendering, new_key
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def eval_step(self, v, axis, i, x, y, z, d, t, target, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, v, axis, target, new_key)
#         loss, [emission, rendering] = vals
#         return loss, state, emission, rendering, new_key
#         
# class PREDICT_EMISSION_3D(nn.Module):
#     """Full function to predict emission with MLP at a time step."""
#     posenc_deg: int = 3
#     
#     @nn.compact
#     def __call__(self, x, y, z, t, v, axis):
# 
#         emission_MLP = MLP()
# 
#         def predict_emission(x, y, z, t, v, axis):
#             net_input = bhnerf.emission.velocity_warp((x, y, z), t, v, axis, use_jax=True)
#             valid_inputs_mask = jnp.isfinite(net_input)
#             net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
#             net_output = emission_MLP(posenc(net_input, self.posenc_deg))
#             emission = nn.sigmoid(net_output[..., 0] - 10.)
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             return emission
# 
#         emission = predict_emission(x, y, z, t, v, axis)
# 
#         return emission
#     
#     def lossfn(self, params, x, y, z, d, t, v, axis, target, key):
#         emission = self.apply({'params': params}, x, y, z, t, v, axis)
#         valid_d_mask = jnp.isfinite(d)
#         d = jnp.where(valid_d_mask, d, jnp.zeros_like(d))
#         rendering = jnp.sum(emission * d, axis=-1)
#         loss = jnp.mean((rendering - target)**2)
#         return loss, [emission, rendering]
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def train_step(self, v, axis, i, x, y, z, d, t, target, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, v, axis, target, new_key)
#         grads = jax.lax.pmean(grads, axis_name='batch')
#         state = state.apply_gradients(grads=grads)
#         loss, [emission, rendering] = vals
#         return loss, state, emission, rendering, new_key
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def eval_step(self, v, axis, i, x, y, z, d, t, target, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, v, axis, target, new_key)
#         loss, [emission, rendering] = vals
#         return loss, state, emission, rendering, new_key
#     
# class PREDICT_EMISSION_3D_NO_VELOCITY(nn.Module):
#     """Full function to predict emission with MLP at a time step."""
#     posenc_deg: int = 3
#     
#     @nn.compact
#     def __call__(self, x, y, z, t, v, axis):
# 
#         emission_MLP = MLP()
# 
#         def predict_emission(x, y, z, t, v, axis):
#             net_input = jnp.stack([x, y, z, t], axis=-1)
#             valid_inputs_mask = jnp.isfinite(net_input)
#             net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
#             net_output = emission_MLP(posenc(net_input, self.posenc_deg))
#             emission = nn.sigmoid(net_output[..., 0] - 10.)
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             return emission
# 
#         emission = predict_emission(x, y, z, t, v, axis)
# 
#         return emission
#     
#     def lossfn(self, params, x, y, z, d, t, v, axis, target, key):
#         emission = self.apply({'params': params}, x, y, z, t, v, axis)
#         valid_d_mask = jnp.isfinite(d)
#         d = jnp.where(valid_d_mask, d, jnp.zeros_like(d))
#         rendering = jnp.sum(emission * d, axis=-1)
#         loss = jnp.mean((rendering - target)**2)
#         return loss, [emission, rendering]
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def train_step(self, v, axis, i, x, y, z, d, t, target, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, v, axis, target, new_key)
#         grads = jax.lax.pmean(grads, axis_name='batch')
#         state = state.apply_gradients(grads=grads)
#         loss, [emission, rendering] = vals
#         return loss, state, emission, rendering, new_key
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def eval_step(self, v, axis, i, x, y, z, d, t, target, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, v, axis, target, new_key)
#         loss, [emission, rendering] = vals
#         return loss, state, emission, rendering, new_key
#     
# class PREDICT_EMISSION_AND_ROTAXIS_3D(nn.Module):
#     """Full function to predict emission at a time step."""
#     posenc_deg: int = 3
#     
#     @nn.compact
#     def __call__(self, x, y, z, t, v):
# 
#         emission_MLP = MLP()
#         axis = self.param('axis', lambda key, values: jnp.array(values, dtype=jnp.float32), [0, 0, -1])
#         # axis = jnp.array([ 0.5      ,  0.       , -0.8660254])
#         
#         def predict_emission(x, y, z, t, v, axis):
#             net_input = bhnerf.emission.velocity_warp((x, y, z), t, v, axis, use_jax=True)
#             valid_inputs_mask = jnp.isfinite(net_input)
#             net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
#             net_output = emission_MLP(posenc(net_input, self.posenc_deg))
#             emission = nn.sigmoid(net_output[..., 0])
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             
#             """
#             # Ground truth emission
#             radius_gt = 3.5
#             theta_gt = np.pi / 3.
#             phi_gt = 0.0
#             center = radius_gt * jnp.array([jnp.cos(phi_gt)*jnp.sin(theta_gt), 
#                                             jnp.sin(phi_gt)*jnp.sin(theta_gt), 
#                                             jnp.cos(theta_gt)])
#             std = 0.4
#             emission = jnp.exp(-0.5*( (net_input[...,0] - center[0])**2 + (net_input[...,1] - center[1])**2 + (net_input[...,2] - center[2])**2) / std**2)
#             emission = jnp.where(jnp.isfinite(emission), emission, jnp.zeros_like(emission))
#             """
#             return emission
# 
#         emission = predict_emission(x, y, z, t, v, axis)
# 
#         return emission, axis
#     
#     def lossfn(self, params, x, y, z, d, t, v, target, key):
#         emission, axis = self.apply({'params': params}, x, y, z, t, v)
#         rendering = jnp.sum(emission * d, axis=-1)
#         loss = jnp.mean((rendering - target)**2 + (jnp.sqrt(jnp.dot(axis, axis))-1)**2)
#         return loss, [emission, rendering, axis]
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def train_step(self, v, i, x, y, z, d, t, target, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, v, target, new_key)
#         grads = jax.lax.pmean(grads, axis_name='batch')
#         state = state.apply_gradients(grads=grads)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def eval_step(self, v, i, x, y, z, d, t, target, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, v, target, new_key)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
#       
# class PREDICT_EMISSION_AND_MLP_ROTAXIS_3D(nn.Module):
#     """Full function to predict emission at a time step."""
#     posenc_deg: int = 3
# 
#     @nn.compact
#     def __call__(self, x, y, z, t, velocity, tstart, tstop):
# 
#         emission_MLP = MLP()
#         axis_MLP = MLP(net_depth=3, net_width=32, out_channel=3, do_skip=False, activation=lambda x: x)
#         
#         def predict_emission(x, y, z, t, velocity, axis, tstart, tstop):
#             net_input = bhnerf.emission.velocity_warp(
#                 (x, y, z), t, velocity, axis, tstart, tstop, use_jax=True)
#             valid_inputs_mask = jnp.isfinite(net_input)
#             net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
#             net_output = emission_MLP(posenc(net_input, self.posenc_deg))
#             emission = nn.sigmoid(net_output[..., 0] - 10.)
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             return emission
#         
#         axis_net_input = jnp.array([-0.59162719, -2.67309611, -0.56935515]) # Normal distribution sampled fixed input
#         axis = axis_MLP(axis_net_input)
#         emission = predict_emission(x, y, z, t, velocity, axis, tstart, tstop)
# 
#         return emission, axis
#     
#     def lossfn(self, params, x, y, z, d, t, velocity, uv, fov, tstart, tstop, target, rendering_true, sigma, bg_image, key):
#         emission, axis = self.apply({'params': params}, x, y, z, t, velocity, tstart, tstop)
#         valid_d_mask = jnp.isfinite(d)
#         d = jnp.where(valid_d_mask, d, jnp.zeros_like(d))
#         rendering = jnp.sum(emission * d, axis=-1) + bg_image
#         loss = jnp.mean((rendering - rendering_true)**2)
#         return loss, [emission, rendering, axis]
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def train_step(self, velocity, i, x, y, z, d, t, uv, fov, tstart, tstop, target, rendering_true, sigma, bg_image, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, uv, fov, tstart, tstop, target, rendering_true, sigma, bg_image, new_key)
#         grads = jax.lax.pmean(grads, axis_name='batch')
#         state = state.apply_gradients(grads=grads)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def eval_step(self, velocity, i, x, y, z, d, t, uv, fov, tstart, tstop, target, rendering_true, sigma, bg_image, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, uv, fov, tstart, tstop, target, rendering_true, sigma, bg_image, new_key)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
#       
# class PREDICT_EMISSION_AND_ROTAXIS_3D_FROM_VIS(nn.Module):
#     """Full function to predict emission at a time step."""
#     posenc_deg: int = 3
# 
#     @nn.compact
#     def __call__(self, x, y, z, t, velocity, tstart, tstop):
# 
#         emission_MLP = MLP()
#         axis = self.param('axis', lambda key, values: jnp.array(values, dtype=jnp.float32), [0, 0, -1])   
#         # axis = jnp.array([0.5, 0., -0.8660254])
#         # axis = array([-5.00000000e-01,  6.12323400e-17, -8.66025404e-01])
#         def predict_emission(x, y, z, t, velocity, axis, tstart, tstop):
#             net_input = bhnerf.emission.velocity_warp(
#                 (x, y, z), t, velocity, axis, tstart, tstop, use_jax=True)
#             valid_inputs_mask = jnp.isfinite(net_input)
#             net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
#             net_output = emission_MLP(posenc(net_input, self.posenc_deg))
#             emission = nn.sigmoid(net_output[..., 0])
#             # emission = nn.relu(net_output[..., 0])
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             
#             # Ground truth emission
#             # radius_gt = 3.5
#             # theta_gt = np.pi / 3.
#             # phi_gt = 0.0
#             # center = radius_gt * jnp.array([jnp.cos(phi_gt)*jnp.sin(theta_gt), 
#             #                                 jnp.sin(phi_gt)*jnp.sin(theta_gt), 
#             #                                 jnp.cos(theta_gt)])
#             # std = 0.4
#             # emission = jnp.exp(-0.5*( (net_input[...,0] - center[0])**2 + (net_input[...,1] - center[1])**2 +
#             #                          (net_input[...,2] - center[2])**2) / std**2)
#             # normalization_factor = 0.03305721036378928
#             # emission *= normalization_factor
#             # emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
# 
#             return emission
#         
#         emission = predict_emission(x, y, z, t, velocity, axis, tstart, tstop)
# 
#         return emission, axis
#     
#     def lossfn(self, params, x, y, z, d, t, velocity, uv, fov, tstart, tstop, target, key):
#         emission, axis = self.apply({'params': params}, x, y, z, t, velocity, tstart, tstop)
#         valid_d_mask = jnp.isfinite(d)
#         d = jnp.where(valid_d_mask, d, jnp.zeros_like(d))
#         rendering = jnp.sum(emission * d, axis=-1)
#         valid_uv_mask = jnp.isfinite(uv[0])
#         uv = jnp.where(valid_uv_mask, uv[0], jnp.zeros_like(uv[0]))
#         visibilities = observation_utils.observe_nonoise(
#             rendering[0], uv[...,0], uv[...,1], fov, use_jax=True)
#         visibilities = jnp.where(valid_uv_mask[...,0], visibilities, jnp.zeros_like(visibilities))
#         loss = jnp.mean(jnp.abs(visibilities - target)**2 + (jnp.sqrt(jnp.dot(axis, axis))-1)**2)
#         return loss, [emission, rendering, axis]
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def train_step(self, velocity, i, x, y, z, d, t, uv, fov, tstart, tstop, target, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, uv, fov, tstart, tstop, target, new_key)
#         grads = jax.lax.pmean(grads, axis_name='batch')
#         state = state.apply_gradients(grads=grads)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def eval_step(self, velocity, i, x, y, z, d, t, uv, fov, tstart, tstop, target, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, uv, fov, tstart, tstop, target, new_key)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
# 
# class PREDICT_EMISSION_AND_MLP_ROTAXIS_3D_FROM_VIS(nn.Module):
#     """Full function to predict emission at a time step."""
#     posenc_deg: int = 3
# 
#     @nn.compact
#     def __call__(self, x, y, z, t, velocity, tstart, tstop):
# 
#         emission_MLP = MLP()
#         axis_MLP = MLP(net_depth=3, net_width=32, out_channel=3, do_skip=False, activation=lambda x: x)
#         
#         def predict_emission(x, y, z, t, velocity, axis, tstart, tstop):
#             net_input = bhnerf.emission.velocity_warp(
#                 (x, y, z), t, velocity, axis, tstart, tstop, use_jax=True)
#             valid_inputs_mask = jnp.isfinite(net_input)
#             net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
#             net_output = emission_MLP(posenc(net_input, self.posenc_deg))
#             emission = nn.sigmoid(net_output[..., 0])
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             
#             # Ground truth emission
#             # radius_gt = 3.5
#             # theta_gt = np.pi / 3.
#             # phi_gt = 0.0
#             # center = radius_gt * jnp.array([jnp.cos(phi_gt)*jnp.sin(theta_gt), 
#             #                                 jnp.sin(phi_gt)*jnp.sin(theta_gt), 
#             #                                 jnp.cos(theta_gt)])
#             # std = 0.4
#             # emission = jnp.exp(-0.5*( (net_input[...,0] - center[0])**2 + (net_input[...,1] - center[1])**2 +
#             #                          (net_input[...,2] - center[2])**2) / std**2)
#             # normalization_factor = 0.03
#             # emission *= normalization_factor
#             # emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
# 
#             return emission
#         
#         axis_net_input = jnp.array([-0.59162719, -2.67309611, -0.56935515]) # Normal distribution sampled fixed input
#         axis = axis_MLP(axis_net_input)
#         emission = predict_emission(x, y, z, t, velocity, axis, tstart, tstop)
# 
#         return emission, axis
#     
#     def lossfn(self, params, x, y, z, d, t, velocity, uv, fov, tstart, tstop, target, sigma, key):
#         emission, axis = self.apply({'params': params}, x, y, z, t, velocity, tstart, tstop)
#         valid_d_mask = jnp.isfinite(d)
#         d = jnp.where(valid_d_mask, d, jnp.zeros_like(d))
#         rendering = jnp.sum(emission * d, axis=-1)
#         valid_uv_mask = jnp.isfinite(uv)
#         uv = jnp.where(valid_uv_mask, uv, jnp.zeros_like(uv))
#         visibilities = observation_utils.observe_nonoise(
#             rendering, uv[...,0], uv[...,1], fov, use_jax=True)
#         visibilities = jnp.where(valid_uv_mask[...,0], visibilities, jnp.zeros_like(visibilities))
#         loss = jnp.mean((jnp.abs(visibilities - target)/sigma)**2) # + (jnp.sqrt(jnp.dot(axis, axis))-1)**2
#         return loss, [emission, rendering, axis]
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def train_step(self, velocity, i, x, y, z, d, t, uv, fov, tstart, tstop, target, sigma, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, uv, fov, tstart, tstop, target, sigma, new_key)
#         grads = jax.lax.pmean(grads, axis_name='batch')
#         state = state.apply_gradients(grads=grads)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def eval_step(self, velocity, i, x, y, z, d, t, uv, fov, tstart, tstop, target, sigma, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, uv, fov, tstart, tstop, target, sigma, new_key)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
# 
# class PREDICT_EMISSION_AND_MLP_ROTAXIS_3D_FROM_VIS_W_BG(nn.Module):
#     """Full function to predict emission at a time step."""
#     posenc_deg: int = 3
#     axis_net_depth: int = 3
#     
#     @nn.compact
#     def __call__(self, x, y, z, t, velocity, tstart, tstop):
# 
#         emission_MLP = MLP()
#         axis_MLP = MLP(net_depth=self.axis_net_depth, net_width=32, out_channel=3, do_skip=False, activation=lambda x: x)
#         
#         def predict_emission(x, y, z, t, velocity, axis, tstart, tstop):
#             net_input = bhnerf.emission.velocity_warp(
#                 (x, y, z), t, velocity, axis, tstart, tstop, use_jax=True)
#             valid_inputs_mask = jnp.isfinite(net_input)
#             net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
#             net_output = emission_MLP(posenc(net_input, self.posenc_deg))
#             emission = nn.sigmoid(net_output[..., 0] - 10.)
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             return emission
#         axis_net_input = jnp.array([-0.59162719, -2.67309611, -0.56935515]) # Normal distribution sampled fixed input
#         
#         axis = axis_MLP(axis_net_input)
#         # axis = self.param('axis', lambda key, values: jnp.array(values, dtype=jnp.float32), [0, 0, -1])
#         # axis = jnp.array([-0.70062927, -0.50903696, -0.5])
#         # axis =  jnp.array([0.5, 0., -0.8660254])
#         emission = predict_emission(x, y, z, t, velocity, axis, tstart, tstop)
# 
#         return emission, axis
#     
#     def lossfn(self, params, x, y, z, d, t, velocity, ft_mats, tstart, tstop, target, sigma, bg_image, window, key):
#         emission, axis = self.apply({'params': params}, x, y, z, t, velocity, tstart, tstop)
#         rendering = jnp.sum(emission * d, axis=-1)
#         fourier = jnp.fft.fft2(jnp.fft.ifftshift(rendering)) * jnp.fft.fft2(jnp.fft.ifftshift(window))
#         rendering = jnp.fft.ifftshift(jnp.fft.ifft2(fourier)).real + bg_image
#         vis = jnp.stack([jnp.matmul(ft, image.ravel()) for ft, image in zip(ft_mats, rendering)])
#         valid_mask = jnp.isfinite(sigma)
#         vis = jnp.where(valid_mask, vis, jnp.zeros_like(vis))
#         loss = jnp.mean((jnp.abs(vis - target)/sigma)**2)
#         return loss, [emission, rendering, axis]
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def train_step(self, velocity, i, x, y, z, d, t, ft_mats, tstart, tstop, target, sigma, bg_image, window, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, ft_mats, tstart, tstop, target, sigma, bg_image, window, new_key)
#         grads = jax.lax.pmean(grads, axis_name='batch')
#         state = state.apply_gradients(grads=grads)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def eval_step(self, velocity, i, x, y, z, d, t, ft_mats, tstart, tstop, target, sigma, bg_image, window, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, ft_mats, tstart, tstop, target, sigma, bg_image, window, new_key)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
# 
# class PREDICT_EMISSION_AND_MLP_ROTAXIS_3D_FROM_IMAGE_W_BG(nn.Module):
#     """Full function to predict emission at a time step."""
#     posenc_deg: int = 3
#     axis_net_depth: int = 3
#     
#     @nn.compact
#     def __call__(self, x, y, z, t, velocity, tstart, tstop):
# 
#         emission_MLP = MLP()
#         axis_MLP = MLP(net_depth=self.axis_net_depth, net_width=32, out_channel=3, do_skip=False, activation=lambda x: x)
#          
#         def predict_emission(x, y, z, t, velocity, axis, tstart, tstop):
#             net_input = bhnerf.emission.velocity_warp(
#                 (x, y, z), t, velocity, axis, tstart, tstop, use_jax=True)
#             valid_inputs_mask = jnp.isfinite(net_input)
#             net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
#             net_output = emission_MLP(posenc(net_input, self.posenc_deg))
#             emission = nn.sigmoid(net_output[..., 0] - 10.)
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             return emission
#         
#         axis_net_input = jnp.array([-0.59162719, -2.67309611, -0.56935515])  # Normal distribution sampled fixed input
#         axis = axis_MLP(axis_net_input)
#         # axis =  jnp.array([0.5, 0., -0.8660254])
#         # axis =  jnp.array([-0.70062927, -0.50903696, -0.5])
#         emission = predict_emission(x, y, z, t, velocity, axis, tstart, tstop)
# 
#         return emission, axis
#     
#     def lossfn(self, params, x, y, z, d, t, velocity, ft_mats, tstart, tstop, target, sigma, bg_image, window, key):
#         emission, axis = self.apply({'params': params}, x, y, z, t, velocity, tstart, tstop)
#         rendering = jnp.sum(emission * d, axis=-1)
#         fourier = jnp.fft.fft2(jnp.fft.ifftshift(rendering)) * jnp.fft.fft2(jnp.fft.ifftshift(window))
#         rendering = jnp.fft.ifftshift(jnp.fft.ifft2(fourier)).real + bg_image
#         loss = jnp.mean((jnp.abs(rendering - target))**2)
#         return loss, [emission, rendering, axis]
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def train_step(self, velocity, i, x, y, z, d, t, ft_mats, tstart, tstop, target, sigma, bg_image, window, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, ft_mats, tstart, tstop, target, sigma, bg_image, window, new_key)
#         grads = jax.lax.pmean(grads, axis_name='batch')
#         state = state.apply_gradients(grads=grads)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def eval_step(self, velocity, i, x, y, z, d, t, ft_mats, tstart, tstop, target, sigma, bg_image, window, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, ft_mats, tstart, tstop, target, sigma, bg_image, window, new_key)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
# 
# class PREDICT_EMISSION_AND_DIRECT_ROTAXIS_3D_FROM_VIS_W_BG(nn.Module):
#     """Full function to predict emission at a time step."""
#     posenc_deg: int = 3
#     
#     @nn.compact
#     def __call__(self, x, y, z, t, velocity, tstart, tstop, axis_init):
# 
#         emission_MLP = MLP()
# 
#         def predict_emission(x, y, z, t, velocity, axis, tstart, tstop):
#             net_input = bhnerf.emission.velocity_warp(
#                 (x, y, z), t, velocity, axis, tstart, tstop, use_jax=True)
#             valid_inputs_mask = jnp.isfinite(net_input)
#             net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
#             net_output = emission_MLP(posenc(net_input, self.posenc_deg))
#             emission = nn.sigmoid(net_output[..., 0] - 10.)
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             return emission
#         
#         axis = self.param('axis', lambda key, values: jnp.array(values, dtype=jnp.float32), axis_init)
#         emission = predict_emission(x, y, z, t, velocity, axis, tstart, tstop)
# 
#         return emission, axis
#     
#     def lossfn(self, params, x, y, z, d, t, velocity, ft_mats, tstart, tstop, target, sigma, bg_image, window, axis_init, key):
#         emission, axis = self.apply({'params': params}, x, y, z, t, velocity, tstart, tstop, axis_init)
#         rendering = jnp.sum(emission * d, axis=-1) + bg_image
#         # fourier = jnp.fft.fft2(jnp.fft.ifftshift(rendering, axes=(-2, -1))) * jnp.fft.fft2(jnp.fft.ifftshift(window))
#         # rendering = jnp.fft.ifftshift(jnp.fft.ifft2(fourier), axes=(-2, -1)).real + bg_image
#         vis = jnp.stack([jnp.matmul(ft, image.ravel()) for ft, image in zip(ft_mats, rendering)])
#         valid_mask = jnp.isfinite(sigma)
#         vis = jnp.where(valid_mask, vis, jnp.zeros_like(vis))
#         loss = jnp.mean((jnp.abs(vis - target)/sigma)**2)
#         return loss, [emission, rendering, axis]
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def train_step(self, velocity, i, x, y, z, d, t, ft_mats, tstart, tstop, target, sigma, bg_image, window, axis_init, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, ft_mats, tstart, tstop, target, sigma, bg_image, window, axis_init, new_key)
#         grads = jax.lax.pmean(grads, axis_name='batch')
#         state = state.apply_gradients(grads=grads)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def eval_step(self, velocity, i, x, y, z, d, t, ft_mats, tstart, tstop, target, sigma, bg_image, window, axis_init, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, ft_mats, tstart, tstop, target, sigma, bg_image, window, axis_init, new_key)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
#     
# class PREDICT_EMISSION_AND_DIRECT_ROTAXIS_3D_FROM_IMAGE_W_BG(nn.Module):
#     """Full function to predict emission at a time step."""
#     posenc_deg: int = 3
#     
#     @nn.compact
#     def __call__(self, x, y, z, t, velocity, tstart, tstop, axis_init):
# 
#         emission_MLP = MLP()
#          
#         def predict_emission(x, y, z, t, velocity, axis, tstart, tstop):
#             net_input = bhnerf.emission.velocity_warp(
#                 (x, y, z), t, velocity, axis, tstart, tstop, use_jax=True)
#             valid_inputs_mask = jnp.isfinite(net_input)
#             net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
#             net_output = emission_MLP(posenc(net_input, self.posenc_deg))
#             emission = nn.sigmoid(net_output[..., 0] - 10.)
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             return emission
#         
#         axis = self.param('axis', lambda key, values: jnp.array(values, dtype=jnp.float32), axis_init)
#         emission = predict_emission(x, y, z, t, velocity, axis, tstart, tstop)
# 
#         return emission, axis
#     
#     def lossfn(self, params, x, y, z, d, t, velocity, ft_mats, tstart, tstop, target, sigma, bg_image, window, axis_init, key):
#         emission, axis = self.apply({'params': params}, x, y, z, t, velocity, tstart, tstop, axis_init)
#         rendering = jnp.sum(emission * d, axis=-1)
#         fourier = jnp.fft.fft2(jnp.fft.ifftshift(rendering, axes=(-2, -1))) * jnp.fft.fft2(jnp.fft.ifftshift(window))
#         rendering = jnp.fft.ifftshift(jnp.fft.ifft2(fourier), axes=(-2, -1)).real + bg_image
#         loss = jnp.mean((jnp.abs(rendering - target))**2)
#         return loss, [emission, rendering, axis]
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def train_step(self, velocity, i, x, y, z, d, t, ft_mats, tstart, tstop, target, sigma, bg_image, window, axis_init, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, ft_mats, tstart, tstop, target, sigma, bg_image, window, axis_init, new_key)
#         grads = jax.lax.pmean(grads, axis_name='batch')
#         state = state.apply_gradients(grads=grads)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def eval_step(self, velocity, i, x, y, z, d, t, ft_mats, tstart, tstop, target, sigma, bg_image, window, axis_init, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, ft_mats, tstart, tstop, target, sigma, bg_image, window, axis_init, new_key)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
#     
# class PREDICT_EMISSION_AND_MLP_ROTAXIS_3D_FROM_IMAGE_W_BG1(nn.Module):
#     """Full function to predict emission at a time step."""
#     posenc_deg: int = 3
# 
#     @nn.compact
#     def __call__(self, x, y, z, t, velocity):
# 
#         emission_MLP = MLP()
#         axis_MLP = MLP(net_depth=3, net_width=32, out_channel=3, do_skip=False, activation=lambda x: x)
#         
#         def predict_emission(x, y, z, t, velocity, axis):
#             net_input = bhnerf.emission.velocity_warp(
#                 x, y, z, t, velocity, axis, use_jax=True)
#             valid_inputs_mask = jnp.isfinite(net_input)
#             net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
#             net_output = emission_MLP(posenc(net_input, self.posenc_deg))
#             emission = nn.sigmoid(net_output[..., 0] - 10.)
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             return emission
#         
#         axis_net_input = jnp.array([-0.59162719, -2.67309611, -0.56935515]) # Normal distribution sampled fixed input
#         axis = axis_MLP(axis_net_input)
#         emission = predict_emission(x, y, z, t, velocity, axis)
# 
#         return emission, axis
#     
#     def lossfn(self, params, x, y, z, d, t, velocity, target, bg_image, key):
#         emission, axis = self.apply({'params': params}, x, y, z, t, velocity)
#         rendering = jnp.sum(emission * d, axis=-1) + bg_image
#         loss = jnp.mean((jnp.abs(rendering - target))**2)
#         return loss, [emission, rendering, axis]
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def train_step(self, velocity, i, x, y, z, d, t, target, bg_image, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, target, bg_image, new_key)
#         grads = jax.lax.pmean(grads, axis_name='batch')
#         state = state.apply_gradients(grads=grads)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def eval_step(self, velocity, i, x, y, z, d, t, target, bg_image, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, target, bg_image, new_key)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
#     
# class PREDICT_EMISSION_AND_MLP_ROTAXIS_3D_FROM_VIS_W_BG_PRATUL(nn.Module):
#     """Full function to predict emission at a time step."""
#     posenc_deg: int = 3
# 
#     @nn.compact
#     def __call__(self, x, y, z, t, velocity, tstart, tstop):
# 
#         emission_MLP = MLP()
#         axis_MLP = MLP(net_depth=3, net_width=32, out_channel=3, do_skip=False, activation=lambda x: x)
#         
#         def predict_emission(x, y, z, t, velocity, axis, tstart, tstop):
#             net_input = bhnerf.emission.velocity_warp(
#                 (x, y, z), t, velocity, axis, tstart, tstop, use_jax=True)
#             valid_inputs_mask = jnp.isfinite(net_input)
#             net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
#             net_output = emission_MLP(posenc(net_input, self.posenc_deg))
#             emission = nn.sigmoid(net_output[..., 0] - 10.)
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             
# #             Ground truth emission
# #             radius_gt = 3.5
# #             theta_gt = np.pi / 3.
# #             phi_gt = 0.0
# #             center = radius_gt * jnp.array([jnp.cos(phi_gt)*jnp.sin(theta_gt), 
# #                                             jnp.sin(phi_gt)*jnp.sin(theta_gt), 
# #                                             jnp.cos(theta_gt)])
# #             std = 0.4
# #             emission = jnp.exp(-0.5*( (net_input[...,0] - center[0])**2 + (net_input[...,1] - center[1])**2 +
# #                                      (net_input[...,2] - center[2])**2) / std**2)
# #             normalization_factor = 1.0
# #             emission *= normalization_factor
# #             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             
#             return emission
#         
#         axis_net_input = jnp.array([-0.59162719, -2.67309611, -0.56935515]) # Normal distribution sampled fixed input
#         axis = axis_MLP(axis_net_input)
#         emission = predict_emission(x, y, z, t, velocity, axis, tstart, tstop)
# 
#         return emission, axis
#     
#     def lossfn(self, params, x, y, z, d, t, velocity, uv, fov, tstart, tstop, target, rendering_true, sigma, bg_image, key):
#         emission, axis = self.apply({'params': params}, x, y, z, t, velocity, tstart, tstop)
#         valid_d_mask = jnp.isfinite(d)
#         d = jnp.where(valid_d_mask, d, jnp.zeros_like(d))
#         rendering = jnp.sum(emission * d, axis=-1) + bg_image
#         valid_uv_mask = jnp.isfinite(uv)
#         uv = jnp.where(valid_uv_mask, uv, jnp.zeros_like(uv))
#         visibilities = observation_utils.observe_nonoise(
#             rendering, uv[...,0], uv[...,1], fov, use_jax=True)
#         visibilities = jnp.where(valid_uv_mask[...,0], visibilities, jnp.zeros_like(visibilities))
#         
#         # Pratul testing with same observation function for us and ground truth
#         target = observation_utils.observe_nonoise(
#             rendering_true, uv[...,0], uv[...,1], fov, use_jax=True)
#         target = jnp.where(valid_uv_mask[...,0], target, jnp.zeros_like(target))
#         
#         loss = jnp.mean((jnp.abs(visibilities - target)/sigma)**2)
#         return loss, [emission, rendering, axis]
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def train_step(self, velocity, i, x, y, z, d, t, uv, fov, tstart, tstop, target, rendering_true, sigma, bg_image, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, uv, fov, tstart, tstop, target, rendering_true, sigma, bg_image, new_key)
#         grads = jax.lax.pmean(grads, axis_name='batch')
#         state = state.apply_gradients(grads=grads)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def eval_step(self, velocity, i, x, y, z, d, t, uv, fov, tstart, tstop, target, rendering_true, sigma, bg_image, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, uv, fov, tstart, tstop, target, rendering_true, sigma, bg_image, new_key)
#         loss, [emission, rendering, axis] = vals
#         return loss, state, emission, rendering, axis, new_key
#      
# class PREDICT_EMISSION_AND_ROTAXIS_3D_FROM_FOURIER(nn.Module):
#     """Full function to predict emission at a time step."""
#     posenc_deg: int = 3
#     
#     @nn.compact
#     def __call__(self, x, y, z, t, velocity, tstart, tstop):
# 
#         emission_MLP = MLP()
#         # axis = self.param('axis', lambda key, values: jnp.array(values, dtype=jnp.float32), [0, 0, -1])   
#         axis = jnp.array([ 0.5      ,  0.       , -0.8660254])
#         
#         def predict_emission(x, y, z, t, velocity, axis, tstart, tstop):
#             net_input = bhnerf.emission.velocity_warp(
#                 (x, y, z), t, velocity, axis, tstart, tstop, use_jax=True)
#             valid_inputs_mask = jnp.isfinite(net_input)
#             net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
#             net_output = emission_MLP(posenc(net_input, self.posenc_deg))
#             emission = nn.sigmoid(net_output[..., 0])
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             
#             """
#             # Ground truth emission
#             radius_gt = 3.5
#             theta_gt = np.pi / 3.
#             phi_gt = 0.0
#             center = radius_gt * jnp.array([jnp.cos(phi_gt)*jnp.sin(theta_gt), 
#                                             jnp.sin(phi_gt)*jnp.sin(theta_gt), 
#                                             jnp.cos(theta_gt)])
#             std = 0.4
#             emission = jnp.exp(-0.5*( (net_input[...,0] - center[0])**2 + (net_input[...,1] - center[1])**2 + (net_input[...,2] - center[2])**2) / std**2)
#             normalization_factor = 0.03
#             emission *= normalization_factor
#             emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))
#             """
#             return emission
#         
#         emission = predict_emission(x, y, z, t, velocity, axis, tstart, tstop)
# 
#         return emission, axis
#     
#     def lossfn(self, params, x, y, z, d, t, velocity, tstart, tstop, target, key):
#         emission, axis = self.apply({'params': params}, x, y, z, t, velocity, tstart, tstop)
#         rendering = jnp.sum(emission * d, axis=-1)
#         fourier = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(rendering)))
#         loss = jnp.mean(jnp.abs(fourier - target)**2 + (jnp.sqrt(jnp.dot(axis, axis))-1)**2)
#         return loss, [emission, rendering, axis, fourier]
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def train_step(self, velocity, i, x, y, z, d, t, tstart, tstop, target, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, tstart, tstop, target, new_key)
#         grads = jax.lax.pmean(grads, axis_name='batch')
#         state = state.apply_gradients(grads=grads)
#         loss, [emission, rendering, axis, fourier] = vals
#         return loss, state, emission, rendering, axis, new_key, fourier
# 
#     @functools.partial(jit, static_argnums=(0, 1))
#     def eval_step(self, velocity, i, x, y, z, d, t, tstart, tstop, target, state, key):
#         key, new_key = random.split(key)
#         vals, grads = jax.value_and_grad(self.lossfn, argnums=(0), has_aux=True)(
#             state.params, x, y, z, d, t, velocity, tstart, tstop, target, new_key)
#         loss, [emission, rendering, axis, fourier] = vals
#         return loss, state, emission, rendering, axis, new_key, fourier
#     
# def safe_sin(x):
#     """jnp.sin() on a TPU will NaN out for moderately large values."""
#     return jnp.sin(x % (100 * jnp.pi))
# 
# def posenc(x, deg):
#     """Concatenate `x` with a positional encoding of `x` with degree `deg`.
# 
#     Instead of computing [sin(x), cos(x)], we use the trig identity
#     cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).
# 
#     Args:
#     x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].
#     deg: int, the degree of the encoding.
# 
#     Returns:
#     encoded: jnp.ndarray, encoded variables.
#     """
#     if deg == 0:
#         return x
#     scales = jnp.array([2**i for i in range(deg)])
#     xb = jnp.reshape((x[..., None, :] * scales[:, None]),
#                      list(x.shape[:-1]) + [-1])
#     four_feat = safe_sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
#     return jnp.concatenate([x] + [four_feat], axis=-1)
# 
