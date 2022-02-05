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