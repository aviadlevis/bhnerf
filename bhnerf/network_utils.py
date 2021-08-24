from flax import linen as nn
from typing import Any, Callable
import numpy as np
import jax.numpy as jnp

class MLP(nn.Module):
    """A simple MLP."""

    net_depth: int = 4
    net_width: int = 256
    activation: Callable[..., Any] = nn.elu
    out_channel: int = 1
    do_skip: bool = True

    @nn.compact
    def __call__(self, x):
        """Multi-layer perception for nerf.

        Args:
          x: jnp.ndarray(float32), [batch_size * n_samples, feature], points.
          net_depth: int, the depth of the first part of MLP.
          net_width: int, the width of the first part of MLP.
          activation: function, the activation function used in the MLP.
          out_channel: int, the number of alpha_channels.
          do_skip: boolean, whether or not to use a skip connection

        Returns:
          out: jnp.ndarray(float32), [batch_size * n_samples, out_channel].
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

class PREDICT_EMISSION(nn.Module):
    """Full function to predict emission at a time step."""

    @nn.compact
    def __call__(self, x, y, t):

        emission_MLP = MLP()

        def predict_emission(x, y, t):
    
            radius = jnp.sqrt(x**2 + y**2)
            theta = jnp.arctan2(y, x)

            theta_rot = theta - 2 * jnp.pi * t / orbit_period * radius**(-3/2)
            x_rot = radius * jnp.cos(theta_rot)
            y_rot = radius * jnp.sin(theta_rot)

            net_input = jnp.stack([x_rot, y_rot], axis=-1)
            valid_inputs_mask = jnp.isfinite(net_input)
            net_input = jnp.where(valid_inputs_mask, net_input, jnp.zeros_like(net_input))
            net_output = emission_MLP(posenc(net_input, posenc_deg))
            emission = nn.sigmoid(net_output[..., 0])
            emission = jnp.where(valid_inputs_mask[..., 0], emission, jnp.zeros_like(emission))

            # Ground truth emission
            # radius_gt = 3.5
            # theta_gt = np.pi / 2.
            # center = [radius_gt * jnp.cos(theta_gt), radius_gt * jnp.sin(theta_gt)]
            # std = 0.4
            # emission = jnp.exp(-0.5*( (y_rot - center[1])**2 + (x_rot - center[0])**2 ) / std**2)
            # emission = jnp.where(jnp.isfinite(emission), emission, jnp.zeros_like(emission))

            return emission

        emission = predict_emission(x, y, t)

        return emission

def safe_sin(x):
    """jnp.sin() on a TPU will NaN out for moderately large values."""
    return jnp.sin(x % (100 * jnp.pi))

def posenc(x, deg):
    """Concatenate `x` with a positional encoding of `x` with degree `deg`.

    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Args:
    x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].
    deg: int, the degree of the encoding.

    Returns:
    encoded: jnp.ndarray, encoded variables.
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

def get_input_coords(sensor, nt, ngeo=None, npix=None):
    
    ngeo = ngeo if ngeo else sensor.geo.size
    npix = npix if npix else sensor.pix.size
    
    if 't' in sensor:
        sensor = sensor.drop('t')
        
    interpolated_sensor = sensor.expand_dims(t=nt).interp(
        geo=np.linspace(0, sensor.geo.size-1, ngeo),
        pix=np.linspace(0, sensor.pix.size-1, npix)
    )
    t = np.broadcast_to(np.linspace(0, 1, nt)[:, None, None], [nt, npix, ngeo])
    coordinates = {
        't': t.reshape(nt*npix, ngeo),
        'x': interpolated_sensor.x.data.reshape(nt*npix, ngeo),
        'y': interpolated_sensor.y.data.reshape(nt*npix, ngeo),
        'z': interpolated_sensor.z.data.reshape(nt*npix, ngeo),
        'd': interpolated_sensor.deltas.data.reshape(nt*npix, ngeo)
    }
    return coordinates