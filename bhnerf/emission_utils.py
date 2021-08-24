import utils
import numpy as np
import xarray as xr
import skimage.transform

def gaussian_2d(ny, nx, std, fwhm=None, fov=(1.0, 'unitless'), center=(0,0), std_clip=np.inf):
    """
    Gaussian image.

    Parameters
    ----------
    ny, nx: int,
            Number of (y/x)-axis grid points.
    std: float, 
        Gaussian standard deviation. Used if fwhm is not specified.
    fwhm: float, optional,
        Gaussian full width half max. Overrides the std parameter.
    fov: (float, str), default=(1.0, 'unitless')
        Field of view and units. Default is unitless 1.0.
    total_flux: float, default=1.0,
        Total flux normalization of the image (sum over pixels).
    center: tuple, default=(0,0)
        Center of the gaussian in the image coordinates ('y', 'x')
    std_clip: float, default=np.inf
        Clip after this number of standard deviations

    Returns
    -------
    image: xr.DataArray,
        An image DataArray with dimensions ['y', 'x'].
    """
    if fwhm is None:
        fwhm = 2 * np.sqrt(2 * np.log(2)) * std
    else:
        std = fwhm / (2 * np.sqrt(2 * np.log(2)))

    grid = utils.linspace_2d((ny, nx), (-fov[0] / 2.0, -fov[0] / 2.0), (fov[0] / 2.0, fov[0] / 2.0), units=fov[1])

    data = np.exp(-0.5*( (grid.y - center[0])**2 + (grid.x - center[1])**2 ) / std**2)
    threshold = np.exp(-0.5 * std_clip ** 2)
    image = xr.DataArray(
        name='image',
        data=np.array(data.where(data > threshold).fillna(0.0), dtype=np.float64, order='C'),
        coords=grid.coords,
        dims=['y', 'x'],
        attrs={
            'envelope_model': 'gaussian',
            'fov': fov,
            'std': std,
            'fwhm': fwhm,
            'center': center,
            'std_clip': std_clip
        })
    return image
  
def generate_hotspots_2d(ny, nx, theta, orbit_radius, std, r_isco=3, fov=(10.0, 'GM/c^2'), std_clip=3, normalize=True):
    
    if np.any(orbit_radius < r_isco):
        raise AttributeError('hotspot center ({}) is is within r_isco: {}'.format(orbit_radius, r_isco))
        
    thetas = np.atleast_1d(theta)
    orbit_radii = np.atleast_1d(orbit_radius)
    stds = np.atleast_1d(std)
    initial_frame = np.zeros([ny, nx])
    
    for theta, orbit_radius, std in zip(thetas, orbit_radii, stds):
        x, y = orbit_radius * np.array([np.cos(theta), np.sin(theta)])
        hotspot = gaussian_2d(ny, nx, std, fov=fov, center=(y, x), std_clip=std_clip)
        initial_frame += hotspot
    if normalize:
        initial_frame /= initial_frame.max()
        
    return initial_frame

def world_to_image_coords(coords, fov, npix):
    for i in range(coords.shape[-1]):
        coords[...,i] = (coords[...,i] + fov[i]/2.0) / fov[i] * (npix[i] - 1)
    return coords
    
def generate_orbit_2d(initial_frame, nt, velocity_field):
    movie = []
    fovx = (initial_frame.x.max() - initial_frame.x.min()).data
    fovy = (initial_frame.y.max() - initial_frame.y.min()).data
    npix = (initial_frame.x.size, initial_frame.y.size)
    x, y = np.meshgrid(initial_frame.x, initial_frame.y) 
    for t in np.linspace(0.0, 1.0, nt): 
        warped_coords = velocity_warp_2d(x, y, t, velocity_field)
        image_coords = world_to_image_coords(warped_coords, fov=(fovx, fovy), npix=npix).T
        frame_data = skimage.transform.warp(initial_frame, image_coords, mode='constant', cval=0.0)
        frame = xr.DataArray(frame_data, dims=['y', 'x'], coords={'t': t, 'y': initial_frame.y, 'x':initial_frame.x})
        
        movie.append(frame.expand_dims('t'))
    movie = xr.concat(movie, dim='t')
    return movie

def velocity_warp_2d(x, y, t, velocity_field, jax=False):

    if jax: 
        import jax.numpy as _np
    else: 
        _np = np
    
    radius = _np.sqrt(x**2 + y**2) 
    theta = _np.arctan2(y, x)
    
    theta_rot = theta - 2 * np.pi * t * velocity_field

    x_rot = radius * _np.cos(theta_rot)
    y_rot = radius * _np.sin(theta_rot)
    warped_coords = _np.stack([x_rot, y_rot], axis=-1)
    return warped_coords

def integrate_rays(medium, sensor, dim='geo'):
    sensor = sensor.fillna(-1e9)
    coords = {'x': sensor.x, 'y': sensor.y}
    if 'z' in medium.dims:
        coords['z'] = sensor.z
    inteporlated_values = medium.interp(coords)
    pixels = (inteporlated_values.fillna(0.0) * sensor.deltas).sum(dim)
    return pixels