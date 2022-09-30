import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ipywidgets import interact
from bhnerf.utils import normalize
import jax
from jax import numpy as jnp
import functools


def plot_evpa_ticks(Q, U, alpha, beta, ax=None, scale=None, color=None, pivot='mid', headaxislength=0, headlength=0, width=0.005):
    aolp = (np.arctan2(U, Q) / 2) 
    dolp = np.sqrt(Q**2 + U**2)
    if ax is None: fig, ax = plt.subplots(1, 1)
    ax.quiver(alpha, beta, dolp*np.sin(aolp), -dolp*np.cos(aolp), pivot='mid', 
              headaxislength=0, headlength=0, width=0.005, scale=scale, color=color)
    
def slider_frame_comparison(frames1, frames2, axis=0, scale='amp'):
    """
    Slider comparison of two 3D xr.DataArray along a chosen dimension.
    Parameters
    ----------
    frames1: xr.DataArray
        A 3D array with 'axis' dimension to compare along
    frames2:  xr.DataArray
        A 3D array with 'axis' dimension to compare along
    scale: 'amp' or 'log', default='amp'
        Compare absolute values or log of the fractional deviation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    plt.tight_layout()
    mean_images = [frames1.mean(axis=axis), frames2.mean(axis=axis),
                   (np.abs(frames1 - frames2)).mean(axis=axis)]
    cbars = []
    titles = [None]*3
    if scale == 'amp':
        titles[2] = 'Absolute difference'
    elif scale == 'log':
        titles[2] = 'Log relative difference'

    for ax, image in zip(axes, mean_images):
        im = ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbars.append(fig.colorbar(im, cax=cax))

    def imshow_frame(frame):
        image1 = np.take(frames1, frame, axis=axis)
        image2 = np.take(frames2, frame, axis=axis)

        if scale == 'amp':
            image3 = np.abs(np.take(frames1, frame, axis=axis) - np.take(frames2, frame, axis=axis))
        elif scale == 'log':
            image3 = np.log(np.abs(np.take(frames1, frame, axis=axis) / np.take(frames2, frame, axis=axis)))

        for ax, img, title, cbar in zip(axes, [image1, image2, image3], titles, cbars):
            ax.imshow(img, origin='lower')
            ax.set_title(title)
            cbar.mappable.set_clim([img.min(), img.max()])

    num_frames = min(frames1.shape[axis], frames2.shape[axis])
    plt.tight_layout()
    interact(imshow_frame, frame=(0, num_frames-1));
    
def plot_geodesic_3D(data_array, geos, method='interact', max_r=10, figsize=(5,5), init_alpha=0, 
                     init_beta=0, vmin=None, vmax=None, cbar_shrink=0.65, fps=10, horizon=True, wire_sphere_r=None):
    
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from mpl_toolkits.mplot3d import Axes3D
    import ipywidgets as widgets
    
    if ('alpha' in geos.dims) and ('beta' in geos.dims):
        def update(ialpha, ibeta, vmin, vmax):
            trajectory = geos.isel(alpha=ialpha, beta=ibeta)
            values = data_array.isel(alpha=ialpha, beta=ibeta)
            trajectory = trajectory.where(trajectory.r < 2*max_r)
            x, y, z = trajectory.x.data, trajectory.y.data, trajectory.z.data
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc.set_segments(segments)
            lc.set_array(values)
            if vmin is None:
                vmin = values.min()
            if vmax is None:
                vmax = values.max()
            lc.set_clim([vmin, vmax])
            ax.set_title('alpha={}, beta={}'.format(values.alpha, values.beta))
            return lc,
    elif ('pix' in geos.dims):
        def update(pix, vmin, vmax):
            trajectory = geos.isel(pix=pix)
            values = data_array.isel(pix=pix)
            trajectory = trajectory.where(trajectory.r < 2*max_r)
            x, y, z = trajectory.x.data, trajectory.y.data, trajectory.z.data
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc.set_segments(segments)
            lc.set_array(values)
            if vmin is None:
                vmin = values.min()
            if vmax is None:
                vmax = values.max()
            lc.set_clim([vmin, vmax])
            ax.set_title('alpha={}, beta={}'.format(values.alpha, values.beta))
            return lc,
    else: 
        raise AttributeError
        
    if method not in ['interact', 'static', 'animate']:
        raise AttributeError('undefined method: {}'.format(method))
    
    fig = plt.figure()
    ax = plt.subplot(projection='3d')
    ax.set_xlim([-max_r, max_r])
    ax.set_ylim([-max_r, max_r])
    ax.set_zlim([-max_r, max_r])
    
    # Plot the black hole event horizon (r_plus) 
    if horizon:
        r_plus = float(1 + np.sqrt(1 - geos.spin**2))
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(r_plus*x, r_plus*y, r_plus*z, linewidth=0.0, color='black')

    # Plot the ISCO as a wire-frame
    if wire_sphere_r is not None:
        ax.plot_wireframe(wire_sphere_r*x, wire_sphere_r*y, wire_sphere_r*z, rcount=10, ccount=10, linewidth=0.3)
    
    if ('alpha' in geos.dims) and ('beta' in geos.dims):
        trajectory = geos.isel(alpha=init_alpha, beta=init_beta)
        values = data_array.isel(alpha=init_alpha, beta=init_beta)
    elif ('pix' in geos.dims):
        trajectory = geos.isel(pix=0)
        values = data_array.isel(pix=0)
        
    trajectory = trajectory.where(trajectory.r < 2*max_r)
    x, y, z = trajectory.x.data, trajectory.y.data, trajectory.z.data

    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = Line3DCollection(segments, cmap='viridis')
    lc.set_array(values)
    lc.set_clim([vmin, vmax])
    lc.set_linewidth(2)

    line = ax.add_collection3d(lc)
    cb = fig.colorbar(line, ax=ax, shrink=cbar_shrink, location='left')

    output = fig
    if method == 'interact':
        if ('alpha' in geos.dims) and ('beta' in geos.dims):
            widgets.interact(update, ialpha=(0, geos.alpha.size-1), ibeta=(0, geos.beta.size-1),
                             vmin=widgets.fixed(vmin), vmax=widgets.fixed(vmax))
        elif ('pix' in geos.dims):
            widgets.interact(update, pix=(0, geos.pix.size-1), vmin=widgets.fixed(vmin), vmax=widgets.fixed(vmax))
    if method == 'animate':
        raise NotImplementedError
        # output = animation.FuncAnimation(fig, lambda pix: update(pix, vmin, vmax), 
        #                                 frames=geos.pix.size-1, interval=1e3 / fps)
    return output

        
def animate_synced(movie, measurements, axes, t_dim='t', vmin=None, vmax=None, cmap='RdBu_r', add_ticks=True,
                   add_colorbar=True, title=None, fps=10, output=None):

    def animate_both(i):
        return animate_frame(i), animate_plot(i)
        
    # Image animation function (called sequentially)
    def animate_frame(i):
        im.set_array(movie.isel({t_dim: i}))
        return im
    
    def animate_plot(i):
        line.set_xdata(measurements.isel({t_dim: i}))  # update the data.
        return line,
    
    fig = plt.gcf()
    num_frames, nx, ny = movie.sizes.values()
    image_dims = ['x', 'y']
    extent = [movie[image_dims[0]].min(), movie[image_dims[0]].max(),
              movie[image_dims[1]].min(), movie[image_dims[1]].max()]

    if add_ticks == False:
        axes[0].set_xticks([])
        axes[0].set_yticks([])
    axes[0].set_title(title)

    im =  axes[0].imshow(np.zeros((nx, ny)), extent=extent, origin='lower', cmap=cmap, aspect="equal")

    if add_colorbar:
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)
    
    vmin = vmin if vmin else movie.min().data
    vmax = vmax if vmax else movie.max().data
    im.set_clim(vmin, vmax)

    y = np.linspace(movie.y[0], movie.y[-1], measurements.pix.size)
    line, = axes[1].plot(measurements.isel(t=0), y)
    axes[1].set_title(title)
    
    asp = np.diff(axes[1].get_xlim())[0] / np.diff(axes[1].get_ylim())[0]
    axes[1].set_aspect(asp)
    
    plt.tight_layout()
    anim = animation.FuncAnimation(fig, animate_both, frames=num_frames, interval=1e3 / fps)

    if output is not None:
        anim.save(output, writer='imagemagick', fps=fps)
    return anim

def animate_movies_synced(movie_list, axes, t_dim='t', vmin=None, vmax=None, cmaps='afmhot', add_ticks=False,
                   add_colorbars=True, titles=None, fps=10, output=None, flipy=False):
    """
    Synchronous animation of multiple 3D xr.DataArray along a chosen dimension.

    Parameters
    ----------
    movie_list: list of xr.DataArrays
        A list of movies to animated synchroniously.
    axes: list of matplotlib axis,
        List of matplotlib axis object for the visualization. Should have same length as movie_list.
    t_dim: str, default='t'
        The dimension along which to animate frames
    vmin, vmax : float, optional
        vmin and vmax define the data range that the colormap covers.
        By default, the colormap covers the complete value range of the supplied data.
    cmaps : list of str or matplotlib.colors.Colormap, optional
        If this is a scalar then it is extended for all movies.
        The Colormap instance or registered colormap name used to map scalar data to colors.
        Defaults to :rc:`image.cmap`.
    add_ticks: bool, default=True
        If true then ticks will be visualized.
    add_colorbars: list of bool, default=True
        If this is a scalar then it is extended for all movies. If true then a colorbar will be visualized.
    titles: list of strings, optional
        List of titles for each animation. Should have same length as movie_list.
    fps: float, default=10,
        Frames per seconds.
    output: string,
        Path to save the animated gif. Should end with .gif.
    flipy: bool, default=False,
        Flip y-axis to match ehtim plotting function

    Returns
    -------
    anim: matplotlib.animation.FuncAnimation
        Animation object.
    """
    # Image animation function (called sequentially)
    def animate_frame(i):
        for movie, im in zip(movie_list, images):
            im.set_array(movie.isel({t_dim: i}))
        return images

    fig = plt.gcf()
    num_frames, nx, ny = movie_list[0].sizes.values()

    image_dims = list(movie_list[0].sizes.keys())
    image_dims.remove('t')
    extent = [movie_list[0][image_dims[0]].min(), movie_list[0][image_dims[0]].max(),
              movie_list[0][image_dims[1]].min(), movie_list[0][image_dims[1]].max()]

    # initialization function: plot the background of each frame
    images = []
    titles = [movie.name for movie in movie_list] if titles is None else titles
    cmaps = [cmaps]*len(movie_list) if isinstance(cmaps, str) else cmaps
    vmin_list = [movie.min() for movie in movie_list] if vmin is None else vmin
    vmax_list = [movie.max() for movie in movie_list] if vmax is None else vmax

    for movie, ax, title, cmap, vmin, vmax in zip(movie_list, axes, titles, cmaps, vmin_list, vmax_list):
        if add_ticks == False:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(np.zeros((nx, ny)), extent=extent, origin='lower', cmap=cmap)
        if add_colorbars:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax)
        im.set_clim(vmin, vmax)
        images.append(im)
        if flipy:
            ax.invert_yaxis()

    plt.tight_layout()
    anim = animation.FuncAnimation(fig, animate_frame, frames=num_frames, interval=1e3 / fps)

    if output is not None:
        anim.save(output, writer='imagemagick', fps=fps)
    return anim

@xr.register_dataarray_accessor("visualization")
class _VisualizationAccessor(object):
    """
    Register a custom accessor VisualizationAccessor on xarray.DataArray object.
    This adds methods for visualization of Gaussian Random Fields (3D DataArrays) along a single axis.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def slider(self, t_dim='t', ax=None, cmap=None):
        """
        Interactive slider visualization of a 3D xr.DataArray along specified dimension.

        Parameters
        ----------
        t_dim: str, default='t'
            The dimension along which to animate frames
        ax: matplotlib axis,
            A matplotlib axis object for the visualization.
        cmap : str or matplotlib.colors.Colormap, optional
            The Colormap instance or registered colormap name used to map scalar data to colors.
            Defaults to :rc:`image.cmap`.
        """
        movie = self._obj.squeeze()
        if movie.ndim != 3:
            raise AttributeError('Movie dimensions ({}) different than 3'.format(movie.ndim))

        num_frames = movie[t_dim].size
        image_dims = list(movie.dims)
        image_dims.remove(t_dim)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        extent = [movie[image_dims[0]].min(), movie[image_dims[0]].max(),
                  movie[image_dims[1]].min(), movie[image_dims[1]].max()]

        im = ax.imshow(movie.isel({t_dim: 0}), extent=extent, cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax)

        def imshow_frame(frame):
            img = movie.isel({t_dim: frame})
            im.set_array(movie.isel({t_dim: frame}))
            cbar.mappable.set_clim([img.min(), img.max()])

        interact(imshow_frame, frame=(0, num_frames-1));

    def animate(self, t_dim='t', ax=None, vmin=None, vmax=None, cmap='RdBu_r', add_ticks=True, add_colorbar=True,
                fps=10, output=None):
        """
        Animate a 3D xr.DataArray along a chosen dimension.

        Parameters
        ----------
        t_dim: str, default='t'
            The dimension along which to animate frames
        ax: matplotlib axis,
            A matplotlib axis object for the visualization.
        vmin, vmax : float, optional
            vmin and vmax define the data range that the colormap covers.
            By default, the colormap covers the complete value range of the supplied data.
        cmap : str or matplotlib.colors.Colormap, default='RdBu_r'
            The Colormap instance or registered colormap name used to map scalar data to colors.
            Defaults to :rc:`image.cmap`.
        add_ticks: bool, default=True
            If true then ticks will be visualized.
        add_colorbar: bool, default=True
            If true then a colorbar will be visualized
        fps: float, default=10,
            Frames per seconds.
        output: string,
            Path to save the animated gif. Should end with .gif.

        Returns
        -------
        anim: matplotlib.animation.FuncAnimation
            Animation object.
        """
        movie = self._obj.squeeze()
        if movie.ndim != 3:
            raise AttributeError('Movie dimensions ({}) different than 3'.format(movie.ndim))

        num_frames = movie[t_dim].size
        image_dims = list(movie.dims)
        image_dims.remove(t_dim)
        nx, ny = [movie.sizes[dim] for dim in image_dims]

        # Image animation function (called sequentially)
        def animate_frame(i):
            im.set_array(movie.isel({t_dim: i}))
            return [im]

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        extent = [movie[image_dims[0]].min(), movie[image_dims[0]].max(),
                  movie[image_dims[1]].min(), movie[image_dims[1]].max()]

        # Initialization function: plot the background of each frame
        im = ax.imshow(np.zeros((nx, ny)), extent=extent, origin='lower', cmap=cmap)
        if add_colorbar:
            fig.colorbar(im)
        if add_ticks == False:
            ax.set_xticks([])
            ax.set_yticks([])
        vmin = movie.min() if vmin is None else vmin
        vmax = movie.max() if vmax is None else vmax
        im.set_clim(vmin, vmax)
        anim = animation.FuncAnimation(fig, animate_frame, frames=num_frames, interval=1e3 / fps)

        if output is not None:
            anim.save(output, writer='imagemagick', fps=fps)
        return anim

class VolumeVisualizer(object):
    def __init__(self, width, height, samples):
        """
        A Volume visualization class
        
        Parameters
        ----------
        width: int
            camera horizontal resolution.
        height: int
            camera vertical resolution.
        samples: int
            Number of integration points along a ray.
        """
        self.width = width
        self.height = height
        self.samples = samples 
        self.focal = .5 * width / jnp.tan(.5 * 0.7)
        self._pts = None
        
    def set_view(self, radius, azimuth, zenith, up=np.array([0., 0., 1.])):
        """
        Set camera view geometry
        
        Parameters
        ----------
        radius: float,
            Distance from the origin
        azimuth: float, 
            Azimuth angle in radians
        zenith: float, 
            Zenith angle in radians
        up: array, default=[0,0,1]
            The up direction determines roll of the camera
        """
        camorigin = radius * np.array([np.cos(azimuth)*np.sin(zenith), 
                                       np.sin(azimuth)*np.sin(zenith), 
                                       np.cos(zenith)])
        self._viewmatrix = self.viewmatrix(camorigin, up, camorigin)
        rays_o, rays_d = self.generate_rays(
            self._viewmatrix, self.width, self.height, self.focal)
        self._pts = self.sample_along_rays(rays_o, rays_d, 15., 35., self.samples)
        self.x, self.y, self.z = self._pts[...,0], self._pts[...,1], self._pts[...,2]
        self.d = jnp.linalg.norm(jnp.concatenate([jnp.diff(self._pts, axis=2), 
                                                  jnp.zeros_like(self._pts[...,-1:,:])], 
                                                 axis=2), axis=-1)
    
    def render(self, emission, facewidth, jit=False, bh_radius=0.0, norm_const=1.0, linewidth=0.1, bh_albedo=[0,0,0], cmap='hot'):
        """
        Render an image of the 3D emission
        
        Parameters
        ----------
        emission: 3D array 
            3D array with emission values
        jit: bool, default=False,
            Just in time compilation. Set true for rendering multiple frames.
            First rendering will take more time due to compilation.
        bh_radius: float, default=0.0
            Radius at which to draw a black hole (for visualization). 
            If bh_radius=0 then no black hole is drawn.
        norm_const: float, default=1.0 
            normalize (divide) by this constant) 
        facewidth: float, default=10.0 
            width of the enclosing cube face
        linewidth: float, default=0.1
            width of the cube lines
        bh_albedo: list, default=[0,0,0]
            Albedo (rgb) of the black hole. default is completly black.
        cmap: str, default='hot'
            Colormap for visualization
        Returns
        -------
        rendering: array,
            Rendered image
        """
        if self._pts is None: 
            raise AttributeError('must set view before rendering')
        emission = emission / norm_const
        
        cm = plt.get_cmap('hot') 
        emission_cm = cm(emission)
        emission_cm = jnp.clip(emission_cm - 0.05, 0.0, 1.0)
        emission_cm = jnp.concatenate([emission_cm[..., :3], emission[..., None] / jnp.amax(emission)], axis=-1)

        if jit:
            emission_cube = draw_cube_jit(emission_cm, self._pts, facewidth, linewidth)
            if bh_radius > 0:
                emission_cube = draw_bh_jit(emission_cube, self._pts, bh_radius, bh_albedo)
        else:
            emission_cube = draw_cube(emission_cm, self._pts, facewidth, linewidth)
            if bh_radius > 0:
                emission_cube = draw_bh(emission_cube, self._pts, bh_radius, bh_albedo)
        rendering = alpha_composite(emission_cube, self.d, self._pts, bh_radius)
        return rendering
    
    def viewmatrix(self, lookdir, up, position):
        """Construct lookat view matrix."""
        vec2 = normalize(lookdir)
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def generate_rays(self, camtoworlds, width, height, focal):
        """Generating rays for all images."""
        x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
            np.arange(width, dtype=np.float32),  # X-Axis (columns)
            np.arange(height, dtype=np.float32),  # Y-Axis (rows)
            indexing='xy')
        camera_dirs = np.stack(
            [(x - width * 0.5 + 0.5) / focal,
             -(y - height * 0.5 + 0.5) / focal, -np.ones_like(x)],
            axis=-1)
        directions = ((camera_dirs[..., None, :] *
                       camtoworlds[None, None, :3, :3]).sum(axis=-1))
        origins = np.broadcast_to(camtoworlds[None, None, :3, -1],
                                  directions.shape)

        return origins, directions

    def sample_along_rays(self, rays_o, rays_d, near, far, num_samples):
        t_vals = jnp.linspace(near, far, num_samples)
        pts = rays_o[..., None, :] + t_vals[None, None, :, None] * rays_d[..., None, :]
        return pts
    
    @property
    def coords(self):
        coords = None if self._pts is None else jnp.moveaxis(self._pts, -1, 0)
        return coords

def alpha_composite(emission, dists, pts, bh_rad, inside_halfwidth=4.5):
    emission = np.clip(emission, 0., 1.)
    color = emission[..., :-1] * dists[0, ..., None]
    alpha = emission[..., -1:] 
    
    # mask for points inside wireframe
    inside = np.where(np.less(np.amax(np.abs(pts), axis=-1), inside_halfwidth), 
                      np.ones_like(pts[..., 0]),
                      np.zeros_like(pts[..., 0]))
    
    # masks for points outside black hole
    bh = np.where(np.greater(np.linalg.norm(pts, axis=-1), bh_rad),
                  np.ones_like(pts[..., 0]),
                  np.zeros_like(pts[..., 0]))
    
    combined_mask = np.logical_and(inside, bh)
    
    
    rendering = np.zeros_like(color[:, :, 0, :])
    acc = np.zeros_like(color[:, :, 0, 0])
    outside_acc = np.zeros_like(color[:, :, 0, 0])
    for i in range(alpha.shape[-2]):
        ind = alpha.shape[-2] - i - 1
        
        # if pixels inside cube and outside black hole, don't alpha composite
        rendering = rendering + combined_mask[..., ind, None] * color[..., ind, :]
        
        # else, alpha composite      
        outside_alpha = alpha[..., ind, :] * (1. - combined_mask[..., ind, None])
        rendering = rendering * (1. - outside_alpha) + color[..., ind, :] * outside_alpha 
        
        acc = alpha[..., ind, 0] + (1. - alpha[..., ind, 0]) * acc
        outside_acc = outside_alpha[..., 0] + (1. - outside_alpha[..., 0]) * outside_acc
        
    rendering += np.array([1., 1., 1.])[None, None, :] * (1. - acc[..., None])
    return rendering

@jax.jit
def draw_cube_jit(emission_cm, pts, facewidth, linewidth):
    
    linecolor = jnp.array([0.0, 0.0, 0.0, 1000.0])
    vertices = jnp.array([[-facewidth/2., -facewidth/2., -facewidth/2.],
                        [facewidth/2., -facewidth/2., -facewidth/2.],
                        [-facewidth/2., facewidth/2., -facewidth/2.],
                        [facewidth/2., facewidth/2., -facewidth/2.],
                        [-facewidth/2., -facewidth/2., facewidth/2.],
                        [facewidth/2., -facewidth/2., facewidth/2.],
                        [-facewidth/2., facewidth/2., facewidth/2.],
                        [facewidth/2., facewidth/2., facewidth/2.]])
    dirs = jnp.array([[-1., 0., 0.],
                      [1., 0., 0.],
                      [0., -1., 0.],
                      [0., 1., 0.],
                      [0., 0., -1.],
                      [0., 0., 1.]])

    for i in range(vertices.shape[0]):

        for j in range(dirs.shape[0]):
            # Draw line segments from each vertex
            line_seg_pts = vertices[i, None, :] + jnp.linspace(0.0, facewidth, 64)[:, None] * dirs[j, None, :]

            for k in range(line_seg_pts.shape[0]):
                dists = jnp.linalg.norm(pts - jnp.broadcast_to(line_seg_pts[k, None, None, None, :], pts.shape), axis=-1)
                emission_cm += linecolor[None, None, None, :] * jnp.exp(-1. * dists / linewidth ** 2)[..., None]

    out = jnp.where(jnp.greater(jnp.broadcast_to(jnp.amax(jnp.abs(pts), axis=-1, keepdims=True), emission_cm.shape), 
                                facewidth/2. + linewidth), jnp.zeros_like(emission_cm), emission_cm)
    return out

def draw_cube(emission_cm, pts, facewidth, linewidth):
    linecolor = jnp.array([0.0, 0.0, 0.0, 1000.0])
    vertices = jnp.array([[-facewidth/2., -facewidth/2., -facewidth/2.],
                        [facewidth/2., -facewidth/2., -facewidth/2.],
                        [-facewidth/2., facewidth/2., -facewidth/2.],
                        [facewidth/2., facewidth/2., -facewidth/2.],
                        [-facewidth/2., -facewidth/2., facewidth/2.],
                        [facewidth/2., -facewidth/2., facewidth/2.],
                        [-facewidth/2., facewidth/2., facewidth/2.],
                        [facewidth/2., facewidth/2., facewidth/2.]])
    dirs = jnp.array([[-1., 0., 0.],
                      [1., 0., 0.],
                      [0., -1., 0.],
                      [0., 1., 0.],
                      [0., 0., -1.],
                      [0., 0., 1.]])

    for i in range(vertices.shape[0]):

        for j in range(dirs.shape[0]):
            # Draw line segments from each vertex
            line_seg_pts = vertices[i, None, :] + jnp.linspace(0.0, facewidth, 64)[:, None] * dirs[j, None, :]

            for k in range(line_seg_pts.shape[0]):
                dists = jnp.linalg.norm(pts - jnp.broadcast_to(line_seg_pts[k, None, None, None, :], pts.shape), axis=-1)
                emission_cm += linecolor[None, None, None, :] * jnp.exp(-1. * dists / linewidth ** 2)[..., None]

    out = jnp.where(jnp.greater(jnp.broadcast_to(jnp.amax(jnp.abs(pts), axis=-1, keepdims=True), emission_cm.shape), 
                                facewidth/2. + linewidth), jnp.zeros_like(emission_cm), emission_cm)
    return out

@jax.jit
def draw_bh_jit(emission, pts, bh_radius, bh_albedo):
    bh_albedo = jnp.array(bh_albedo)[None, None, None, :]
    lightdir = jnp.array([-1., -1., 1.])
    lightdir /= jnp.linalg.norm(lightdir, axis=-1, keepdims=True)
    bh_color = jnp.sum(lightdir * pts, axis=-1)[..., None] * bh_albedo
    emission = jnp.where(jnp.less(jnp.linalg.norm(pts, axis=-1, keepdims=True), bh_radius),
                    jnp.concatenate([bh_color, jnp.ones_like(emission[..., 3:])], axis=-1), emission)
    return emission

def draw_bh(emission, pts, bh_radius, bh_albedo):
    bh_albedo = jnp.array(bh_albedo)[None, None, None, :]
    lightdir = jnp.array([-1., -1., 1.])
    lightdir /= jnp.linalg.norm(lightdir, axis=-1, keepdims=True)
    bh_color = jnp.sum(lightdir * pts, axis=-1)[..., None] * bh_albedo
    emission = jnp.where(jnp.less(jnp.linalg.norm(pts, axis=-1, keepdims=True), bh_radius),
                    jnp.concatenate([bh_color, jnp.ones_like(emission[..., 3:])], axis=-1), emission)
    return emission