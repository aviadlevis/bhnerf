import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ipywidgets import interact


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

@xr.register_dataarray_accessor("utils_visualization")
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
        divider = _make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax)

        def imshow_frame(frame):
            img = movie.isel({t_dim: frame})
            ax.imshow(img, origin='lower', extent=extent, cmap=cmap)
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
