import numpy as np
import xarray as xr

def linspace_2d(num, start=(-0.5, -0.5), stop=(0.5, 0.5), endpoint=(True, True), units='unitless'):
    """
    Return a 2D DataArray with coordinates spaced over a specified interval.

    Parameters
    ----------
    num: int or tuple
        Number of grid points in (y, x) dimensions. If num is a scalar the 2D coordinates are assumed to
        have the same number of points.
    start: (float, float)
        (y, x) starting grid point (included in the grid)
    stop: (float, float)
        (y, x) ending grid point (optionally included in the grid)
    endpoint: (bool, bool)
        Optionally include the stop points in the grid.
    units: str, default='unitless'
        Store the units of the underlying grid.

    Returns
    -------
    grid: xr.DataArray
        A DataArray with coordinates linearly spaced over the desired interval

    Notes
    -----
    Also computes image polar coordinates (r, theta).
    """
    num = (num, num) if np.isscalar(num) else num
    y = np.linspace(start[0], stop[0], num[0], endpoint=endpoint[0])
    x = np.linspace(start[1], stop[1], num[1], endpoint=endpoint[1])
    grid = xr.Dataset(coords={'y': y, 'x': x})
    grid.y.attrs.update(units=units)
    grid.x.attrs.update(units=units)
    return grid.utils_polar.add_coords()
  

@xr.register_dataset_accessor("utils_polar")
@xr.register_dataarray_accessor("utils_polar")
class _PolarAccessor(object):
    """
    Register a custom accessor PolarAccessor on xarray.DataArray and xarray.Dataset objects.
    This adds methods for polar coordinate processing on a 2D (x,y) grid.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def add_coords(self, image_dims=['y', 'x']):
        """
        Add polar coordinates to image_dimensions

        Parameters
        ----------
        image_dims: [dim1, dim2], default=['y', 'x'],
            Image data dimensions.

        Returns
        -------
        grid: xr.DataArray,
            A data array with additional polar coordinates 'r' and 'theta'
        """
        if not(image_dims[0] in self._obj.coords and image_dims[1] in self._obj.coords):
            raise AttributeError('Coordinates have to contain both x and y')
        y, x = self._obj[image_dims[0]], self._obj[image_dims[1]]
        yy, xx = np.meshgrid(y, x, indexing='ij')
        r = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        grid = self._obj.assign_coords({'r': ([image_dims[0], image_dims[1]], r),
                                        'theta': ([image_dims[0], image_dims[1]], theta)})

        # Add units to 'r' and 'theta'
        units = None
        if ('units' in grid[image_dims[0]].attrs) and ('units' in grid[image_dims[1]].attrs):
            units = 'units'
        elif ('inverse_units' in grid[image_dims[0]].attrs) and ('inverse_units' in grid[image_dims[1]].attrs):
            units = 'inverse_units'

        if units is not None:
            if grid[image_dims[0]].attrs[units] != grid[image_dims[1]].attrs[units]:
                raise AttributeError('different units for x and y not supported')
            grid['r'].attrs.update({units: grid[image_dims[0]].attrs[units]})
        grid['theta'].attrs.update(units='rad')
        return grid