import numpy as np
import xarray as xr
import functools

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
  
def linspace_3d(num, start=(-0.5, -0.5, -0.5), stop=(0.5, 0.5, 0.5), endpoint=(True, True, True), units='unitless'):
    """
    Return a 2D DataArray with coordinates spaced over a specified interval.

    Parameters
    ----------
    num: int or tuple
        Number of grid points in (x, y, z) dimensions. If num is a scalar the 3D coordinates are assumed to
        have the same number of points.
    start: (float, float, float)
        (x, y, z) starting grid point (included in the grid)
    stop: (float, float, float)
        (x, y, z) ending grid point (optionally included in the grid)
    endpoint: (bool, bool, bool)
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
    num = (num, num, num) if np.isscalar(num) else num
    x = np.linspace(start[0], stop[0], num[0], endpoint=endpoint[0])
    y = np.linspace(start[1], stop[1], num[1], endpoint=endpoint[1])
    z = np.linspace(start[2], stop[2], num[2], endpoint=endpoint[2])
    grid = xr.Dataset(coords={'x': x, 'y': y, 'z': z})
    grid.x.attrs.update(units=units)
    grid.y.attrs.update(units=units)
    grid.z.attrs.update(units=units)
    
    return grid.utils_polar.add_coords(image_dims=['x','y','z'])

def rotation_matrix(axis, angle):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis
    
    Parameters
    ----------
    axis: list or np.array, dim=3
        Axis of rotation
    angle: float or numpy array of floats,
        Angle of rotation in radians
        
    Returns
    -------
    rotation_matrix: np.array(shape=(3,3,...)),
        A rotation matrix. If angle is a numpy array additional dimensions are stacked at the end.
        
    References
    ----------
    [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
    [2] https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    axis = np.array(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2.0)
    b, c, d = np.stack([-ax * np.sin(angle / 2.0) for ax in axis])
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


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
            A data array with additional polar coordinates 'r'
        """
        if not(np.all([dim in self._obj.coords for dim in image_dims])):
            raise AttributeError('Grid coordinates have to contain image dimensions')
        coords = np.meshgrid(*self._obj.coords.values(), indexing='ij')
        r = np.sqrt(functools.reduce(lambda x, y: x**2 + y**2, coords))
        grid = self._obj.assign_coords(r=(image_dims, r))

        # Add units to 'r'
        units = [grid[dim].units for dim in image_dims]
        if not(np.all([units[0] == x for x in units])):
            raise AttributeError('different units for different axis not supported')
        grid['r'].attrs.update(units=units[0])
        return grid