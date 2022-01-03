import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import ehtim as eh
import ehtim.const_def as ehc
from jax import numpy as jnp
import jax.scipy.ndimage as jnd
import scipy.ndimage as nd
import utils 

def plot_uv_coverage(obs, ax=None, fontsize=14, s=None, cmap='rainbow', add_conjugate=True, xlim=(-9.5, 9.5), ylim=(-9.5, 9.5),
                     shift_inital_time=True, cbar=True, cmap_ticks=[0, 4, 8, 12], time_units='Hrs'):
    """
    Plot the uv coverage as a function of observation time.
    x axis: East-West frequency
    y axis: North-South frequency

    Parameters
    ----------
    obs: ehtim.Obsdata
        ehtim Observation object
    ax: matplotlib axis,
        A matplotlib axis object for the visualization.
    fontsize: float, default=14,
        x/y-axis label fontsize.
    s: float,
        Marker size of the scatter points
    cmap : str, default='rainbow'
        A registered colormap name used to map scalar data to colors.
    add_conjugate: bool, default=True,
        Plot the conjugate points on the uv plane.
    xlim, ylim: (xmin/ymin, xmax/ymax), default=(-9.5, 9.5)
        x-axis range in [Giga lambda] units
    shift_inital_time: bool,
        If True, observation time starts at t=0.0
    cmap_ticks: list,
        List of the temporal ticks on the colorbar
    time_units: str,
        Units for the colorbar
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    giga = 10**9
    u = np.concatenate([obsdata['u'] for obsdata in obs.tlist()]) / giga
    v = np.concatenate([obsdata['v'] for obsdata in obs.tlist()]) / giga
    t = np.concatenate([obsdata['time'] for obsdata in obs.tlist()])

    if shift_inital_time:
        t -= t.min()

    if add_conjugate:
        u = np.concatenate([u, -u])
        v = np.concatenate([v, -v])
        t = np.concatenate([t, t])

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    if time_units == 'mins':
        t *= 60.0
        
    sc = ax.scatter(u, v, c=t, cmap=plt.cm.get_cmap(cmap), s=s)
    ax.set_xlabel(r'East-West Freq $[G \lambda]$', fontsize=fontsize)
    ax.set_ylabel(r'North-South Freq $[G \lambda]$', fontsize=fontsize)
    ax.invert_xaxis()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    
    if cbar is True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3.5%', pad=0.2)
        cbar = fig.colorbar(sc, cax=cax, ticks=cmap_ticks)
        cbar.set_ticklabels(['{} {}'.format(tick, time_units) for tick in cbar.get_ticks()])
    plt.tight_layout()
    
def empty_eht_obs(array, nt, tint, tstart=4.0, tstop=15.5,
                  ra=17.761121055553343, dec=-29.00784305556, rf=226191789062.5, mjd=57850,
                  bw=1856000000.0, timetype='UTC', polrep='stokes'):
    """
    Generate an empty ehtim.Observation from an array configuration and time constraints

    Parameters
    ----------
    array: ehtim.Array
        ehtim ehtim Array object (e.g. from: ehtim.array.load_txt(array_path))
    nt: int,
        Number of temporal frames.
    tint: float,
        Scan integration time in seconds
    tstart: float, default=4.0
        Start time of the observation in hours
    tstop: float, default=15.5
        End time of the observation in hours
    ra: float, default=17.761121055553343,
        Source Right Ascension in fractional hours.
    dec: float, default=-29.00784305556,
        Source declination in fractional degrees.
    rf: float, default=226191789062.5,
        Reference frequency observing at corresponding to 1.3 mm wavelength
    mjd: int, default=57850,
        Modified julian date of observation
    bw: float, default=1856000000.0,
    timetype: string, default='UTC',
        How to interpret tstart and tstop; either 'GMST' or 'UTC'
    polrep: sting, default='stokes',
        Polarization representation, either 'stokes' or 'circ'

    Returns
    -------
    obs: ehtim.Obsdata
        ehtim Observation object
    """
    tadv = (tstop - tstart) * 3600.0/ nt
    obs = array.obsdata(ra=ra, dec=dec, rf=rf, bw=bw, tint=tint, tadv=tadv, tstart=tstart, tstop=tstop, mjd=mjd,
                        timetype=timetype, polrep=polrep)
    return obs

def extra_phase(u, v, psize, image_shape, use_jax=False):
    """
    Extra phase to match centroid convention.

    Parameters
    ----------
    u, v: np.arrays or xr.DataArray
        arrays of Fourier frequencies.
    psize: float,
        pixel size
    image_shape: tuple
        Image dimension sizes (without padding)

    Returns:
    ----------
    phase: xr.DataArray
        A 2D phase function
    """
    _np = jnp if use_jax else np
    phase = _np.exp(-1j * _np.pi * psize * ((1 + image_shape[0] % 2) * u + (1 + image_shape[1] % 2) * v))
    return phase

def trianglePulse2D(u, v, psize, use_jax=False):
    """
    Modification of eht-imaging trianglePulse_F for a DataArray of points

    Parameters
    ----------
    u, v: np.arrays or xr.DataArray
        arrays of Fourier frequencies.
    psize: float,
        pixel size

    Returns
    -------
    pulse: xr.DataArray
        A 2D triangle pulse function

    References:
    ----------
    https://github.com/achael/eht-imaging/blob/50e728c02ef81d1d9f23f8c99b424705e0077431/ehtim/observing/pulses.py#L91
    """
    _np = jnp if use_jax else np
    
    pulse_u = (4.0 / (psize ** 2 * (2 * _np.pi * u) ** 2)) * (_np.sin((psize * (2 * _np.pi * u)) / 2.0)) ** 2
    pulse_u = _np.where(_np.equal(2 * _np.pi * u, 0), _np.ones_like(u), pulse_u)
#     pulse_u[2 * _np.pi * u == 0] = 1.0

    pulse_v = (4.0 / (psize ** 2 * (2 * _np.pi * v) ** 2)) * (_np.sin((psize * (2 * _np.pi * v)) / 2.0)) ** 2
#     pulse_v[2 * _np.pi * v == 0] = 1.0
    pulse_v = _np.where(_np.equal(2 * _np.pi * v, 0), _np.ones_like(v), pulse_v)

    return pulse_u * pulse_v

def observe_nonoise(images, u, v, fov, use_jax=False):
    """
    Modification of ehtim's observe_same_nonoise method to support jax.

    Parameters
    ----------
    images: np.array, shape=(batch,...) , 
        Image array
    u, v: np.arrays,
        1D arrays of u,v frequencies.
    fov: float, 
        Field of view in [uas] 
    use_jax: bool, default=False
        Use Jax flag

    Returns
    -------
    visibilities: np.array or DeviceArray
        array of visibilities, shape = (num_visibilities) with dtype np.complex128.

    Notes
    -----
    1. fft padding is not supported
    2. Jax interpolation supports only order<=1 (differs to ehtim default order=3)

    References
    ----------
    https://github.com/achael/eht-imaging/blob/6b87cdc65bdefa4d9c4530ea6b69df9adc531c0c/ehtim/movie.py#L981
    https://github.com/achael/eht-imaging/blob/6b87cdc65bdefa4d9c4530ea6b69df9adc531c0c/ehtim/observing/obs_simulate.py#L182
    """
    if use_jax == True:
        _np = jnp
        _nd = jnd
        interp_order = 1
    else:
        _np = np
        _nd = nd
        interp_order = 3

    # Pad image according to pad factor (interpolation in Fourier space)
    if images.ndim == 2:
        images = images[None,...]
    nb, ny, nx = images.shape
    b = np.tile(np.array(range(nb))[...,None], u.shape[-1]) # batch indices
    
    # Compute visibilities
    rad_per_uas = 4.848136811094136e-12
    psize = fov / nx * rad_per_uas
    freqs = _np.fft.fftshift(_np.fft.fftfreq(n=nx, d=psize))
    fft = _np.fft.fftshift(_np.fft.fft2(_np.fft.ifftshift(images)))
    
    # Extra phase to match centroid convention
    pulsefac = trianglePulse2D(u, v, psize, use_jax=use_jax)
    phase = extra_phase(u, v, psize, images.shape, use_jax=use_jax)
    coords = _np.stack((v, u), axis=-1)

    fov_uv = _np.array((freqs[-1]-freqs[0], freqs[-1]-freqs[0]))
    fourier_image_coords = _np.concatenate((
        b[...,None], 
        utils.world_to_image_coords(coords, fov_uv, (nx, ny), use_jax=use_jax) + 0.5
    ), axis=-1)
    visibilities = _nd.map_coordinates(fft, fourier_image_coords.T, order=interp_order).T
    visibilities = visibilities * phase * pulsefac
    return visibilities

def observe_same(movie, obs, ttype='nfft', output_path='./caltable', thermal_noise=True, station_noise=False,
                 dterm_noise=False, sigmat=0.25, seed=False):
    """
    Generate an Obeservation object from the movie and add noise.

    Parameters
    ----------
    movie: ehtim.Movie or xr.DataArray
        Input movie. If the movie is an xr.DataArray object it is transformed into an ehtim.Movie first.
    obs: ehtim.Observation
        An (empty) Observation object
    output_path: str
        Output path for caltable
    thermal_noise: bool
        False for no thermal noise noise
    station_noise: bool,
        True for station based gain and phase errors
    dterm_noise: bool,
        True for dterm noise
    sigmat: float,
        Correlation time for random station based errors
    seed: int, default=False
        Seed for the random number generators, uses system time if False

    Returns
    -------
    obs: ehtim.Observation
        Observation object with visibilties of the input movie.
    """
    # these gains are approximated from the EHT 2017 data
    # the standard deviation of the absolute gain of each telescope from a gain of 1
    GAIN_OFFSET = {'ALMA': 0.15, 'APEX': 0.15, 'SMT': 0.15, 'LMT': 0.6, 'PV': 0.15, 'SMA': 0.15, 'JCMT': 0.15,
                   'SPT': 0.15, 'SR': 0.0}
    GAINP = {'ALMA': 0.05, 'APEX': 0.05, 'SMT': 0.05, 'LMT': 0.5, 'PV': 0.05, 'SMA': 0.05, 'JCMT': 0.05,
             'SPT': 0.15, 'SR': 0.0}

    stabilize_scan_phase = True  # If true then add a single phase error for each scan to act similar to adhoc phasing
    stabilize_scan_amp = True    # If true then add a single gain error at each scan
    jones = True                 # Jones matrix corruption & calibration
    inv_jones = True             # Invert the jones matrix
    frcal = True                 # True if you do not include effects of field rotation
    dcal = not dterm_noise       # True if you do not include the effects of leakage
    if dterm_noise:
        dterm_offset = 0.05      # Random offset of D terms is given at each site with this std away from 1
    else:
        dterm_offset = ehc.DTERMPDEF
    neggains = False
    if station_noise:
        ampcal = False          # If False, time-dependent gaussian errors are added to station gains
        phasecal = False        # If False, time-dependent station-based random phases are added
        rlgaincal = False       # If False, time-dependent gains are not equal for R and L pol
        gain_offset = GAIN_OFFSET
        gainp = GAINP
    else:
        ampcal = True
        phasecal = True
        rlgaincal = True
        gain_offset = ehc.GAINPDEF
        gainp = ehc.GAINPDEF
    movie.rf = obs.rf
    obs = movie.observe_same(obs, ttype=ttype, add_th_noise=thermal_noise, ampcal=ampcal, phasecal=phasecal,
                             stabilize_scan_phase=stabilize_scan_phase, stabilize_scan_amp=stabilize_scan_amp,
                             gain_offset=gain_offset, gainp=gainp, jones=jones, inv_jones=inv_jones,
                             dcal=dcal, frcal=frcal, rlgaincal=rlgaincal, neggains=neggains,
                             dterm_offset=dterm_offset, caltable_path=output_path, seed=seed, sigmat=sigmat)

    return obs


def padded_obs(obs, field, fill_value=np.nan):
    obslist = obs.tlist()
    max_num_uv = np.max([len(obsdata[field]) for obsdata in obslist])
    output = np.full((len(obslist), max_num_uv), fill_value, dtype=obslist[0][field].dtype)
    for i, obsdata in enumerate(obslist):
        output[i, :len(obsdata[field])] = obsdata[field]
    return output