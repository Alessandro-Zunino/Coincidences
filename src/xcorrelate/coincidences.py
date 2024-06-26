import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from collections.abc import Sequence
import torch
import gc


def load_streams(filename):
    d = np.fromfile(file=filename, dtype='>u8')
    d = d[:(d.shape[0] // 2) * 2]

    a = d[(d & 0x8000_0000_0000_0000) == 0x0000_0000_0000_0000]
    a = a.astype(dtype=np.int64)

    b = d[(d & 0x8000_0000_0000_0000) == 0x8000_0000_0000_0000] & 0x7FFF_FFFF_FFFF_FFFF
    b = b.astype(dtype=np.int64)

    a_noduplicated = np.unique(a)
    b_noduplicated = np.unique(b)

    return a_noduplicated, b_noduplicated


def filter_afterpulsing(stream, threshold=10):

    time_diff = np.diff(stream)
    idx = np.argwhere(time_diff > threshold)

    return stream[idx + 1].flatten()


def atime_2_trace(stream, dt, coarsening=100000, units='flux'):
    t0 = stream.copy() - stream[0]
    c = int(coarsening)

    if coarsening > 1:
        t0, w0 = np.unique(t0 // c, return_counts=True)
    else:
        w0 = np.ones_like(t0)

    ttrace = np.zeros(t0[-1] + 1)

    ttrace[t0] = w0

    time_step = dt * c

    time = np.arange(ttrace.size) * time_step

    if units == 'flux':
        ttrace = ttrace / time_step

    return time, ttrace


def plot_ttrace(time, ttrace, time_step, flux_range=None):
    if flux_range is None:
        flux_range = [np.min(ttrace), np.max(ttrace)]

    binsize = 1 / time_step

    bins = np.arange(flux_range[0], flux_range[1], binsize)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey='all')

    ax[0].plot(time, ttrace)
    ax[0].set_ylabel('Flux (Hz)')
    ax[0].set_xlabel('Time (s)')

    ax[0].ticklabel_format(style='sci', scilimits=(3, 3), axis='y')

    b = ax[1].hist(ttrace, orientation='horizontal', histtype='step', bins=bins)

    ax[1].hlines(b[1][b[0].argmax()], 0, b[0].max(), color='red')

    fig.tight_layout()

    return fig, ax


def g_2(stream_1, stream_2, dt, delay_range, coarsening=1, normalize=True, processor='gpu'):
    """
    Calculate correlation between two photon arrival time streams equally sampled in time.
    Inspired by Laurence, T. A., Fore, S., Huser, T. (2006). Fast, flexible algorithm for calculating photon
    correlations. Optics Letters , 31 (6), 829–831. https://doi.org/10.1364/OL.31.000829

    Parameters
    ----------
    stream_1 : np.ndarray()
        Vector with arrival times channel 1.
    stream_2 : np.ndarray()
        Vector with arrival times channel 2.
    dt : float
        Multiplication factor for the arrival times vectors [s].
    delay_range : int or np.ndarray()
        Maximum tau value for which to calculate the correlation.
        It can also be an array of int values, where to calculate the correlation.
    coarsening : int
        Binning factor to use for the correlation histogram.
    normalize : bool
        If true, the correlation is normalized.
    processor : str
        If 'gpu', the processing is parallelized on CUDA cores.

    Returns
    -------
    np.ndarray, np.ndarray
        tau and G values.

    """

    if processor == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    offset = np.min((stream_1[0], stream_2[0]))

    t_max = np.max((stream_1[-1], stream_2[-1])) - offset

    t0 = torch.from_numpy(stream_1.copy() - offset).to(device)
    t1 = torch.from_numpy(stream_2.copy() - offset).to(device)

    if coarsening > 1:
        t0, w0 = torch.unique(t0 // coarsening, return_counts=True)
        t1, w1 = torch.unique(t1 // coarsening, return_counts=True)
        t_range = delay_range // coarsening
    else:
        w0 = torch.ones_like(t0)
        w1 = torch.ones_like(t1)
        t_range = delay_range

    if isinstance(t_range, Sequence):
        t_bottom, t_top = t_range[0], t_range[1]
    else:
        t_bottom, t_top = -t_range, t_range

    tau = torch.arange(t_bottom, t_top + 1, dtype=torch.int64)

    # create array for g

    num = len(tau)
    g = torch.empty(num)

    # calculate g for each tau value

    for i in trange(num):
        t = tau[i]
        g[i] = atimes_2_corr_single(t0, t1, w0, w1, t, t_max, coarsening, normalize)

    lag_time = tau * dt * coarsening

    lag_time = lag_time.detach().cpu().numpy()
    g = g.detach().cpu().numpy()

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    return lag_time, g


def atimes_2_corr_single(t0, t1, w0, w1, tau, t_max, coarsening, normalize):
    """
    Calculate single correlation value between two photon streams with arrival
    times t0 and t1, weight vectors w0 and w1, and tau value tau

    Parameters
    ----------
    t0 : np.ndarray()
        Vector with arrival times channel 1.
        [multiples of minimum coarsed macrotime].
    t1 : np.ndarray()
        Vector with arrival times channel 2.
        [multiples of minimum coarsed macrotime].
    w0 : np.ndarray()
        Vector of weights for t0.
    w1 : np.ndarray()
        Vector of weights for t1.
    coarsening : int
        Binning factor to use for the correlation histogram.
    tau : np.array()
        tau value for which to calculate the correlation.
        [multiples of minimum coarsed macrotime].
    t_max : int
        Index of the last photon arrival time event.
    normalize : bool
        If true, the correlation is normalized.

    Returns
    -------
    g : float
        Single value g(tau).

    """

    # calculate time shifted vector
    t1 = t1 + tau

    # find intersection between t0 and t1
    tau_double, idx0, idx1 = torch_intersect1d(t0, t1, return_indices=True)

    # calculate autocorrelation value
    g = (w0[idx0] * w1[idx1]).sum()

    # normalize g
    if normalize is True:
        overlap_time = t_max - tau
        intensity_0 = torch.sum(w0[t0 >= tau])
        intensity_1 = torch.sum(w1[t1 <= t_max])
        g = g * overlap_time / intensity_0 / intensity_1 / coarsening

    return g


def torch_intersect1d(ar1, ar2, return_indices=False):
    aux = torch.cat((ar1, ar2))

    aux, aux_sort_indices = aux.sort()

    mask = (aux[1:] == aux[:-1])
    int1d = aux[:-1][mask]

    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - ar1.size(dim=0)

        return int1d, ar1_indices, ar2_indices
    else:
        return int1d
