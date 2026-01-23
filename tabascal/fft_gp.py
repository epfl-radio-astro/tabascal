"""FFT-based Gaussian Process utilities."""

import functools
from functools import reduce
from typing import Any, Callable, List, Tuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from tabascal.timing import measure_runtime

# Type variable for generic function signatures
F = TypeVar("F", bound=Callable[..., Any])


def _get_dx(x: Array) -> Array:
    """Get the resolution of a regularly spaced 1-D array.

    Returns a JAX array for JIT compatibility.
    """
    if len(x) > 1:
        return jnp.diff(x[:2])[0]
    else:
        return jnp.array(1.0)


@measure_runtime
def supersample_domain_specs(
    xs: List[Array], ss_factors: List[int]
) -> Tuple[List[int], List[float]]:
    """
    Calculate properties (size and resolution) of an integer supersampled domain.

    Parameters
    ----------
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.
    ss_factors : list[int]
        The integer factors to supersample each axis.

    Returns
    -------
    tuple[list[int], list[float]]
        The sizes and resolutions of the supersampled domain.
    """
    factors = jnp.atleast_1d(jnp.asarray(ss_factors)).astype(int)

    # Validation
    if len(xs) != len(factors):
        raise ValueError(f"Length mismatch: len(xs)={len(xs)}, len(ss_factors)={len(factors)}")

    n_ss = [len(x) * f for x, f in zip(xs, factors)]
    dx_ss = [_get_dx(x) / f for x, f in zip(xs, factors)]

    return n_ss, dx_ss


@measure_runtime
def supersample_domain(xs: List[Array], ss_factors: List[int]) -> List[Array]:
    """
    Calculate the supersampled domain given integer factors for each axis.

    Parameters
    ----------
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.
    ss_factors : list[int]
        The integer factors to supersample each axis.

    Returns
    -------
    list[Array]
        The supersampled domain arrays.
    """
    n_ss, dx_ss = supersample_domain_specs(xs, ss_factors)
    return [x[0] + dx * jnp.arange(n) for x, dx, n in zip(xs, dx_ss, n_ss)]


@measure_runtime
def supersample_domain_k(xs: List[Array], ss_factors: List[int]) -> List[Array]:
    """
    Calculate the supersampled k-domain given integer factors for each axis.

    Parameters
    ----------
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.
    ss_factors : list[int]
        The integer factors to supersample each axis.

    Returns
    -------
    list[Array]
        The supersampled k-domain arrays.
    """
    n_ss, dx_ss = supersample_domain_specs(xs, ss_factors)
    return [jnp.fft.fftshift(jnp.fft.fftfreq(n, dx)) for n, dx in zip(n_ss, dx_ss)]


@measure_runtime
def supersample(y: Array, ss_factors: List[int]) -> Array:
    """
    Supersample a signal by integer factors along each axis via zero-padding in Fourier space.

    Parameters
    ----------
    y : Array
        The signal to supersample.
    ss_factors : list[int]
        The integer factors to supersample each axis.

    Returns
    -------
    Array
        The supersampled signal.
    """
    ns = y.shape
    pads = [(n * (f - 1) // 2, n * (f - 1) // 2) for n, f in zip(ns, ss_factors)]

    # Forward FFT (no normalization)
    Y = jnp.fft.fftn(y)

    # Pad in shifted Fourier space, then shift back
    Y_shifted = jnp.fft.fftshift(Y)
    Y_padded = jnp.pad(Y_shifted, pads, mode="constant", constant_values=0)
    Y_ss = jnp.fft.ifftshift(Y_padded)

    # Inverse FFT (default normalization scales by 1/n)
    # Scale by product of supersample factors to preserve signal amplitude
    scale = float(np.prod(ss_factors))
    return jnp.fft.ifftn(Y_ss) * scale


@measure_runtime
def domain_k(xs: List[Array]) -> List[Array]:
    """
    Calculate the k-domain given a regularly sampled domain.

    Parameters
    ----------
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.

    Returns
    -------
    list[Array]
        The 1-D arrays of the k-domain.
    """
    return [jnp.fft.fftshift(jnp.fft.fftfreq(len(x), _get_dx(x))) for x in xs]


@measure_runtime
def pad_domain_specs(
    xs: List[Array], pad_factors: List[float]
) -> Tuple[List[int], List[int], List[Array]]:
    """
    Calculate properties (size, pads, resolution) of a padded domain.

    Parameters
    ----------
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.
    pad_factors : list[float]
        The factors by which to increase the domain size in each axis.

    Returns
    -------
    tuple[list[int], list[int], list[Array]]
        The properties of the padded domain (sizes, pads, resolutions).
        Note: pads and dxs are kept as JAX arrays for JIT compatibility.
    """
    factors = jnp.atleast_1d(jnp.asarray(pad_factors))
    if not jnp.all(factors >= 1.0):
        raise ValueError("Pad factors must be >= 1.0")
    if len(xs) != len(factors):
        raise ValueError(f"Length mismatch: len(xs)={len(xs)}, len(pad_factors)={len(factors)}")

    ns = [len(x) for x in xs]
    # Keep as JAX array to avoid concretization in JIT
    n_pads = [jnp.floor(n * (f - 1) / 2).astype(int) for n, f in zip(ns, factors)]
    dxs = [_get_dx(x) for x in xs]

    return ns, n_pads, dxs


@measure_runtime
def pad_domain(xs: List[Array], pad_factors: List[float]) -> List[Array]:
    """
    Get the padded domain from the original and the padding factors.

    Parameters
    ----------
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.
    pad_factors : list[float]
        The factors by which to increase the domain size in each axis.

    Returns
    -------
    list[Array]
        The 1-D padded domain arrays.
    """
    ns, n_pads, dxs = pad_domain_specs(xs, pad_factors)
    return [
        x[0] + dx * jnp.arange(-n_pad, n + n_pad)
        for x, n, n_pad, dx in zip(xs, ns, n_pads, dxs)
    ]


@measure_runtime
def pad_domain_k(xs: List[Array], pad_factors: List[float]) -> List[Array]:
    """
    Get the padded k-domain from the original and the padding factors.

    Parameters
    ----------
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.
    pad_factors : list[float]
        The factors by which to increase the domain size in each axis.

    Returns
    -------
    list[Array]
        The 1-D padded k-domain arrays.
    """
    ns, n_pads, dxs = pad_domain_specs(xs, pad_factors)
    return [
        jnp.fft.fftshift(jnp.fft.fftfreq(n + 2 * n_pad, dx))
        for n, n_pad, dx in zip(ns, n_pads, dxs)
    ]


def pad(z: Array, pad_factors: List[float]) -> Array:
    """
    Pad a signal with a linear ramp to join the start and end of each axis.
    This implementation is tracer-safe and supports JIT-compiled contexts.

    Parameters
    ----------
    z : Array
        The signal to pad.
    pad_factors : list[float]
        The factors to pad the signal by in each axis.

    Returns
    -------
    Array
        The padded signal.
    """
    if len(pad_factors) != z.ndim:
        raise ValueError(
            f"Length mismatch: len(pad_factors)={len(pad_factors)}, z.ndim={z.ndim}"
        )

    ns = z.shape
    # Compute padding widths as integers using Python arithmetic to keep JIT-compatible
    pads = [(int(n * (f - 1) / 2), int(n * (f - 1) / 2)) for n, f in zip(ns, pad_factors)]

    # 1. Calculate the target boundary averages (tracers allowed here).
    # These define the destination values for the ramps on each axis.
    edge_targets = []
    for i in range(z.ndim):
        start_avg = jnp.mean(jnp.take(z, 0, axis=i))
        end_avg = jnp.mean(jnp.take(z, -1, axis=i))
        edge_targets.append(0.5 * (start_avg + end_avg))

    # 2. Extend the signal using 'edge' mode (which is tracer-safe).
    # This gives us a base for the padded region before we apply the ramp.
    res = jnp.pad(z, pad_width=pads, mode="edge")

    # 3. Apply the linear ramps dimension by dimension.
    # By using a static end_value (0.0) for the weight mask, we satisfy
    # jnp.pad's static requirements while keeping the data part dynamic.
    for i in range(z.ndim):
        n_low, n_high = pads[i]
        if n_low == 0 and n_high == 0:
            continue

        # Create a 1D ramp mask for this axis: 1.0 at original boundary, 0.0 at far edge.
        ones_1d = jnp.ones(ns[i])
        ramp_1d = jnp.pad(ones_1d, (n_low, n_high), mode="linear_ramp", end_values=0.0)

        # Reshape to broadcast ramp_1d across all dimensions of 'res'
        shape = [1] * z.ndim
        shape[i] = -1
        ramp_mask = ramp_1d.reshape(shape)

        # Linearly blend the current extension towards the target edge value
        res = res * ramp_mask + edge_targets[i] * (1.0 - ramp_mask)

    return res


@measure_runtime
def pk_cut(pk: Array, cutoff: float) -> Tuple[List[slice], List[Tuple[int, int]]]:
    """
    Calculate the indexes and pads to cut Fourier modes according to a power spectrum cutoff.

    Parameters
    ----------
    pk : Array
        The power spectrum array with dimensionality equal to the domain.
    cutoff : float
        The relative cutoff for Fourier modes.

    Returns
    -------
    tuple[list[slice], list[tuple[int, int]]]
        The indexes and pads to remove/add Fourier modes relative to the cutoff.
    """
    # Create a mask where ANY dimension's max power is above cutoff
    # This ensures we keep modes that are significant in at least one direction
    masks = [pk > cutoff * pk.max(axis=i, keepdims=True) for i in range(pk.ndim)]
    cond = reduce(jnp.logical_and, masks)

    idx = jnp.where(cond)

    # Calculate bounding box slices and padding to restore size
    idxs = []
    pads = []
    for i in range(pk.ndim):
        im_min = int(idx[i].min())
        im_max = int(idx[i].max())
        idxs.append(slice(im_min, im_max + 1))
        pads.append((im_min, pk.shape[i] - im_max - 1))

    return idxs, pads


@measure_runtime
def fourier_cut(pk: Array, cutoff: float, y: Array) -> Array:
    """
    Remove Fourier modes that do not meet a relative cutoff.

    Parameters
    ----------
    pk : Array
        The power spectrum (same shape as the signal).
    cutoff : float
        The relative cutoff.
    y : Array
        The signal from which to cut Fourier modes.

    Returns
    -------
    Array
        The remaining Fourier modes after cutting (shifted frequency ordering).
    """
    idxs, _ = pk_cut(pk, cutoff)
    Y = jnp.fft.fftshift(jnp.fft.fftn(y, norm="forward"))
    return Y[tuple(idxs)]


@measure_runtime
def fourier_uncut(pk: Array, cutoff: float, Y_cut: Array) -> Array:
    """
    Zero-pad an array of cut Fourier modes back to the original size.

    Parameters
    ----------
    pk : Array
        The power spectrum defining the original size.
    cutoff : float
        The relative cutoff.
    Y_cut : Array
        Array of cut Fourier modes.

    Returns
    -------
    Array
        The zero-padded Fourier array (shifted frequency ordering).
    """
    _, pads = pk_cut(pk, cutoff)
    return jnp.pad(Y_cut, pad_width=pads, mode="constant", constant_values=0.0)


@measure_runtime
def supersample_fourier(Y: Array, factors: List[int]) -> Array:
    """
    Supersample a signal via zero-padding of provided Fourier modes.

    Parameters
    ----------
    Y : Array
        Fourier modes ordered as -k_max, ..., 0, ..., k_max.
    factors : list[int]
        Integer factors by which to supersample each dimension.

    Returns
    -------
    Array
        The supersampled signal.
    """
    ns = Y.shape
    pads = [(n * (f - 1) // 2, n * (f - 1) // 2) for n, f in zip(ns, factors)]

    # Pad and inverse transform
    Y_ss = jnp.fft.ifftshift(jnp.pad(Y, pads, mode="constant", constant_values=0))
    return jnp.fft.ifftn(Y_ss, norm="forward")


@measure_runtime
def domain_ss(
    xs: List[Array], ss_factors: List[int], pad_factors: List[float]
) -> List[Array]:
    """
    Calculate the domain of a signal that has been padded, supersampled, and cropped.

    Parameters
    ----------
    xs : list[Array]
        Regularly-spaced 1-D arrays of the original domain.
    ss_factors : list[int]
        Integer factors to supersample each axis.
    pad_factors : list[float]
        Factors by which to increase the domain size in each axis.

    Returns
    -------
    list[Array]
        Domain values of the supersampled signal.
    """
    ns, n_pads, _ = pad_domain_specs(xs, pad_factors)
    ns_padded = [n + 2 * n_pad for n, n_pad in zip(ns, n_pads)]

    # Slices to extract the unpadded, supersampled portion
    idxs_pad_ss = [
        slice(n * f - (f // 2), N * f - n * f - (f // 2))
        for N, n, f in zip(ns_padded, n_pads, ss_factors)
    ]

    xs_pad_ss = supersample_domain(pad_domain(xs, pad_factors), ss_factors)
    return [x[idx] for x, idx in zip(xs_pad_ss, idxs_pad_ss)]


@measure_runtime
def pow_spec(k: Array, p0: float, k0: float, gamma: float) -> Array:
    """
    1-D power spectrum of the form:
    P(k) = 0.5 * p0 * (exp(-0.5 * (k/k0)^2) + (1 + (k/k0)^2)^-gamma)

    Parameters
    ----------
    k : Array
        Array of k-modes.
    p0 : float
        Power of the k=0 mode.
    k0 : float
        Characteristic k-mode (filter edge).
    gamma : float
        Steepness of the drop-off.

    Returns
    -------
    Array
        Evaluation of the power spectrum.
    """
    k_norm_sq = (k / k0) ** 2
    return 0.5 * p0 * (jnp.exp(-0.5 * k_norm_sq) + (1.0 + k_norm_sq) ** -gamma)


@measure_runtime
def pow_spec_nd(
    ks: List[Array], p0: float, k0s: List[float], gammas: List[float]
) -> Array:
    """
    Calculate an N-D power spectrum where each dimension is independent.

    Parameters
    ----------
    ks : list[Array]
        List of N Arrays with k-modes for each dimension.
    p0 : float
        Power of the k=0 mode.
    k0s : list[float]
        Characteristic k-modes for each dimension.
    gammas : list[float]
        Steepness factors for each dimension.

    Returns
    -------
    Array
        N-D power spectrum (outer product of 1-D spectra).
    """
    if not (len(ks) == len(k0s) == len(gammas)):
        raise ValueError("Inputs ks, k0s, and gammas must have the same length.")

    # Compute 1-D power spectra
    pks = [pow_spec(k, 1.0, k0, gamma) for k, k0, gamma in zip(ks, k0s, gammas)]

    # Create meshgrid and compute N-D outer product
    grids = jnp.meshgrid(*pks, indexing='ij')
    result = p0 * reduce(jnp.multiply, grids)

    return result


@measure_runtime
def get_latent_init(
    xs: List[Array],
    pad_factors: List[float],
    p0: float,
    k0s: List[float],
    gammas: List[float],
    cutoff: float,
) -> Tuple[List[slice], Array]:
    """
    Pre-compute slicing metadata for JIT-compatible latent extraction.

    This function should be called once during setup. The returned
    slicing indices can then be passed to get_latent_apply, which
    is JIT-compatible.

    Parameters
    ----------
    xs : list[Array]
        1-D arrays of original domain.
    pad_factors : list[float]
        Padding factors for each axis.
    p0 : float
        Power of the k=0 mode.
    k0s : list[float]
        Characteristic k-modes for each dimension.
    gammas : list[float]
        Steepness factors for each dimension.
    cutoff : float
        Relative Fourier mode cutoff.

    Returns
    -------
    tuple[list[slice], Array]
        (idxs, pk): Slicing indices and power spectrum for latent extraction.
    """
    k_pad = pad_domain_k(xs, pad_factors)
    pk = pow_spec_nd(k_pad, p0, k0s, gammas)
    idxs, _ = pk_cut(pk, cutoff)

    return idxs, pk


def get_latent_apply(
    y: Array,
    pad_factors: List[float],
    idxs: List[slice],
) -> Array:
    """
    Extract latent Fourier modes using pre-computed slicing indices.

    This function is JIT-compatible when used with indices from get_latent_init.

    Parameters
    ----------
    y : Array
        Signal over original domain.
    pad_factors : list[float]
        Padding factors for each axis.
    idxs : list[slice]
        Pre-computed slicing indices from get_latent_init.

    Returns
    -------
    Array
        Fourier modes above the relative cutoff.

    Example
    -------
    >>> # Setup (call once, not JIT-compatible)
    >>> idxs, pk = get_latent_init(xs, pad_factors, p0, k0s, gammas, cutoff)
    >>>
    >>> # Apply (can be JIT-compiled)
    >>> @jax.jit
    >>> def compute_latent(y):
    >>>     return get_latent_apply(y, pad_factors, idxs)
    >>>
    >>> latent = compute_latent(y)
    """
    y_pad = pad(y, pad_factors)
    Y = jnp.fft.fftshift(jnp.fft.fftn(y_pad, norm="forward"))
    return Y[tuple(idxs)]


@measure_runtime
def latent_init(
    xs: List[Array],
    pad_factors: List[float],
    ss_factors: List[int],
    p0: float,
    k0s: List[float],
    gammas: List[float],
    cutoff: float,
) -> Tuple[Array, List[Array], List[Tuple[int, int]], List[slice]]:
    """
    Calculate metadata needed to transform from latent space to supersampled signal.

    Parameters
    ----------
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.
    pad_factors : list[float]
        The factors by which to increase the domain size in each axis.
    ss_factors : list[int]
        The integer factors to supersample each axis.
    p0 : float
        Power of the k=0 mode.
    k0s : list[float]
        Characteristic k-modes for each dimension.
    gammas : list[float]
        The steepness of the low-pass filter dropoff for each dimension.
    cutoff : float
        The relative cutoff for Fourier modes.

    Returns
    -------
    tuple[Array, list[Array], list[tuple[int, int]], list[slice]]
        (latent_pk, latent_ks, pads, idxs_pad_ss):
        - latent_pk: Power spectrum evaluated at the remaining k-modes.
        - latent_ks: List of k-modes for each dimension after cutting.
        - pads: Padding widths to reach the padded, supersampled Fourier grid.
        - idxs_pad_ss: Slices to extract the supersampled portion from the padded result.
    """
    ks_pad = pad_domain_k(xs, pad_factors)
    pk = pow_spec_nd(ks_pad, p0, k0s, gammas)

    idxs, cut_pads = pk_cut(pk, cutoff)
    latent_pk = pk[tuple(idxs)]
    latent_ks = [k[idx] for k, idx in zip(ks_pad, idxs)]

    # Calculate padding for supersampling
    ns_pad = [len(k) for k in ks_pad]
    ss_pads = [(n * (f - 1) // 2, n * (f - 1) // 2) for n, f in zip(ns_pad, ss_factors)]

    # Combine cutoff padding and supersampling padding
    combined_pads = [
        (sp[0] + cp[0], sp[1] + cp[1]) for cp, sp in zip(cut_pads, ss_pads)
    ]

    # Calculate cropping slices for original domain
    ns, n_pads, _ = pad_domain_specs(xs, pad_factors)
    ns_padded = [n + 2 * npad for n, npad in zip(ns, n_pads)]
    idxs_pad_ss = [
        slice(npad * f - (f // 2), n_tot * f - npad * f - (f // 2))
        for n_tot, npad, f in zip(ns_padded, n_pads, ss_factors)
    ]

    return latent_pk, latent_ks, combined_pads, idxs_pad_ss


def latent_predict(
    Y_latent: Array, pads: List[Tuple[int, int]], ss_idxs: List[slice]
) -> Array:
    """
    Transform latent Fourier modes back to signal space over the supersampled domain.

    Parameters
    ----------
    Y_latent : Array
        Fourier modes (-k_max to k_max ordering).
    pads : list[tuple[int, int]]
        Pads to reach the padded, supersampled size.
    ss_idxs : list[slice]
        Slices to extract the supersampled signal.
    """
    # Pad in shifted space
    Y_padded = jnp.pad(Y_latent, pad_width=pads, mode="constant", constant_values=0.0)
    
    # Shift back and IFFT
    Y_ss = jnp.fft.ifftshift(Y_padded)
    y_padded_ss = jnp.fft.ifftn(Y_ss, norm="forward")

    # Crop to requested domain
    return y_padded_ss[tuple(ss_idxs)]
