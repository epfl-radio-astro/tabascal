import jax.numpy as jnp
from jax import Array
from functools import reduce


def supersample_domain_specs(
    xs: list[Array], ss_factors: list[int]
) -> tuple[list[int], list[float]]:
    """Calculate the properties (size and resolution) of an integer supersampled domain

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

    factors = jnp.atleast_1d(jnp.array(ss_factors)).astype(int)
    assert isinstance(xs, list | tuple)
    assert factors.ndim == 1
    assert jnp.all(jnp.array([x.ndim == 1 for x in xs]))
    assert len(xs) == len(factors)

    n_ss = [len(x) * factor for x, factor in zip(xs, factors)]
    # dx_ss = [jnp.diff(x[:2])[0] / factor for x, factor in zip(xs, factors)]
    dx_ss = [
        float(jnp.diff(x[:2])[0]) / factor if len(x) > 1 else 1 / factor
        for x, factor in zip(xs, factors)
    ]

    return n_ss, dx_ss


def supersample_domain(xs: list[Array], ss_factors: list[int]) -> list[Array]:
    """Calculate the supersampled domain given integer factors for each axis.

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

    x_ss = [x[0] + dx * jnp.arange(n) for x, dx, n in zip(xs, dx_ss, n_ss)]

    return x_ss


def supersample_domain_k(xs: list[Array], ss_factors: list[int]) -> list[Array]:
    """Calculate the supersampled k-domain given integer factors for each axis.

    Parameters
    ----------
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.
    ss_factors : list[int]
        The integer factors to supersample each axis.

    Returns
    -------
    list[Array]
        The supersampled k-domain arrays
    """

    n_ss, dx_ss = supersample_domain_specs(xs, ss_factors)

    k_ss = [jnp.fft.fftshift(jnp.fft.fftfreq(n, dx)) for n, dx in zip(n_ss, dx_ss)]

    return k_ss


def supersample(y: Array, ss_factors: list[int]) -> Array:
    """Supersample a signal by integer factors along each axis.

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
    pads = [2 * (n * (factor - 1) // 2,) for n, factor in zip(ns, ss_factors)]

    shift = jnp.fft.fftshift

    Y = jnp.fft.fftn(y, norm="forward")

    Y_ss = shift(jnp.pad(shift(Y), pads, constant_values=0))

    y_ss = jnp.fft.ifftn(Y_ss, norm="forward")

    return y_ss


def domain_k(xs: list[Array]) -> list[Array]:
    """Calculate the k-domain given a regularly sampled domain.

    Parameters
    ----------
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.

    Returns
    -------
    list[Array]
        The 1-D arrays of the k-domain.
    """

    ks = [jnp.fft.fftshift(jnp.fft.fftfreq(len(x), jnp.diff(x[:2])[0])) for x in xs]

    return ks


def pad_domain_specs(
    xs: list[Array], pad_factors: list[float]
) -> tuple[list[int], list[int], list[float]]:
    """Calculate the properties (size, pads, resolution) of a padded domain.

    Parameters
    ----------
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.
    pad_factors : list[float]
        The factors by whcih to increase the domain size in each axis.

    Returns
    -------
    tuple[list[int], list[int], list[float]]
        The properties of the padded domain (sizes, pads, resolutions).
    """

    factors = jnp.atleast_1d(jnp.array(pad_factors))
    assert jnp.all(factors > 1)
    assert len(xs) == len(factors)

    ns = [len(x) for x in xs]
    n_pads = [int(n * (factor - 1) / 2) for n, factor in zip(ns, factors)]
    dxs = [float(jnp.diff(x[:2])[0]) if len(x) > 1 else 1 for x in xs]

    return ns, n_pads, dxs


def pad_domain(xs: list[Array], pad_factors: list[float]) -> list[Array]:
    """Get the padded domain from the original and the padding factors.

    Parameters
    ----------
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.
    pad_factors : list[float]
        The factors by whcih to increase the domain size in each axis.

    Returns
    -------
    list[Array]
        The 1-D padded domain arrays.
    """

    ns, n_pads, dxs = pad_domain_specs(xs, pad_factors)

    x_padded = [
        x[0] + dx * jnp.arange(-n_pad, n + n_pad)
        for x, n, n_pad, dx in zip(xs, ns, n_pads, dxs)
    ]

    return x_padded


def pad_domain_k(xs: list[Array], pad_factors: list[float]) -> list[Array]:
    """Get the padded k-domain from the original and the padding factors.

    Parameters
    ----------
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.
    pad_factors : list[float]
        The factors by whcih to increase the domain size in each axis.

    Returns
    -------
    list[Array]
        The 1-D padded k-domain arrays.
    """

    ns, n_pads, dxs = pad_domain_specs(xs, pad_factors)

    k_padded = [
        jnp.fft.fftshift(jnp.fft.fftfreq(n + 2 * n_pad, dx))
        for n, n_pad, dx in zip(ns, n_pads, dxs)
    ]

    return k_padded


def pad(z: Array, pad_factors: list[float]) -> Array:
    """Pad a signal with a linear ramp to join the start and end of each axis.

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

    factors = jnp.atleast_1d(jnp.array(pad_factors))
    ndim = z.ndim
    assert len(factors) == ndim
    assert jnp.all(factors > 1)

    ns = z.shape

    pads = tuple([2 * (int(ns[i] * (factors[i] - 1) / 2),) for i in range(ndim)])

    edge_value = lambda i: 0.5 * (
        jnp.mean(jnp.take(z, 0, axis=i)) + jnp.mean(jnp.take(z, -1, axis=i))
    )

    end_values = []
    for i in range(z.ndim):
        edge = edge_value(i)
        if jnp.iscomplexobj(edge):
            end_values.append(complex(edge))
        else:
            end_values.append(float(edge))

    end_values = tuple(end_values)

    z_padded = jnp.pad(z, pad_width=pads, mode="linear_ramp", end_values=end_values)

    return z_padded


def pk_cut(pk: Array, cutoff: float) -> tuple[list[slice], list[tuple[int, int]]]:
    """Calculate the indexes and pads to cut and add back Fourier modes according to a power spectrum and a relative cutoff

    Parameters
    ----------
    pk : Array
        The power spectrum array with dimensionality equal to the domain.
    cutoff : float
        The relative cutoff for Fourier modes.

    Returns
    -------
    tuple[list[slice], list[tuple[int, int]]]
        The indexes and pads to remove and add the Fourier modes below the relative cutoff.
    """

    # Find where the power spectrum value is greater than the relative cutoff value
    cond = jnp.prod(
        jnp.stack(
            [pk > cutoff * pk.max(axis=i, keepdims=True) for i in range(pk.ndim)]
        ),
        axis=0,
    )
    idx = jnp.where(cond)

    # Turn the boolean index into index slices
    idxs = [slice(idx[i].min(), idx[i].max() + 1) for i in range(pk.ndim)]

    # Define the pads to get back to the original size defined by pk
    pads = [
        (int(idx[i].min()), int(pk.shape[i] - idx[i].max() - 1)) for i in range(pk.ndim)
    ]

    return idxs, pads


def fourier_cut(pk: Array, cutoff: float, y: Array) -> Array:
    """Remove the Fourier modes that do not meet a relative cutoff in th given power spectrum.

    Parameters
    ----------
    pk : Array
        The power spectrum of the same shape and the signal.
    cutoff : float
        The relative cutoff.
    y : Array
        The signal from which to cut Fourier modes.

    Returns
    -------
    Array
        The remaining Fourier modes after cutting.
    """

    # Get the index slices and pads for cutting Fourier modes below a relative cutoff
    idxs, pads = pk_cut(pk, cutoff)

    # Fourier transform of y that is shifted to have low k in the centre of the array i.e. -k_max, ..., 0, ..., k_max
    Y = jnp.fft.fftshift(jnp.fft.fftn(y, norm="forward"))

    # Slice the Fourier transform to only include the values where the pk meets the relative cutoff
    Y_cut = Y[tuple(idxs)]

    return Y_cut


def fourier_uncut(pk: Array, cutoff: float, Y_cut: Array) -> Array:
    """Zero pad an array of Fourier modes back to the original size.

    Parameters
    ----------
    pk : Array
        The power spectrum of the size of the original signal.
    cutoff : float
        The relative cutoff.
    Y_cut : Array
        Array of cut Fourier modes.

    Returns
    -------
    Array
        The zero padded Fourier array of the original size defined by pk.
    """

    # Get the index slices and pads for cutting Fourier modes below a relative cutoff
    idxs, pads = pk_cut(pk, cutoff)

    # Zero pad Fourier array back to original size
    Y_uncut = jnp.pad(Y_cut, pad_width=pads, constant_values=0.0)

    return Y_uncut


def supersample_fourier(Y: Array, factors: list[int]) -> Array:
    """Supersample a signal by integer factors, along each dimension, where the Fourier modes have been provided.

    Parameters
    ----------
    Y : Array
        The Fourier modes of the signal. These should be ordered as -k_max, ..., 0, ..., k_max.
    factors : list[int]
        The integer factors by which to supersample the signal in each dimension.

    Returns
    -------
    Array
        The supersampled signal.
    """

    # Calculate zero-padding widths to achieve an integer supersampling
    ns = Y.shape
    pads = tuple([2 * (n * (factor - 1) // 2,) for n, factor in zip(ns, factors)])

    # Zero pad Fourier array to supersampled size
    Y_ss = jnp.fft.fftshift(jnp.pad(Y, pads, constant_values=0))

    # iFFT back to normal space
    y_ss = jnp.fft.ifftn(Y_ss, norm="forward")

    return y_ss


def domain_ss(
    xs: list[Array], ss_factors: list[int], pad_factors: list[float]
) -> list[Array]:
    """Calculate the domain of a signal that has been padded, supersampled, and then cut to the supersampled original size.

    Parameters
    ----------
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.
    ss_factors : list[int]
        The integer factors to supersample each axis.
    pad_factors : list[float]
        The factors by whcih to increase the domain size in each axis.

    Returns
    -------
    list[Array]
        The domain values of the supersampled signal. Each array is in the range x0-dx/2, ..., xN+dx/2.
    """

    # Get the properties of the padded domain
    ns, n_pads, dxs = pad_domain_specs(xs, pad_factors)

    ns_padded = [n + 2 * n_pad for n, n_pad in zip(ns, n_pads)]

    # Calculate the slices to extract the unpadded, supersampled array
    idxs_pad_ss = [
        slice(n * f - int(f / 2), N * f - n * f - int(f / 2))
        for N, n, f in zip(ns_padded, n_pads, ss_factors)
    ]

    # Calculate the padded, supersampled domain
    xs_pad_ss = supersample_domain(pad_domain(xs, pad_factors), ss_factors)

    # Extract the purely supersampled domain
    xs_ss = [x[idx] for x, idx in zip(xs_pad_ss, idxs_pad_ss)]

    return xs_ss


def pow_spec(k: Array, p0: float, k0: float, gamma: float) -> Array:
    """1-D power spectrum of the form:
    .. math::
        P(k) = \\frac{1}{2} \\left( \\exp[-\\frac{k^2}{2k_0^2}] + (1 + \\frac{k^2}{2k_0^2})^\\gamma \\right)

    $P(k) = \\frac{1}{2} \\left( \\exp[-\\frac{k^2}{2k_0^2}] + (1 + \\frac{k^2}{2k_0^2})^\\gamma \\right)$

    Parameters
    ----------
    k : Array
        Array of k-modes.
    p0 : float
        Power of the k=0 mode.
    k0 : float
        Characteristic k-mode. Approximately the edge of the low-pass filter.
    gamma : float
        The steepness of the low-pass filter dropoff. Larger values lead to a steeper dropoff.

    Returns
    -------
    Array
        The power spectrum values evaluated at the provided k-modes.
    """

    # Calculate the power spectrum for a 1-D k-domian
    pk = 0.5 * p0 * (jnp.exp(-0.5 * (k / k0) ** 2) + (1 + (k / k0) ** 2) ** -gamma)

    return pk


def pow_spec_nd(
    ks: list[Array], p0: float, k0s: list[float], gammas: list[float]
) -> Array:
    """Calculate an N-D power spectrum where each dimension is independent.

    Parameters
    ----------
    ks : list[Array]
        List of N Arrays, each with the k-modes for that dimension.
    p0 : float
        Power of the k=0 mode.
    k0s : list[float]
        List of N floats, each giving the chracteristic k-mode forming the low-pass filter edge.
    gammas : list[float]
        List of N floats, each giving the steepness of the low-pass filter dropoff.

    Returns
    -------
    Array
        The power spectrum evaluated on the N-D grid formed by the outer product of the given k-mode arrays.
    """

    assert len(ks) == len(k0s)
    assert len(ks) == len(gammas)

    # Calculate power spectrum on each k-domain axis and create N-D power spectrum through an outer product
    pks = [pow_spec(k, 1.0, k0, gamma) for k, k0, gamma in zip(ks, k0s, gammas)]
    pk = p0 * reduce(jnp.outer, pks)

    return pk


def get_latent(
    y: Array,
    xs: list[Array],
    pad_factors: list[float],
    p0: float,
    k0s: list[float],
    gammas: list[float],
    cutoff: float,
) -> Array:
    """Calculate the un-normalised latent Fourier modes given a signal. The signal is padded according to pad_factors and then the Fourier modes are cut accoring to the power spectrum and reltive cutoff.

    Parameters
    ----------
    y : Array
        The signal over the original domain.
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.
    pad_factors : list[float]
        The factors by whcih to increase the domain size in each axis.
    p0 : float
        Power of the k=0 mode.
    k0s : list[float]
        List of N floats, each giving the chracteristic k-mode forming the low-pass filter edge.
    gammas : list[float]
        List of N floats, each giving the steepness of the low-pass filter dropoff.
    cutoff : float
        The relative cutoff for Fourier modes.

    Returns
    -------
    Array
        The un-normalised Fourier modes with power spectrum values above the relative cutoff.
    """

    # Pad the signal with a linear ramp between the start and the end
    y_pad = pad(y, pad_factors)

    # Get the k-values in the padded domain
    k_pad = pad_domain_k(xs, pad_factors)

    # Calculate the power spectrum over the k coordinate grid
    pk = pow_spec_nd(k_pad, p0, k0s, gammas)

    # Cut the Fourier modes where the power spectrum does not meet the relative cutoff
    Y_pad_cut = fourier_cut(pk, cutoff, y_pad)

    return Y_pad_cut


# def latent_predict(
#     Y_pad_cut: Array,
#     xs: list[Array],
#     ss_factors: list[int],
#     pad_factors: list[float],
#     p0: float,
#     k0s: list[float],
#     gammas: list[float],
#     cutoff: float,
# ) -> Array:

#     # Get the k-values in the padded domain
#     k_pad = pad_domain_k(xs, pad_factors)

#     # Calculate the power spectrum over the k coordinate grid
#     pk = pow_spec_nd(k_pad, p0, k0s, gammas)

#     # Cut the Fourier modes where the power spectrum does not meet the relative cutoff
#     Y_pad = fourier_uncut(pk, cutoff, Y_pad_cut)

#     # Calculate zero-padding widths to achieve an integer supersampling
#     ns = Y_pad.shape
#     ss_pads = [2 * (n * (factor - 1) // 2,) for n, factor in zip(ns, ss_factors)]

#     # Zero pad Fourier array to supersampled size
#     Y_pad_ss = jnp.pad(Y_pad, ss_pads, constant_values=0)

#     # iFFT back to normal space, signal is now padded and supersampled
#     y_pad_ss = jnp.fft.ifftn(jnp.fft.fftshift(Y_pad_ss), norm="forward")

#     # Get the properties of the padded domain
#     ns, n_pads, dxs = pad_domain_specs(xs, pad_factors)

#     # Calculate the slices to extract the unpadded, supersampled array
#     idxs_pad_ss = [
#         slice(n * f - int(f / 2), -n * f - int(f / 2))
#         for n, f in zip(n_pads, ss_factors)
#     ]

#     # Extract the purely supersampled signal
#     y_ss = y_pad_ss[*idxs_pad_ss]

#     return y_ss


def latent_init(
    xs: list[Array],
    pad_factors: list[float],
    ss_factors: list[int],
    p0: float,
    k0s: list[float],
    gammas: list[float],
    cutoff: float,
) -> tuple[Array, list[Array], list[tuple[int, int]], list[slice]]:
    """Calculate the required variance, pads and indexes to transform from the latent space of normalized Fourier modes to supersampled signal domain.

    Parameters
    ----------
    xs : list[Array]
        The regularly-spaced 1-D arrays of the original domain.
    pad_factors : list[float]
        The factors by whcih to increase the domain size in each axis.
    ss_factors : list[int]
        The integer factors to supersample each axis.
    p0 : float
        The power of the k=0 mode.
    k0s : list[float]
        The characteristic k-mode value. The edge of the low-pass filter generally.
    gammas : list[float]
        The steepness of the k-mode edge. The larger the value the steeper the drop.
    cutoff : float
        The relative cutoff for the power spectrum to include Fourier modes.

    Returns
    -------
    tuple[Array, list[Array], list[tuple[int, int]], list[slice]]
        The power spectrum evaluated at the remaining k-modes, the k-modes, the pads to return to the padded, supersampled size, and the indices to extract only the supersampled original portion.
    """

    ks_pad = pad_domain_k(xs, pad_factors)

    pk = pow_spec_nd(ks_pad, p0, k0s, gammas)

    idxs, cut_pads = pk_cut(pk, cutoff)

    latent_pk = pk[*idxs]
    latent_ks = [k_pad[idx] for k_pad, idx in zip(ks_pad, idxs)]

    ns = (len(k) for k in ks_pad)
    ss_pads = [2 * (n * (factor - 1) // 2,) for n, factor in zip(ns, ss_factors)]

    pads = [
        (ss_pad[0] + cut_pad[0], ss_pad[1] + cut_pad[1])
        for cut_pad, ss_pad in zip(cut_pads, ss_pads)
    ]

    ns, n_pads, dxs = pad_domain_specs(xs, pad_factors)

    # idxs_pad_ss = [
    #     slice(n * f - int(f / 2), -n * f - int(f / 2))
    #     for n, f in zip(n_pads, ss_factors)
    # ]

    ns_padded = [n + 2 * n_pad for n, n_pad in zip(ns, n_pads)]

    # Calculate the slices to extract the unpadded, supersampled array
    idxs_pad_ss = [
        slice(n * f - int(f / 2), N * f - n * f - int(f / 2))
        for N, n, f in zip(ns_padded, n_pads, ss_factors)
    ]

    return latent_pk, latent_ks, pads, idxs_pad_ss


# def latent_predict(Y_latent, og_pads, ss_pads, ss_idxs):

# # Zero pad Fourier array back to original size
# Y_pad = jnp.pad(Y_latent, pad_width=og_pads, constant_values=0.0)

# # Zero pad Fourier array by integer factor
# Y_pad_ss = jnp.pad(Y_pad, pad_width=ss_pads, constant_values=0.0)

# # iFFT to padded and supersampled normal space
# y_pad_ss = jnp.fft.ifftn(jnp.fft.fftshift(Y_pad_ss), norm="forward")

# # Slice borders to extract supersampled, unpadded array
# y_ss = y_pad_ss[*ss_idxs]

# return y_ss


def latent_predict(
    Y_latent: Array, pads: list[tuple[int, int]], ss_idxs: list[slice]
) -> Array:
    """Take a limited set of Fourier modes from regularly spaced k-domain and zero pad them to a padded, supersampled size. Then iFFT them to the signal space and cut the padding to a supersampled version of the signal over the intended domain.

    Parameters
    ----------
    Y_latent : Array
        The regularly spaced, limited set of Fourier modes. These should be ordered as -k_max, ..., 0, ..., k_max.
    pads : list[tuple[int, int]]
        The pads to get to the padded, supersampled size.
    ss_idxs : list[slice]
        The indexes the extract the purely supersampled signal.

    Returns
    -------
    Array
        The supersampled signal over the domain x0-dx/2, ..., xN+dx/2.
    """

    # Zero pad Fourier array to an integer multiple of original padded size
    Y_pad_ss = jnp.pad(Y_latent, pad_width=pads, constant_values=0.0)

    # iFFT to padded and supersampled normal space
    y_pad_ss = jnp.fft.ifftn(jnp.fft.fftshift(Y_pad_ss), norm="forward")

    # Slice borders to extract supersampled, unpadded array
    y_ss = y_pad_ss[*ss_idxs]

    return y_ss
