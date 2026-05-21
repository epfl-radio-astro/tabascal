"""Sky-source resolution for the astronomical sky-signal components.

A *sky source* is a representation-agnostic description of sky brightness. It is
resolved from a config spec (``{"type": ...}``) into a :class:`SkySource` that
can be asked for the representation a consumer needs:

- :meth:`SkySource.catalogue` -> ``(radec, flux)`` point list,
- :meth:`SkySource.image`     -> a per-channel image on the config grid,
- :meth:`SkySource.visibilities` -> model visibilities.

Each source implements the conversions it can; an unsupported one raises a clear
error (e.g. a FITS image cannot be turned back into a catalogue without source
finding). The same source spec can therefore seed a component in any role
(fixed sky, learnable init, or learnable prior mean) — the consumer picks the
representation and inverts it into its own parameter space.

Reserved polarisation convention (Stokes I only for now; shapes are documented
so consumers can be extended without a schema break):

- image     ``(n_freq, n_pol, n_l, n_m)``     -- this module uses ``(n_freq, n_l, n_m)`` with ``n_pol == 1``
- catalogue flux ``(n_src, n_pol, n_freq)``   -- this module uses ``(n_src, n_freq)`` with ``n_pol == 1``

Source specs may carry an optional ``stokes`` list (default ``["I"]``); anything
other than Stokes I is rejected until full polarisation lands.
"""

from abc import ABC
import warnings

import numpy as np
import jax.numpy as jnp
import xarray as xr
from jax_nufft import vis2dirty


# A direct-DFT point sky scales as O(n_src * n_bl * n_time * n_freq); above this
# many sources the wgridder image path is far cheaper (and bounded in memory).
DFT_SOURCE_WARN_THRESHOLD = 2000


def warn_if_large_catalogue(n_src: int, where: str) -> None:
    """Warn when a catalogue large enough to make the direct DFT expensive is
    routed to the point-source (DFT) visibility path."""
    if n_src > DFT_SOURCE_WARN_THRESHOLD:
        warnings.warn(
            f"{where}: {n_src} sources routed to the direct-DFT point path; this "
            f"scales as O(n_src*n_bl*n_time*n_freq). For large catalogues prefer "
            f"an image-based sky (rasterise + wgridder).",
            UserWarning,
        )


def _check_stokes(spec: dict) -> None:
    stokes = [str(s).upper() for s in spec.get("stokes", ["I"])]
    if stokes != ["I"]:
        raise NotImplementedError(
            f"Only Stokes I is supported; got stokes={spec.get('stokes')}."
        )


def _interp_to_freqs(src_freqs, values, dst_freqs, axis):
    """Interpolate ``values`` along ``axis`` from ``src_freqs`` onto ``dst_freqs``.

    A single source channel is broadcast (flat spectrum); otherwise 1-D linear
    interpolation per element (the spectrum is smooth, so this is adequate)."""
    src_freqs = np.asarray(src_freqs, dtype=float)
    dst_freqs = np.asarray(dst_freqs, dtype=float)
    if src_freqs.size == 1:
        return np.repeat(values, dst_freqs.size, axis=axis)
    order = np.argsort(src_freqs)
    src_sorted = src_freqs[order]
    vals_sorted = np.take(values, order, axis=axis)
    return np.apply_along_axis(
        lambda y: np.interp(dst_freqs, src_sorted, y), axis, vals_sorted
    )


def _radec_to_lm(ra, dec, ra0, dec0):
    """Direction cosines of (ra, dec) about phase centre (ra0, dec0), radians."""
    dra = ra - ra0
    l = np.cos(dec) * np.sin(dra)
    m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(dra)
    return l, m


def _rasterise(radec, flux, grid, ra0, dec0, n_freq):
    """Nearest-pixel rasterisation of a point catalogue onto the cosine grid.

    ``l_i = (i - n_pix/2) * pixsize``; sources outside the grid are dropped."""
    radec = np.asarray(radec)
    flux = np.asarray(flux)
    l, m = _radec_to_lm(radec[:, 0], radec[:, 1], ra0, dec0)

    n_pix = grid.n_pix
    pixsize = float(grid.pixsize)
    i_l = np.rint(l / pixsize + n_pix / 2).astype(int)
    i_m = np.rint(m / pixsize + n_pix / 2).astype(int)
    inside = (i_l >= 0) & (i_l < n_pix) & (i_m >= 0) & (i_m < n_pix)

    image = np.zeros((n_freq, n_pix, n_pix))
    for k in np.nonzero(inside)[0]:
        image[:, i_l[k], i_m[k]] += flux[k]
    return jnp.asarray(image)


class SkySource(ABC):
    """A resolved sky source. Subclasses implement the conversions they support."""

    def __init__(self, config):
        self.n_freq = int(config.n_freq)
        self.freqs = np.asarray(config.freqs, dtype=float)
        self.ra0 = float(np.deg2rad(config.phase_centre["ra"]))
        self.dec0 = float(np.deg2rad(config.phase_centre["dec"]))

    def catalogue(self):
        """Return ``(radec[n_src, 2] radians, flux[n_src, n_freq] Jy)``."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot be expressed as a point catalogue "
            "(source finding is not supported)."
        )

    def image(self, grid):
        """Return a per-channel Stokes-I image ``(n_freq, n_l, n_m)`` Jy/pixel."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot be rendered to an image."
        )

    def visibilities(self):
        """Return model visibilities ``(n_bl, n_freq, n_time)``."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot provide visibilities."
        )


class ZerosSource(SkySource):
    """An empty sky: a zero image / empty catalogue. Useful as a neutral prior
    mean or an explicit 'no sky'."""

    def catalogue(self):
        return jnp.zeros((0, 2)), jnp.zeros((0, self.n_freq))

    def image(self, grid):
        return jnp.zeros((self.n_freq, grid.n_pix, grid.n_pix))


class CatalogueSource(SkySource):
    """A point-source catalogue (tabsim zarr or WSClean/DP3 BBS text)."""

    def __init__(self, config, path=None, fmt="zarr"):
        super().__init__(config)
        if path is None:
            path = config.args.get("data", {}).get("zarr_path")
        if path is None:
            raise ValueError(
                "from_catalogue requires a 'path' (or a data.zarr_path fallback)."
            )
        self.path = str(path)
        if fmt == "zarr":
            self._radec, self._flux = self._read_zarr(self.path)
        elif fmt == "bbs":
            self._radec, self._flux = self._read_bbs(self.path)
        else:
            raise ValueError(
                f"Unsupported catalogue fmt '{fmt}'. Choose from (zarr, bbs)."
            )
        self.n_src = int(self._radec.shape[0])

    def _read_zarr(self, path):
        xds = xr.open_zarr(path)
        if "ast_p_radec" not in xds:
            raise ValueError(
                f"No point sources found in zarr at {path}. Expected 'ast_p_radec'."
            )
        radec = jnp.deg2rad(jnp.asarray(xds.ast_p_radec.values))   # (n_src, 2)
        flux = np.asarray(xds.ast_p_I.values).mean(axis=1)         # (n_src, n_chan)
        if "freq" in xds.coords or "freq" in xds.variables:
            flux = _interp_to_freqs(np.asarray(xds.freq.values), flux,
                                    self.freqs, axis=1)
        elif flux.shape[1] != self.n_freq:
            raise ValueError(
                f"Catalogue at {path} has {flux.shape[1]} channels and no 'freq' "
                f"coordinate to interpolate onto the {self.n_freq} model channels."
            )
        return radec, jnp.asarray(flux)

    def _read_bbs(self, path):
        radec_deg, flux = _parse_bbs(path, self.freqs)
        return jnp.deg2rad(jnp.asarray(radec_deg)), jnp.asarray(flux)

    def catalogue(self):
        return self._radec, self._flux

    def image(self, grid):
        return _rasterise(np.asarray(self._radec), np.asarray(self._flux),
                          grid, self.ra0, self.dec0, self.n_freq)


class FitsSource(SkySource):
    """A FITS image (2-D continuum or spectral cube) on the config grid."""

    def __init__(self, config, path, unit=None, hdu=0):
        super().__init__(config)
        self.path = str(path)
        self.unit = unit
        self.hdu = hdu

    def image(self, grid):
        from astropy.io import fits

        with fits.open(self.path) as hdul:
            data = np.squeeze(np.asarray(hdul[self.hdu].data, dtype=float))
            header = hdul[self.hdu].header

        n_pix = grid.n_pix
        data = data * self._jy_beam_to_jy_pixel_factor(header)

        if data.ndim == 2:
            if data.shape != (n_pix, n_pix):
                raise ValueError(
                    f"FITS spatial shape {data.shape} does not match grid "
                    f"({n_pix}, {n_pix}); reprojection is not supported."
                )
            return jnp.asarray(
                np.broadcast_to(data[None], (self.n_freq, n_pix, n_pix)).copy()
            )

        if data.ndim == 3:
            n_chan = data.shape[0]
            if data.shape[1:] != (n_pix, n_pix):
                raise ValueError(
                    f"FITS spatial shape {data.shape[1:]} does not match grid "
                    f"({n_pix}, {n_pix}); reprojection is not supported."
                )
            if n_chan == self.n_freq:
                return jnp.asarray(data)
            fits_freqs = self._fits_freqs(header, n_chan)
            return jnp.asarray(_interp_to_freqs(fits_freqs, data, self.freqs, axis=0))

        raise ValueError(f"Unsupported FITS data shape {data.shape}.")

    def _jy_beam_to_jy_pixel_factor(self, header):
        """Scale converting a WSClean-style Jy/beam image to Jy/pixel.

        ``Jy/pixel = Jy/beam * pixel_area / beam_area`` with the Gaussian
        restoring-beam solid angle ``Omega = (pi / 4 ln2) * BMAJ * BMIN``. A
        ``unit`` override on the spec wins over the header ``BUNIT``; a model
        image (Jy/pixel) needs no conversion.
        """
        bunit = (self.unit or str(header.get("BUNIT", ""))).strip().lower().replace(" ", "")
        if "beam" not in bunit:
            return 1.0
        bmaj, bmin = header.get("BMAJ"), header.get("BMIN")
        cdelt1, cdelt2 = header.get("CDELT1"), header.get("CDELT2")
        if None in (bmaj, bmin, cdelt1, cdelt2):
            raise ValueError(
                "FITS unit is Jy/beam but the restoring beam (BMAJ/BMIN) or pixel "
                "scale (CDELT1/CDELT2) is missing, so it cannot be converted to "
                "Jy/pixel."
            )
        beam_area = (np.pi / (4.0 * np.log(2.0))) * float(bmaj) * float(bmin)
        pixel_area = abs(float(cdelt1) * float(cdelt2))
        return pixel_area / beam_area

    @staticmethod
    def _fits_freqs(header, n_chan):
        from astropy.wcs import WCS

        wcs = WCS(header)
        if not wcs.has_spectral:
            raise ValueError(
                "FITS cube has no spectral WCS; provide a frequency axis or a "
                "single-channel (continuum) image."
            )
        return wcs.spectral.pixel_to_world(np.arange(n_chan)).to("Hz").value


class MSSource(SkySource):
    """A MeasurementSet visibility column. Its image is the (adjoint) dirty
    image; this is independent of the likelihood's ``data.data_col``."""

    def __init__(self, config, column=None):
        super().__init__(config)
        self._config = config
        self.column = column or config.args["data"]["data_col"]

    def _read_vis(self):
        data_col = self._config.args["data"]["data_col"]
        if self.column == data_col:
            vis_obs = getattr(self._config, "vis_obs", None)
            if vis_obs is None:
                raise ValueError(
                    f"from_ms column '{self.column}' is the data column but "
                    "config.vis_obs is not set."
                )
            return vis_obs
        from tabascal.tab_tools import read_ms

        ms = read_ms(
            self._config.ms_path,
            self._config.args["data"]["freq"],
            None,
            self._config.args["data"]["corr"],
            self.column,
        )
        return ms["vis_obs"]

    def visibilities(self):
        return self._read_vis()

    def image(self, grid):
        # (n_bl, n_freq, n_time) -> rows (n_bl*n_time, n_freq). Cast to the
        # default complex precision so it matches the plan's float precision
        # (MS data is complex64 but the x64 plan is float64; jax-finufft requires
        # source and coordinates to share precision).
        vis = jnp.asarray(self._read_vis()).astype(complex)
        vis_rows = vis.transpose(0, 2, 1).reshape(-1, self.n_freq)
        return vis2dirty(grid.plan, vis_rows)               # (n_freq, n_l, n_m)


def resolve_sky_source(spec: dict, config) -> SkySource:
    """Resolve a ``{"type": ...}`` source spec into a :class:`SkySource`.

    Types: ``zeros``, ``from_catalogue`` (``path``, ``fmt``: zarr|bbs),
    ``from_fits`` (``path``, ``unit``, ``hdu``), ``from_ms`` (``column``).
    The ``sample``/``prior`` init policies are parameter-space concepts handled
    by the consuming component, not sky sources, and are not resolved here.
    """
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"Sky-source spec must be a dict with a 'type'; got {spec!r}.")
    _check_stokes(spec)
    t = spec["type"]
    if t == "zeros":
        return ZerosSource(config)
    if t == "from_catalogue":
        return CatalogueSource(config, spec.get("path"), spec.get("fmt", "zarr"))
    if t == "from_fits":
        if "path" not in spec:
            raise ValueError("from_fits requires a 'path'.")
        return FitsSource(config, spec["path"], unit=spec.get("unit"),
                          hdu=spec.get("hdu", 0))
    if t == "from_ms":
        return MSSource(config, spec.get("column"))
    raise ValueError(
        f"Unknown sky-source type '{t}'. Choose from "
        "(zeros, from_catalogue, from_fits, from_ms)."
    )


# ---------------------------------------------------------------------------
# WSClean / DP3 BBS component-list parser
# ---------------------------------------------------------------------------

def _parse_sexagesimal_ra(s: str) -> float:
    """RA string -> degrees. Accepts decimal degrees, ``HH:MM:SS(.s)`` or
    ``HHhMMmSSs`` (hour-angle)."""
    s = s.strip()
    if any(c in s for c in ":hH") and not s.lstrip("+-").replace(".", "").isdigit():
        from astropy.coordinates import Angle
        import astropy.units as u

        return float(Angle(s.replace("h", ":").replace("m", ":").replace("s", ""),
                           unit=u.hourangle).deg)
    return float(s)


def _parse_sexagesimal_dec(s: str) -> float:
    """Dec string -> degrees. Accepts decimal degrees, ``DD.MM.SS(.s)`` or
    ``DDdMMmSSs``/``DD:MM:SS``."""
    s = s.strip()
    from astropy.coordinates import Angle
    import astropy.units as u

    if "d" in s or ":" in s:
        return float(Angle(s.replace("d", ":").replace("m", ":").replace("s", ""),
                           unit=u.deg).deg)
    parts = s.lstrip("+-").split(".")
    # Decimal degrees look like "-52.34" (<= 2 dotted groups); sexagesimal
    # "DD.MM.SS.sss" has >= 3.
    if len(parts) >= 3:
        sign = -1.0 if s.strip().startswith("-") else 1.0
        deg, arcmin, arcsec = float(parts[0]), float(parts[1]), float(".".join(parts[2:]))
        return sign * (deg + arcmin / 60.0 + arcsec / 3600.0)
    return float(s)


def _parse_bbs(path: str, dst_freqs):
    """Parse a WSClean/DP3 BBS component list into ``(radec_deg, flux)``.

    Supports the ``format = ...`` header, POINT/GAUSSIAN types (GAUSSIAN treated
    as a point at its centroid, with a warning), Stokes-I flux, and a
    logarithmic spectral index with a reference frequency:
    ``log10(I/I0) = sum_k si[k] * log10(f/f0)**(k+1)``. Returns flux on
    ``dst_freqs`` with shape ``(n_src, n_freq)``.
    """
    dst_freqs = np.asarray(dst_freqs, dtype=float)
    fmt = None
    defaults = {}
    rows = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if fmt is None and "format" in low and "=" in line:
                # DP3:     "Format = Name, Type, Ra, Dec, I, SpectralIndex, ..."
                # WSClean: "# (Name, Type, Ra, Dec, I, ...) = format"
                # Split on the bare `format` keyword's side, not the first '='
                # (a column default like ReferenceFrequency='150e6' has its own).
                stripped = line.strip().strip("#").strip()
                if stripped.lower().endswith("format"):
                    cols = stripped.rsplit("=", 1)[0]            # "(cols) = format"
                else:
                    cols = stripped.split("=", 1)[1]             # "format = cols"
                cols = cols.strip().strip("#() ")
                fmt = []
                for col in cols.split(","):
                    col = col.strip()
                    if not col:
                        continue
                    # A column may carry a default, e.g. "ReferenceFrequency='150e6'".
                    if "=" in col:
                        name, dflt = col.split("=", 1)
                        name = name.strip().lower()
                        defaults[name] = dflt.strip().strip("'\"")
                    else:
                        name = col.strip().lower()
                    fmt.append(name)
                continue
            if line.startswith("#"):
                continue
            rows.append([c.strip() for c in line.split(",")])

    if fmt is None:
        raise ValueError(f"BBS file {path} has no 'format' header line.")

    idx = {name: i for i, name in enumerate(fmt)}
    for required in ("name", "ra", "dec", "i"):
        if required not in idx:
            raise ValueError(f"BBS format missing required column '{required}': {fmt}")

    def cell(row, key, default=""):
        i = idx.get(key)
        if i is not None and i < len(row) and row[i].strip():
            return row[i].strip()
        return defaults.get(key, default)

    n_si_warned = False
    radec, flux = [], []
    for row in rows:
        stype = cell(row, "type", "POINT").upper()
        if stype.startswith("GAUSS") and not n_si_warned:
            warnings.warn(
                f"BBS {path}: GAUSSIAN components treated as points at their "
                "centroid (extent ignored).", UserWarning)
            n_si_warned = True
        ra_deg = _parse_sexagesimal_ra(cell(row, "ra"))
        dec_deg = _parse_sexagesimal_dec(cell(row, "dec"))
        i0 = float(cell(row, "i", "0") or 0.0)

        si_str = cell(row, "spectralindex", "")
        f0_str = cell(row, "referencefrequency", "")
        if si_str and f0_str:
            si = [float(x) for x in si_str.strip("[]").split(",") if x.strip()]
            f0 = float(f0_str)
            x = np.log10(dst_freqs / f0)
            expo = sum(si[k] * x ** (k + 1) for k in range(len(si)))
            spec = i0 * 10.0 ** expo
        else:
            spec = np.full(dst_freqs.shape, i0)
        radec.append((ra_deg, dec_deg))
        flux.append(spec)

    if not radec:
        raise ValueError(f"BBS file {path} contained no components.")
    return np.asarray(radec), np.asarray(flux)
