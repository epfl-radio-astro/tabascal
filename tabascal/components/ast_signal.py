"""
ast_signal.py — sky-signal components.

Components that put a *sky* into the model state, as opposed to ``ast_vis``, which turns
a sky into visibilities.

"Discrete" here means a discrete set of parametric sources — a catalogue of positions,
fluxes and shapes — as opposed to the image-plane sky an ``ImageSky``/``ImageSkyVis``
pair would carry.
"""

import os

import jax.numpy as jnp
import numpy as np

from tabascal.components import Component


#: The OSKAR sky model columns, in file order. Whitespace-separated text with ``#``
#: comments; trailing columns may be omitted and default to zero. This is the format
#: Karabo emits.
OSKAR_COLUMNS = (
    "ra_deg",               # Right ascension, degrees
    "dec_deg",              # Declination, degrees
    "I",                    # Stokes I, Jy, at ref_freq_hz
    "Q",                    # Stokes Q, Jy
    "U",                    # Stokes U, Jy
    "V",                    # Stokes V, Jy
    "ref_freq_hz",          # Reference frequency, Hz
    "alpha",                # Spectral index
    "rm",                   # Rotation measure, rad / m^2
    "fwhm_major_arcsec",    # Gaussian FWHM along the major axis, arcsec
    "fwhm_minor_arcsec",    # Gaussian FWHM along the minor axis, arcsec
    "position_angle_deg",   # Major-axis position angle, degrees, north through east
)

#: Parsed but not modelled. See :func:`_check_unpolarised`.
POLARISATION_COLUMNS = ("Q", "U", "V", "rm")

ARCSEC_TO_RAD = np.pi / (180 * 3600)


def read_oskar_sky_model(path: str) -> list:
    """Read an OSKAR 12-column sky model file into a list of row dicts.

    One source per line, whitespace-separated, ``#`` starts a comment; blank lines are
    skipped. The columns are :data:`OSKAR_COLUMNS`; a row may stop after any column from
    Stokes I onwards and the rest default to zero, so ``ra dec I`` is a valid
    flat-spectrum point source.
    """
    rows = []
    name = os.path.basename(path)

    with open(path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue

            fields = line.split()
            if not 3 <= len(fields) <= len(OSKAR_COLUMNS):
                raise ValueError(
                    f"{path} line {lineno}: an OSKAR sky model row has between 3 and "
                    f"{len(OSKAR_COLUMNS)} whitespace-separated columns, got "
                    f"{len(fields)}: {line!r}."
                )

            try:
                values = [float(x) for x in fields]
            except ValueError as e:
                raise ValueError(
                    f"{path} line {lineno}: every OSKAR sky model column is a number, "
                    f"but {line!r} is not ({e})."
                ) from e

            values += [0.0] * (len(OSKAR_COLUMNS) - len(values))
            row = dict(zip(OSKAR_COLUMNS, values))
            row["name"] = f"{name} line {lineno}"
            rows.append(row)

    return rows


def _inline_row(source: dict, idx: int) -> dict:
    """Normalise one inline-YAML source onto the OSKAR column names.

    The YAML form names the reference frequency in MHz (``ref_freq_mhz``) because that is
    how catalogue fluxes are usually quoted; every other field maps straight across.
    """
    get = lambda key, default=0.0: float(source.get(key, default) or default)

    row = {
        "ra_deg": float(source["ra"]),
        "dec_deg": float(source["dec"]),
        "I": float(source["I"]),
        "ref_freq_hz": get("ref_freq_mhz") * 1e6,
        "alpha": get("alpha"),
        "name": str(source.get("name", f"src{idx}")),
    }
    for key in ("Q", "U", "V", "rm", "fwhm_major_arcsec", "fwhm_minor_arcsec",
                "position_angle_deg"):
        row[key] = get(key)

    return row


def _check_unpolarised(rows: list) -> None:
    """Reject a catalogue carrying polarisation, naming the issue that will add it.

    Q, U, V and the rotation measure are read rather than ignored so that the OSKAR
    format is accepted *in full*: modelling polarisation later (issue #151) is then a
    widening of what these values mean, not a change to what the file may contain. Until
    then a non-zero value would be silently dropped, which is worse than refusing it.
    """
    offenders = [
        f"{row['name']} ({col}={row[col]:g})"
        for row in rows
        for col in POLARISATION_COLUMNS
        if row[col] != 0.0
    ]
    if offenders:
        raise ValueError(
            "Polarised sources are not modelled yet, so a non-zero Stokes Q/U/V or "
            "rotation measure cannot be honoured: "
            + "; ".join(offenders)
            + ". Set them to zero to model Stokes I alone; polarisation is tracked in "
            "issue #151."
        )


def _read_sources(point_sources) -> list:
    """Resolve ``ast.point_sources`` — a path to an OSKAR file or an inline list."""
    if isinstance(point_sources, (str, os.PathLike)):
        rows = read_oskar_sky_model(point_sources)
    else:
        rows = [_inline_row(s, i) for i, s in enumerate(point_sources or [])]

    if len(rows) == 0:
        raise ValueError(
            f"{FixedDiscreteSky.__name__} is in model.components but ast.point_sources "
            "is empty."
        )

    return rows


class FixedDiscreteSky(Component):
    """A fixed (non-fitted) sky of discrete sources — points and elliptical Gaussians.

    Why this exists: a free per-antenna gain is only identifiable against a sky the gain
    cannot deform. Every other sky model here is flexible — the astronomical GP has
    per-baseline freedom, so ``g_p conj(g_q) * vis_ast`` is a reparametrisation of an
    already-free ``vis_ast`` and the gain is a flat direction of the likelihood. A source
    with a KNOWN position, flux and shape is rigid, so it anchors the gain. The gain
    component it is meant to be paired with is the constant-gain model of issue #124.

    "Discrete" is the set of sources, not their size: a discrete source may be a point or
    an extended Gaussian. ``ImageSky`` is reserved for a sky carried as an image.

    Sources come either as an inline list, in degrees and Jy::

        ast:
          point_sources:
            - {name: Fornax A, ra: 50.6738, dec: -37.2083, I: 750.0,
               ref_freq_mhz: 154.0, alpha: -0.77}

    or as a path to an OSKAR 12-column sky model file (see
    :func:`read_oskar_sky_model`)::

        ast:
          point_sources: /path/to/sky.osm

    ``I`` is the flux at the reference frequency and the spectrum is the power law
    ``I(nu) = I * (nu / ref_freq)**alpha`` (``alpha: 0`` for a flat spectrum). A source
    with a non-zero major/minor FWHM is an elliptical Gaussian, with its position angle
    measured from north through east; zero FWHM is a point.

    Writes ``ast_radec`` (n_src, 2) in radians, ``ast_I`` (n_src, n_freq) in Jy and
    ``ast_shape`` (n_src, 3) as (FWHM major, FWHM minor, position angle) in radians.
    Carries no free parameters and pairs with
    :class:`~tabascal.components.ast_vis.DiscreteSkyVis`, which must be listed after it.

    NOTE the flux is in the same scale as the data the model is fit to, so with a gain
    table (data calibrated to Jy) these are physical Jy. Without it, the data are in raw
    correlator units and a Jy catalogue flux is meaningless.
    """

    required_inputs = {}  # No inputs needed
    parameter_shapes = {}

    def setup(self, config):
        try:
            self.n_freq = config.n_freq
            self.freqs = jnp.asarray(config.freqs)  # Hz

            rows = _read_sources(config.args["ast"].get("point_sources"))
            _check_unpolarised(rows)

            self.ast_radec = jnp.deg2rad(
                jnp.asarray([[r["ra_deg"], r["dec_deg"]] for r in rows], dtype=float)
            )
            self.ast_I = jnp.stack([self._spectrum(r) for r in rows], axis=0)
            self.ast_shape = jnp.asarray(
                [
                    [
                        r["fwhm_major_arcsec"] * ARCSEC_TO_RAD,
                        r["fwhm_minor_arcsec"] * ARCSEC_TO_RAD,
                        np.deg2rad(r["position_angle_deg"]),
                    ]
                    for r in rows
                ],
                dtype=float,
            )
            self.n_src = int(self.ast_radec.shape[0])

            band = jnp.mean(self.ast_I, axis=1)
            print(
                "\nFixed discrete sky: "
                + ", ".join(f"{r['name']} ({b:.1f} Jy)" for r, b in zip(rows, band))
            )

            self._set_outputs()
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def _spectrum(self, row):
        """The (n_freq,) flux of one source, ``I(nu) = I (nu / nu_ref)**alpha``."""
        alpha = row["alpha"]
        if alpha == 0.0:
            return row["I"] * jnp.ones(self.n_freq)

        ref_freq = row["ref_freq_hz"]
        if ref_freq <= 0:
            # Silently falling back to a flat spectrum would put a source at the wrong
            # flux everywhere except one channel, with nothing in the output to say so.
            raise ValueError(
                f"source {row['name']} has a spectral index ({alpha:g}) but no positive "
                "reference frequency for it to be measured against; give a reference "
                "frequency or set the spectral index to zero."
            )

        return row["I"] * (self.freqs / ref_freq) ** alpha

    def build_constants(self):
        return {
            "ast_radec": self.ast_radec,
            "ast_I": self.ast_I,
            "ast_shape": self.ast_shape,
        }

    def build_forward(self):
        prefix = self.prefix

        def forward(params, state, constants):
            return {
                **state,
                "ast_radec": constants[f"{prefix}/ast_radec"],
                "ast_I": constants[f"{prefix}/ast_I"],
                "ast_shape": constants[f"{prefix}/ast_shape"],
            }

        return forward

    def _set_outputs(self):
        self.state_outputs = {
            "ast_radec": self.ast_radec,
            "ast_I": self.ast_I,
            "ast_shape": self.ast_shape,
        }
