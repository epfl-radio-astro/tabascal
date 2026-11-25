"""
ast_signal.py — sky-signal components.

Components that put a *sky* into the model state, as opposed to ``ast_vis``, which turns
a sky into visibilities.

"Discrete" here means a discrete set of parametric sources — a catalogue of positions,
fluxes and shapes — as opposed to the image-plane sky an ``ImageSky``/``ImageSkyVis``
pair would carry.
"""

import os
from typing import Any, ClassVar, Optional, Union
import warnings

import jax.numpy as jnp
import numpy as np
from pydantic import Field, ValidationInfo, field_validator, model_validator

from tabascal.components import Component
from tabascal.components.ast_vis import radec_to_lmn
from tabascal.schema import FiniteFloat, StrictConfigModel


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

#: The legacy 11-column OSKAR layout: the first eight modern columns followed by the
#: Gaussian shape, with no rotation measure. It is not the modern layout truncated, so
#: an 11-column row read as a modern one puts the major axis in the rotation-measure
#: column and the shape one place left of where it belongs.
OSKAR_LEGACY_COLUMNS = OSKAR_COLUMNS[:8] + OSKAR_COLUMNS[9:]

#: Parsed but not modelled. See :func:`_check_unpolarised`.
POLARISATION_COLUMNS = ("Q", "U", "V", "rm")

#: The inline-YAML source fields: the OSKAR column each maps to, and a description of
#: the value used when reporting a bad one. ``ra``, ``dec`` and ``I`` are required; the
#: rest default to zero. ``name`` is handled separately, as it is not a number.
INLINE_FIELDS = {
    "ra": ("ra_deg", "right ascension in degrees"),
    "dec": ("dec_deg", "declination in degrees"),
    "I": ("I", "Stokes I flux in Jy"),
    "ref_freq_mhz": ("ref_freq_hz", "reference frequency in MHz"),
    "alpha": ("alpha", "spectral index, dimensionless"),
    "Q": ("Q", "Stokes Q flux in Jy"),
    "U": ("U", "Stokes U flux in Jy"),
    "V": ("V", "Stokes V flux in Jy"),
    "rm": ("rm", "rotation measure in rad/m^2"),
    "fwhm_major_arcsec": ("fwhm_major_arcsec", "major-axis FWHM in arcsec"),
    "fwhm_minor_arcsec": ("fwhm_minor_arcsec", "minor-axis FWHM in arcsec"),
    "position_angle_deg": ("position_angle_deg", "position angle in degrees"),
}

INLINE_REQUIRED = ("ra", "dec", "I")

ARCSEC_TO_RAD = np.pi / (180 * 3600)


def _oskar_columns(n_field: int):
    """The column names an OSKAR row of ``n_field`` fields carries, or None.

    OSKAR's fixed-format reader accepts three lengths: 3 to 9 fields are the leading
    modern columns with the rest defaulted, 11 is the legacy layout, and 12 is the full
    modern one. 10 is a half-specified Gaussian (a major axis with no minor axis or
    position angle) and 13 or more is not the format at all.
    """
    if 3 <= n_field <= 9:
        return OSKAR_COLUMNS[:n_field]
    if n_field == len(OSKAR_LEGACY_COLUMNS):
        return OSKAR_LEGACY_COLUMNS
    if n_field == len(OSKAR_COLUMNS):
        return OSKAR_COLUMNS
    return None


def read_oskar_sky_model(path: str) -> list:
    """Read an OSKAR sky model file into a list of row dicts.

    One source per line, fields separated by whitespace and/or commas, ``#`` starts a
    comment and blank lines are skipped. The columns are :data:`OSKAR_COLUMNS`; a row may
    stop after any column from Stokes I onwards and the rest default to zero, so
    ``ra dec I`` is a valid flat-spectrum point source. An 11-column row is the legacy
    layout, :data:`OSKAR_LEGACY_COLUMNS`.
    """
    rows = []
    name = os.path.basename(path)

    with open(path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue

            fields = line.replace(",", " ").split()
            columns = _oskar_columns(len(fields))
            if columns is None:
                raise ValueError(
                    f"{path} line {lineno}: an OSKAR sky model row has 3 to 9, 11 "
                    f"(legacy) or 12 columns, got {len(fields)}: {line!r}."
                )

            try:
                values = [float(x) for x in fields]
            except ValueError as e:
                raise ValueError(
                    f"{path} line {lineno}: every OSKAR sky model column is a number, "
                    f"but {line!r} is not ({e})."
                ) from e

            # float() reads 'nan', 'inf' and an overflowing '1e999' without complaint.
            # Left alone they turn the visibilities of every baseline the source touches
            # non-finite, and the optimiser reports that as a failure to converge a long
            # way from the row that caused it.
            for column, value in zip(columns, values):
                if not np.isfinite(value):
                    raise ValueError(
                        f"{path} line {lineno}: {column} is {value}, but every OSKAR "
                        f"sky model value must be finite: {line!r}."
                    )

            row = {column: 0.0 for column in OSKAR_COLUMNS}
            row.update(zip(columns, values))
            row["name"] = f"{name} line {lineno}"
            rows.append(row)

    return rows


def _inline_row(source, idx: int) -> dict:
    """Normalise one inline-YAML source onto the OSKAR column names.

    The YAML form names the reference frequency in MHz (``ref_freq_mhz``) because that is
    how catalogue fluxes are usually quoted; every other field maps straight across.
    """
    if not isinstance(source, dict):
        raise ValueError(
            f"FixedDiscreteSky.point_sources entry {idx} is a {type(source).__name__}, not a source: "
            "each entry is a mapping of fields, e.g. {name: Fornax A, ra: 50.7, "
            "dec: -37.2, I: 750.0}. "
            f"Got {source!r}."
        )

    name = source.get("name")
    # The index goes in even when the source is named: it is what locates the entry in
    # the file, and two sources may share a name.
    where = f"FixedDiscreteSky.point_sources entry {idx}" + (f" ({name!r})" if name else "")

    # Checked before the fields are read, so a misspelt key is reported as itself rather
    # than as whatever it left missing. An unrecognised field would otherwise be dropped
    # in silence and take its source's shape or spectrum with it.
    unknown = sorted(set(source) - set(INLINE_FIELDS) - {"name"}, key=str)
    if unknown:
        raise ValueError(
            f"{where} has unrecognised field(s) {unknown}. Fields are case-sensitive; "
            f"the accepted ones are {sorted(set(INLINE_FIELDS) | {'name'})}. A typo is "
            "not ignored here because it would quietly change the source: 'fwhm_maj' "
            "for 'fwhm_major_arcsec' would leave a Gaussian modelled as a point."
        )

    def value_of(key):
        units = INLINE_FIELDS[key][1]

        # An absent key and an explicit YAML null both mean "unset" -- null is how these
        # config files spell a default throughout. Anything else that is present has to
        # parse: `alpha: ""` is malformed catalogue data, not a request for a flat
        # spectrum, and quietly reading it as the default hides a broken source.
        if source.get(key) is None:
            if key in INLINE_REQUIRED:
                raise ValueError(
                    f"{where} has no {key!r}, which is required ({units}). "
                    f"Given fields: {sorted(source)}."
                )
            return 0.0

        value = source[key]
        # bool is an int in Python, so float(False) would pass silently as 0.
        if not isinstance(value, bool):
            try:
                number = float(value)
            except (TypeError, ValueError):
                pass
            else:
                # NaN and infinity convert without complaint and then spread into the
                # visibility of every baseline this source touches.
                if np.isfinite(number):
                    return number
                raise ValueError(
                    f"{where} has {key!r} = {value!r}, which is not a finite number "
                    f"({units})."
                )

        raise ValueError(
            f"{where} has {key!r} = {value!r}, which is not a number ({units})."
        )

    row = {column: value_of(key) for key, (column, _) in INLINE_FIELDS.items()}
    row["ref_freq_hz"] *= 1e6
    row["name"] = str(name) if name else f"src{idx}"

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


#: Inline source fields that default to zero when absent or null.
_OPTIONAL_SOURCE_FIELDS = (
    "ref_freq_mhz",
    "alpha",
    "Q",
    "U",
    "V",
    "rm",
    "fwhm_major_arcsec",
    "fwhm_minor_arcsec",
    "position_angle_deg",
)


class PointSourceConfig(StrictConfigModel):
    """One inline discrete source in :class:`FixedDiscreteSkyConfig`."""

    name: Optional[str] = Field(default=None, description="Optional human-readable source label.")
    ra: FiniteFloat = Field(description="Right ascension in degrees (J2000).")
    dec: FiniteFloat = Field(description="Declination in degrees (J2000).")
    I: FiniteFloat = Field(description="Stokes I flux density in Jy at ref_freq_mhz.")
    ref_freq_mhz: FiniteFloat = Field(default=0.0, description="Reference frequency in MHz.")
    alpha: FiniteFloat = Field(default=0.0, description="Power-law spectral index (dimensionless).")
    Q: FiniteFloat = Field(default=0.0, description="Stokes Q flux density in Jy (must be 0; unpolarised only).")
    U: FiniteFloat = Field(default=0.0, description="Stokes U flux density in Jy (must be 0; unpolarised only).")
    V: FiniteFloat = Field(default=0.0, description="Stokes V flux density in Jy (must be 0; unpolarised only).")
    rm: FiniteFloat = Field(default=0.0, description="Rotation measure in rad/m^2 (must be 0; unpolarised only).")
    fwhm_major_arcsec: FiniteFloat = Field(default=0.0, description="Gaussian major-axis FWHM in arcseconds.")
    fwhm_minor_arcsec: FiniteFloat = Field(default=0.0, description="Gaussian minor-axis FWHM in arcseconds.")
    position_angle_deg: FiniteFloat = Field(default=0.0, description="Major-axis position angle in degrees, north through east.")

    @field_validator("name", mode="before")
    @classmethod
    def _name_as_text(cls, value: Any) -> Any:
        # A catalogue name may be all digits (`name: 1934`), which YAML reads as a number.
        return None if value is None else str(value)

    @field_validator(
        "ra",
        "dec",
        "I",
        *_OPTIONAL_SOURCE_FIELDS,
        mode="before",
    )
    @classmethod
    def _read_number(cls, value: Any, info: ValidationInfo) -> Any:
        # null is how these config files spell a default; only the optional fields have one.
        if value is None and info.field_name in _OPTIONAL_SOURCE_FIELDS:
            return 0.0
        # Catalogue values are often quoted (`I: '750.0'`); read them as the numbers they
        # are, and leave the finite check to FiniteFloat.
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"must be a finite number, got {value!r}") from None
        return value

    @model_validator(mode="after")
    def _check_source_physics(self) -> "PointSourceConfig":
        offenders = [
            f"{col}={getattr(self, col):g}"
            for col in POLARISATION_COLUMNS
            if getattr(self, col) != 0.0
        ]
        if offenders:
            label = self.name or "source"
            raise ValueError(
                "Polarised sources are not modelled yet, so a non-zero Stokes Q/U/V or "
                f"rotation measure cannot be honoured: {label} ({', '.join(offenders)}). "
                "Set them to zero to model Stokes I alone; polarisation is tracked in "
                "issue #151."
            )
        if self.alpha != 0.0 and self.ref_freq_mhz <= 0.0:
            label = self.name or "source"
            raise ValueError(
                f"source {label} has a spectral index ({self.alpha:g}) but no positive "
                "reference frequency for it to be measured against; give a positive "
                "ref_freq_mhz or set alpha to zero."
            )
        return self

    def to_oskar_row(self, idx: int) -> dict:
        return {
            "ra_deg": self.ra,
            "dec_deg": self.dec,
            "I": self.I,
            "Q": self.Q,
            "U": self.U,
            "V": self.V,
            "ref_freq_hz": self.ref_freq_mhz * 1e6,
            "alpha": self.alpha,
            "rm": self.rm,
            "fwhm_major_arcsec": self.fwhm_major_arcsec,
            "fwhm_minor_arcsec": self.fwhm_minor_arcsec,
            "position_angle_deg": self.position_angle_deg,
            "name": str(self.name) if self.name else f"src{idx}",
        }


class FixedDiscreteSkyConfig(StrictConfigModel):
    """Configuration for :class:`FixedDiscreteSky`.

    ``point_sources`` is required: either a path to an OSKAR sky model file or a
    non-empty list of inline source mappings.
    """

    point_sources: Union[str, list[PointSourceConfig]] = Field(
        ...,
        description=(
            "Fixed (non-fitted) sky sources: a path to an OSKAR 12-column sky model "
            "file or a non-empty inline list of source mappings ({ra, dec, I, ...})."
        ),
        json_schema_extra={"input_path": True},
    )

    legacy_keys: ClassVar[dict[str, tuple[str, ...]]] = {
        "point_sources": ("ast", "point_sources"),
    }

    @field_validator("point_sources", mode="before")
    @classmethod
    def _validate_point_sources(cls, value: Any) -> Any:
        if value is None:
            raise ValueError(
                "FixedDiscreteSky requires point_sources: a path to an OSKAR sky model "
                "file or a non-empty list of inline source mappings."
            )
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("point_sources path must not be empty.")
            return value
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise ValueError("point_sources list must not be empty.")
            return list(value)
        raise ValueError(
            "point_sources must be a path to an OSKAR file or a non-empty list of "
            f"source mappings, got {type(value).__name__}."
        )


def _read_sources(point_sources) -> list:
    """Resolve ``point_sources`` — a path to an OSKAR file or an inline list."""
    if isinstance(point_sources, (str, os.PathLike)):
        rows = read_oskar_sky_model(point_sources)
    else:
        rows = [
            s.to_oskar_row(i) if isinstance(s, PointSourceConfig) else _inline_row(s, i)
            for i, s in enumerate(point_sources or [])
        ]

    if len(rows) == 0:
        raise ValueError(
            f"{FixedDiscreteSky.__name__} point_sources is empty."
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

    Config = FixedDiscreteSkyConfig

    required_inputs = {}  # No inputs needed
    output_shapes = {
        "ast_radec": ("n_src", 2),
        "ast_I": ("n_src", "n_freq"),
        "ast_shape": ("n_src", 3),
    }
    parameter_shapes = {}

    def setup(self, config):
        try:
            self.n_freq = config.n_freq
            self.freqs = jnp.asarray(config.freqs)  # Hz

            rows = _read_sources(self.config.point_sources)
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
            self._warn_beyond_the_horizon(rows, config.phase_centre)

            band = jnp.mean(self.ast_I, axis=1)
            print(
                "\nFixed discrete sky: "
                + ", ".join(f"{r['name']} ({b:.1f} Jy)" for r, b in zip(rows, band))
            )

            self._set_outputs()
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def _warn_beyond_the_horizon(self, rows, phase_centre):
        """Warn about sources more than 90 degrees from the phase centre.

        ``DiscreteSkyVis`` models these correctly -- only ``n - 1`` enters the phase and
        it is exact over the whole sphere -- so they are not rejected. But no real
        observation has a source on the far side of the sky in its field, so in practice
        this is a swapped or mis-signed coordinate in the catalogue, and silently
        modelling a mirrored sky is what a fixed sky exists to prevent.
        """
        _, _, n, _ = radec_to_lmn(
            self.ast_radec[:, 0],
            self.ast_radec[:, 1],
            jnp.deg2rad(phase_centre["ra"]),
            jnp.deg2rad(phase_centre["dec"]),
        )
        far = [row["name"] for row, n_k in zip(rows, np.asarray(n)) if n_k <= 0]
        if far:
            warnings.warn(
                f"{len(far)} source(s) are more than 90 degrees from the phase centre "
                f"and so behind the sky plane: {', '.join(far)}. They are modelled as "
                "given; check the catalogue coordinates and the phase centre.",
                UserWarning,
                stacklevel=2,
            )

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
