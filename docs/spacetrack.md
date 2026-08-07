# TLE Retrieval, Caching, and Space-Track Fallback

TABASCAL retrieves satellite TLEs from the
[IAU CPS SatChecker](https://satchecker.cps.iau.org/) service — **no account or
credentials are required**. This page explains where TLEs come from, how they
are cached, how to reproduce a run's TLEs exactly, and how to supply TLEs
manually (e.g. from Space-Track) when SatChecker cannot provide them. The
second half of the page is a column reference for the Space-Track/OMM format
that manually supplied files may use.

## Where TLEs come from

For each requested NORAD ID, sources are consulted **independently per
satellite**, in this order:

1. **`extra_tle_dir`** — a user-supplied directory of local TLE files
   (`--extra-tle-dir` on the CLI, or `satellites.extra_tle_dir` in the config).
   The record whose TLE epoch is closest to the observation is chosen and, if it
   passes the `extra_tle_max_age_days` policy, it wins outright — no service
   call is made for that satellite.
2. **The managed catalogue snapshot** — a full SatChecker catalogue downloaded
   once per fixed UTC time bucket (default 2 h wide, see
   `tle_catalogue_interval_hours`) and cached locally.
3. **Per-satellite fallback** — an individual SatChecker `get-nearest-tle`
   request for any ID still missing, cached alongside the snapshot.

If a satellite cannot be resolved by any source, the run continues without it —
with a loud warning naming the excluded NORAD IDs — since a missing RFI source
degrades subtraction quality. If *no* satellite resolves, the run stops with a
clear error.

## Caching behaviour by scenario

The managed cache lives in the platform user-cache directory (e.g.
`~/.cache/tle-cache` on Linux, `~/Library/Caches/tle-cache` on macOS), or
`TLE_CACHE_DIR` if set. Snapshots are keyed by a *canonical epoch*: the midpoint
of the fixed UTC bucket containing the observation, so the snapshot used depends
only on the observation time and bucket width — never on what happens to be
cached already.

| Scenario | What happens | Later runs |
|---|---|---|
| Historical epoch, catalogue available | Full catalogue downloaded once, stored as `catalogue-<stamp>.json` | Served from cache **forever** — deterministic, never refreshed |
| Recent epoch beyond SatChecker's data horizon (catalogue empty) | Per-satellite fallback; records stored as `catalogue-<stamp>-extra.json`; **no snapshot is stored** | The catalogue is re-attempted on **every** run; once SatChecker backfills the epoch, the snapshot is fetched and takes precedence over the cached fallback records |
| Satellite missing from an existing snapshot | Per-satellite fallback for that ID, cached in the `-extra` file | Reused from the `-extra` cache; **not refreshed** even if SatChecker later adds the satellite to the catalogue |
| Service unreachable, snapshot cached | Cache hit — run proceeds offline | — |
| Service unreachable, no snapshot | Run fails fast with a clear error (no satellite-by-satellite retry storm) | — |

Two consequences worth understanding:

- **Determinism wins over freshness.** Once a snapshot exists for a bucket it is
  never refreshed, even if SatChecker later serves better (closer-epoch) records
  for that time. This is deliberate: rerunning an analysis gives the same
  trajectory priors. To force a refresh, delete the relevant
  `catalogue-<stamp>.json` / `catalogue-<stamp>-extra.json` files from the cache
  directory (or point `TLE_CACHE_DIR` at a fresh directory) — the next run
  re-downloads.
- **The empty-catalogue fallback self-heals.** Because an empty catalogue is
  never stored, a recent observation first processed with (possibly stale)
  per-satellite TLEs will automatically pick up the proper catalogue snapshot
  once SatChecker's ingest catches up — improving the priors on a rerun. If you
  need the *original* run's priors instead, use the saved run TLEs (next
  section).

## Reproducing a run's TLEs exactly

Every `tabascal` run writes the TLEs it actually used to
`<sim_dir>/results/used_tles_<name>.json`, in exactly the format
`--extra-tle-dir` reads. To reproduce the run's trajectory priors later —
regardless of cache state or what SatChecker serves by then — copy that file
into a directory and pass it back:

```bash
mkdir run_tles && cp path/to/results/used_tles_Custom.json run_tles/
tabascal run -c config.yaml -s sim_dir --extra-tle-dir run_tles
```

With the default `extra_tle_max_age_days: null` (unlimited), every satellite
resolves from the saved file and no service call is made.

## Supplying TLEs manually (Space-Track workaround)

SatChecker's archive does not cover all epochs (very recent observations can be
beyond its ingest horizon, and old observations may predate its records). For
any satellite/epoch it cannot serve, supply TLEs yourself via `--extra-tle-dir`:

1. Obtain TLEs near your observation epoch — for historical data, a
   [Space-Track](https://www.space-track.org/) account gives access to the
   `gp_history` API, e.g. (after authenticating):

   ```
   https://www.space-track.org/basicspacedata/query/class/gp_history/
       NORAD_CAT_ID/25544,20452/EPOCH/2023-02-20--2023-02-23/format/json
   ```

2. Save the records as one or more `*.json` files in a directory, as a
   pandas-oriented JSON table (`pandas.DataFrame.to_json(path)`) with at least
   the columns `NORAD_CAT_ID`, `TLE_LINE1` and `TLE_LINE2`. Space-Track's JSON
   output already uses these column names (the full column set is documented
   below); orbital-element columns are ignored — TABASCAL parses the elements
   locally from the two TLE lines.

   ```python
   import pandas as pd
   df = pd.DataFrame(records)  # from the Space-Track JSON response
   df.to_json("my_tles/2023-02-21-gps.json")
   ```

3. Pass the directory to the run: `--extra-tle-dir my_tles` (or set
   `satellites.extra_tle_dir`). Set `extra_tle_max_age_days` if you want stale
   local records rejected in favour of the service.

---

# Space-Track Orbital Data Column Definitions (legacy format reference)

Complete reference for all columns returned by Space-Track.org GP/GP_History API when retrieving satellite orbital data in JSON/OMM format. TABASCAL no longer queries Space-Track directly, but `extra_tle_dir` files may use this format (only `NORAD_CAT_ID`, `TLE_LINE1` and `TLE_LINE2` are required).

**Data Source:** Space-Track.org (18th Space Defense Squadron, US Space Force)  
**Standard:** CCSDS Orbit Data Messages (ODM) Recommended Standard 502.0-B-3  
**Propagation Model:** SGP4/SDP4 (Simplified General Perturbations)

---

## Header/Metadata Fields

### CCSDS_OMM_VERS
**Type:** `varchar(3)`

OMM (Orbit Mean-Elements Message) format version number.
- Typically "2.0" for current CCSDS standard
- Indicates which version of the CCSDS 502.0 standard is being used

### COMMENT
**Type:** `varchar(33)`

Free-text comment field describing the data.
- Often contains information about the data source or generation method
- Example: "GENERATED VIA SPACE-TRACK.ORG API"

### CREATION_DATE
**Type:** `datetime`

UTC timestamp when this element set was created/published.
- Format: `YYYY-MM-DDTHH:MM:SS` or `YYYY-MM-DDTHH:MM:SS.ssssss`
- When 18 SDS generated this particular TLE/OMM

### ORIGINATOR
**Type:** `varchar(7)`

Organization that created/published the orbital elements.
- For Space-Track data, typically "18 SPCS" or "JSPOC"
- 18 SPCS = 18th Space Defense Squadron
- JSPOC = Joint Space Operations Center (legacy)

---

## Object Identification

### OBJECT_NAME
**Type:** `varchar(25)` (nullable)

Common name of the satellite.
- Examples: "ISS (ZARYA)", "STARLINK-1234", "COSMOS 2251 DEB"
- May be blank for analyst objects (80000-series catalog numbers)
- Not guaranteed to be unique

### OBJECT_ID
**Type:** `varchar(12)` (nullable)

International Designator (also called COSPAR ID).
- Format: `YYYY-NNN[A-ZZZ]` (year-launch number-piece)
- Example: "1998-067A" = 67th launch of 1998, piece A
- Permanently tied to the original launch, unlike NORAD_CAT_ID
- Components:
  - `YYYY` = launch year
  - `NNN` = launch number of that year (001-999)
  - `A-ZZZ` = piece of that launch (A, B, C, ..., AA, AB, ...)

### NORAD_CAT_ID
**Type:** `int(10) unsigned`

NORAD Catalog Number (unique satellite identifier).
- 5 digits historically, now expanding to 9 digits (Alpha-5 encoding)
- Assigned sequentially by US Space Force as objects are tracked
- Example: 25544 = ISS
- Can change if object is re-cataloged (rare)
- Primary key for satellite catalog

### CENTER_NAME
**Type:** `varchar(5)`

Central body being orbited.
- Always "EARTH" for Space-Track data
- CCSDS standard allows other bodies (MOON, MARS, etc.)

### REF_FRAME
**Type:** `varchar(4)`

Reference frame for the orbital elements.
- For Space-Track: "TEME" (True Equator Mean Equinox)
- TEME is the native frame for SGP4/SDP4 propagation
- **Note:** Ambiguity exists between TEME of Date vs TEME of Epoch

### TIME_SYSTEM
**Type:** `varchar(3)`

Time system used for EPOCH and other timestamps.
- Always "UTC" for Space-Track data
- Coordinated Universal Time

### MEAN_ELEMENT_THEORY
**Type:** `varchar(4)`

Orbital propagation model/theory used.
- Typically "SGP4" for Space-Track data
- Indicates these are mean elements, not osculating elements
- Mean elements are averaged over orbital period

---

## Epoch and Keplerian Orbital Elements

### EPOCH
**Type:** `datetime(6)` (nullable)  
**Units:** UTC time

Reference time for the orbital elements.
- Format: `YYYY-MM-DDTHH:MM:SS.ssssss`
- The "true position" time - elements are most accurate here
- Accuracy degrades as you propagate away from epoch
- Typical accuracy: ~1 km at epoch, degrades with time

### MEAN_MOTION
**Type:** `decimal(13,8)` (nullable)  
**Units:** revolutions/day

Mean motion in revolutions per day.
- How many times the satellite orbits Earth in 24 hours
- Example: ~15.5 for ISS (about 93-minute orbit)
- Used by SGP4 instead of semi-major axis
- Range: ~0.5 (GEO) to ~17 (very low LEO)

### ECCENTRICITY
**Type:** `decimal(13,8)` (nullable)  
**Units:** dimensionless (0 ≤ e < 1)

Orbital eccentricity.
- 0 = perfect circle, closer to 1 = more elliptical
- Example: 0.0001 for near-circular LEO, 0.7+ for Molniya
- Defines orbit shape (how "stretched" the ellipse is)
- Most LEO satellites: e < 0.01

### INCLINATION
**Type:** `decimal(7,4)` (nullable)  
**Units:** degrees (0° to 180°)

Orbital inclination.
- Angle between orbital plane and Earth's equator
- 0° = equatorial orbit
- 90° = polar orbit
- \>90° = retrograde (moves opposite to Earth's rotation)
- Example: 51.6° for ISS

### RA_OF_ASC_NODE
**Type:** `decimal(7,4)` (nullable)  
**Units:** degrees (0° to 360°)

Right Ascension of Ascending Node (RAAN).
- Longitude where orbit crosses equator going northward
- Also called Ω (Omega, uppercase)
- Defines orientation of orbital plane
- Precesses due to Earth's oblateness (J2 effect)
- Sun-synchronous satellites use J2 precession to maintain constant RAAN

### ARG_OF_PERICENTER
**Type:** `decimal(7,4)` (nullable)  
**Units:** degrees (0° to 360°)

Argument of Perigee.
- Angle from ascending node to perigee (closest point to Earth)
- Also called ω (omega, lowercase)
- Defines where in the orbit perigee occurs
- Undefined for circular orbits (e ≈ 0)

### MEAN_ANOMALY
**Type:** `decimal(7,4)` (nullable)  
**Units:** degrees (0° to 360°)

Mean anomaly.
- Angular position of satellite in its orbit at epoch
- 0° = at perigee, 180° = at apogee
- "Mean" = averaged over the orbit, not true position
- Used instead of true anomaly because it increases linearly with time

---

## TLE-Specific Parameters (SGP4/SDP4 Model)

### EPHEMERIS_TYPE
**Type:** `tinyint(4)` (nullable)

Type of ephemeris/propagator.
- 0 = SGP, 2 = SGP4, 4 = SGP4-XP, 6 = SP
- Typically 0 for modern Space-Track data (though SGP4 is actually used)
- Historical field, mostly ignored by modern propagators
- Default: 0

### CLASSIFICATION_TYPE
**Type:** `char(1)` (nullable)

Security classification of the orbital data.
- "U" = Unclassified (public)
- "C" = Classified, "S" = Secret (not on public Space-Track)
- Public data is always "U"

### ELEMENT_SET_NO
**Type:** `smallint(5) unsigned` (nullable)

Element set number for this satellite.
- Incremented each time a new TLE is generated
- Wraps around at 999 (modulo 1000)
- Used to distinguish different TLEs for same satellite
- Modern Space-Track often shows 999 as default

### REV_AT_EPOCH
**Type:** `mediumint(8) unsigned` (nullable)

Revolution number at epoch.
- Counts completed orbits since launch
- Revolution starts/ends at ascending node (RAAN crossing)
- Can be used to track total orbits over satellite lifetime
- Wraps at different values depending on system

### BSTAR
**Type:** `decimal(19,14)` (nullable)  
**Units:** 1/Earth radii (1/ER)

SGP4/SDP4 drag-like coefficient.
- Combines: drag coefficient, area-to-mass ratio, atmospheric density
- Higher = more drag = faster orbital decay
- Typical range: 1e-5 (high altitude) to 5e-4 (very low LEO)
- Often ~0 for satellites with active station-keeping
- Empirical parameter fitted from tracking data
- Also captures unmodeled perturbations (solar pressure, etc.)

### MEAN_MOTION_DOT
**Type:** `decimal(9,8)` (nullable)  
**Units:** revolutions/day²

First time derivative of mean motion.
- Rate of change of orbital period due to drag
- Used for decay prediction
- Often very small or zero in modern TLEs

### MEAN_MOTION_DDOT
**Type:** `decimal(22,13)` (nullable)  
**Units:** revolutions/day³

Second time derivative of mean motion.
- Acceleration of orbital decay
- Rarely used, often zero
- Generally negligible effect

---

## Derived/Computed Fields (NOT Required for SGP4)

*These fields are computed from the Keplerian elements for convenience. Space-Track added these to allow filtering queries without client-side computation.*

### SEMIMAJOR_AXIS
**Type:** `double(12,3)` (nullable)  
**Units:** km

Semi-major axis.
- Average of perigee and apogee distances from Earth's center
- Derived from mean motion using: `a = (μ/(n·2π/86400)²)^(1/3)`
- μ = 398600.4418 km³/s² (Earth's gravitational parameter)
- Defines orbit size
- For circular orbits: a = orbital radius

### PERIOD
**Type:** `double(12,3)` (nullable)  
**Units:** minutes

Orbital period.
- Time to complete one orbit
- Derived from mean motion: `Period = 1440 / MEAN_MOTION`
- Example: ~93 minutes for ISS
- LEO: ~90-120 min, GEO: ~1436 min (24 hours)

### APOAPSIS
**Type:** `double(12,3)` (nullable)  
**Units:** km

Apogee altitude (above Earth's surface).
- Highest point of orbit above Earth
- Calculated: `(a × (1 + e)) - 6378.137`
- Uses WGS-84 Earth radius (6378.137 km)
- For circular orbits: APOAPSIS ≈ PERIAPSIS

### PERIAPSIS
**Type:** `double(12,3)` (nullable)  
**Units:** km

Perigee altitude (above Earth's surface).
- Lowest point of orbit above Earth
- Calculated: `(a × (1 - e)) - 6378.137`
- Uses WGS-84 Earth radius (6378.137 km)
- Negative values indicate sub-orbital or decaying objects

---

## Satellite Catalog (SATCAT) Fields

*Additional metadata from the SATCAT database merged with TLE data.*

### OBJECT_TYPE
**Type:** `varchar(12)` (nullable)

Type/category of the space object.
- Values: "PAYLOAD", "ROCKET BODY", "DEBRIS", "UNKNOWN"
- **PAYLOAD** = operational satellites, spacecraft
- **ROCKET BODY** = spent launch vehicle stages
- **DEBRIS** = fragments from breakups/collisions/explosions
- **UNKNOWN** = object with uncertain origin

### RCS_SIZE
**Type:** `char(6)` (nullable)

Radar Cross-Section size category.
- Qualitative measure of object's radar detectability
- Values: "SMALL", "MEDIUM", "LARGE"
- Related to physical size but also shape/material
- Rough size estimates:
  - **SMALL**: < 0.1 m² RCS (typically < 1 m diameter)
  - **MEDIUM**: 0.1-1 m² RCS (typically 1-10 m)
  - **LARGE**: > 1 m² RCS (typically > 10 m)

### COUNTRY_CODE
**Type:** `char(6)` (nullable)

Country or organization code (ISO-like).
- Examples: "US", "CIS" (Russia), "PRC" (China), "ESA", "JPN"
- Based on launch/ownership, not current operator
- May be consortium code for international missions
- CIS = Commonwealth of Independent States (former Soviet)

### LAUNCH_DATE
**Type:** `date` (nullable)

Date of launch.
- Format: `YYYY-MM-DD`
- Taken from SATCAT database
- May be NULL for some analyst objects
- Precision varies (some only have year/month)

### SITE
**Type:** `char(5)` (nullable)

Launch site code.
- Examples:
  - "AFETR" = Cape Canaveral, Florida
  - "TYMSC" = Plesetsk, Russia
  - "VOSTO" = Vostochny, Russia
  - "WLPIS" = Wallops Island, Virginia
  - "UNKN" = Unknown
- Uses standardized 5-character codes
- From SATCAT database

### DECAY_DATE
**Type:** `date` (nullable)

Date of orbital decay/reentry.
- Format: `YYYY-MM-DD`
- NULL if object is still on-orbit
- Predicted or actual reentry date
- Once set, object is considered "decayed" in catalog
- No new TLEs generated after decay

---

## Internal Space-Track Database Fields

### FILE
**Type:** `bigint(20) unsigned` (nullable)

Internal Space-Track file identifier.
- Database reference number
- Used for internal tracking/storage
- Not generally useful for users

### GP_ID
**Type:** `int(10) unsigned`

General Perturbations unique identifier.
- Primary key for GP/GP_History database tables
- Unique for each element set entry
- Used to reference specific TLE in Space-Track database
- Increments sequentially

---

## TLE Text Format Fields

*These contain the same data as above but in traditional TLE text format.*

### TLE_LINE0
**Type:** `varchar(27)` (nullable)

Line 0 of Three-Line Element (3LE) format.
- Contains the satellite name
- Example: "ISS (ZARYA)"
- Same as OBJECT_NAME but in TLE text format
- 24 characters max

### TLE_LINE1
**Type:** `varchar(71)` (nullable)

Line 1 of the TLE in fixed-width text format.
- Contains: catalog ID, classification, international designator, epoch, mean motion derivatives, BSTAR, ephemeris type, element set number
- 69 characters, ends with checksum digit (modulo-10)
- Format is strictly defined for compatibility with legacy systems
- Example:
```
  1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
```

### TLE_LINE2
**Type:** `varchar(71)` (nullable)

Line 2 of the TLE in fixed-width text format.
- Contains: catalog ID, inclination, RAAN, eccentricity, arg of perigee, mean anomaly, mean motion, revolution number
- 69 characters, ends with checksum digit (modulo-10)
- Example:
```
  2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537
```

---

## Important Notes

### SGP4 Minimum Requirements

To propagate an orbit using SGP4, you need:
- `EPOCH` (reference time)
- `MEAN_MOTION` (revolutions/day)
- `ECCENTRICITY` (0-1)
- `INCLINATION` (degrees)
- `RA_OF_ASC_NODE` (degrees)
- `ARG_OF_PERICENTER` (degrees)
- `MEAN_ANOMALY` (degrees)
- `BSTAR` (drag term, 1/ER)
- `NORAD_CAT_ID` (for identification)

### SGP4 vs SDP4 Selection

Automatically determined by orbital period:
- **SGP4**: period < 225 minutes (near-Earth orbits)
- **SDP4**: period ≥ 225 minutes (deep-space, GEO, HEO)

Modern libraries handle this automatically.

### Mean vs Osculating Elements

- TLE/OMM elements are **MEAN** elements (averaged over orbit)
- **NOT** directly convertible to Cartesian state vectors
- **MUST** use SGP4/SDP4 to propagate, then convert to position/velocity
- Using other propagators with TLE data will give incorrect results

### Reference Frame Ambiguity

- TEME has two interpretations: TEME of Date vs TEME of Epoch
- 18 SDS has never officially stated which they use
- For sub-km precision, this ambiguity matters
- For typical applications, the difference is negligible

### Accuracy Expectations

- ~1 km position accuracy at epoch
- Degrades to ~5-10 km after several days (LEO)
- GEO satellites maintain better accuracy over time
- Accuracy limited by:
  - Atmospheric density model uncertainties
  - Unmodeled perturbations
  - Observation errors
  - Mean element theory approximations

### No Covariance Data

Space-Track does **NOT** provide uncertainty/covariance matrices in standard GP data. For uncertainty estimates, you must:
- Analyze historical TLE time series
- Use published error models (~1 km typical)
- Request CDMs (Conjunction Data Messages) if you're an operator

### Derived Fields

`SEMIMAJOR_AXIS`, `PERIOD`, `APOAPSIS`, `PERIAPSIS` are computed from the Keplerian elements and may not exactly match SATCAT values due to different computation methods.

### Data Freshness

- `EPOCH` shows when elements were valid
- `CREATION_DATE` shows when TLE was generated
- These can differ by hours to days
- TLEs are typically updated every 1-3 days for active satellites
- High-interest objects updated more frequently

### Analyst Objects

- Catalog numbers 80000+ are analyst satellites
- Tracked with insufficient fidelity for full catalog
- May have missing fields (`OBJECT_NAME`, `OBJECT_ID`)
- Used for newly launched objects before full characterization

---

## References

- [CCSDS 502.0-B-3: Orbit Data Messages](https://ccsds.org/Pubs/502x0b3e1.pdf) (OMM Standard)
- [Space-Track.org API Documentation](https://www.space-track.org/documentation)
- [CelesTrak Documentation](https://celestrak.org/NORAD/documentation/)
- Spacetrack Report #3: SGP4/SDP4 Models
- Revisiting Spacetrack Report #3 (Vallado et al.)
