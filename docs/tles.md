# Two-Line Elements (TLEs)

Satellite trajectories in TABASCAL are predicted from Two-Line Element sets
(TLEs) — the standard published form of a satellite's mean orbital elements at a
reference epoch. TABASCAL retrieves TLEs from the
[IAU CPS SatChecker](https://satchecker.cps.iau.org/) service — **no account or
credentials are required**. This page explains where TLEs come from, how they
are cached, how to reproduce a run's TLEs exactly, and how to supply TLEs
manually (e.g. from Space-Track) when SatChecker cannot provide them. The
second half of the page is a column reference for the Space-Track/OMM format
that manually supplied files may use.

```{note}
Two of the defaults on this page are **provisional client safeguards, not
scientific results**, and they mean different things:

* `remote_tle_max_age_days: 3` is an emergency ceiling on how old a TLE the
  *service* hands you may be. It is not a claim that a three-day-old TLE is
  accurate enough for your observation. Its calibrated, observation-specific
  replacement is [issue #101](https://github.com/epfl-radio-astro/tabascal/issues/101).
* `tle_catalogue_settle_days: 45` is a *caching* boundary: how long to wait
  before trusting that SatChecker's catalogue for an epoch has stopped filling.
  It is an observed defensive policy, not a SatChecker API guarantee, and it says
  nothing about whether a 45-day-old TLE is usable.

Expect both to change.
```

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
3. **Per-satellite lookup** — an individual SatChecker `get-nearest-tle`
   request, cached alongside the snapshot. Used for any ID the snapshot is
   missing, *and* to replace a snapshot record older than
   `remote_tle_target_age_days` with the nearest one the service holds (see
   [Which endpoint answered changes the TLE age](#sec-endpoint-freshness)).

## Every configured satellite must resolve

A run **stops before subtraction** if any configured NORAD ID has no acceptable
TLE. The check happens during preflight, before the visibilities are read, and
the error names every failing satellite, how close the best available record
was, and the age limit that rejected it:

```text
Error: TLEs could not be resolved for 1 of 3 configured satellites at observation
epoch 2026-08-11T02:14:00 UTC (catalogue bucket 20260811T030000Z):
  25544: best candidate is 31.204 d from the observation (epoch
         2026-07-10T21:20:35 UTC, from SatChecker per-satellite, provider
         celestrak) — rejected by remote_tle_max_age_days=3
```

There are three remedies, and each of them is an explicit, recorded decision:

* supply an acceptable TLE for that satellite in a local directory and pass
  `--extra-tle-dir <dir>` (see *Supplying TLEs manually*, below);
* deliberately change `satellites.remote_tle_max_age_days` (`null` removes the
  ceiling entirely — an expert opt-out, not a default); or
* remove that NORAD ID from `satellites.norad_ids`.

TABASCAL never silently shrinks the satellite model. A satellite quietly dropped
from the RFI model degrades subtraction with no signal in the output, and editing
the configured list instead makes the changed scientific model visible and
reproducible.

## How old a TLE may be

Three settings govern TLE age and they are not interchangeable — two are limits
that can fail a run, the third is a goal that never can:

| Setting | Kind | Applies to | Default |
|---|---|---|---|
| `extra_tle_max_age_days` | limit | Your own files in `extra_tle_dir` | `null` (unlimited) |
| `remote_tle_max_age_days` | limit | SatChecker records and the managed cache | `3` days |
| `remote_tle_target_age_days` | goal | Bulk-catalogue records, which are re-requested per satellite when older | `1` day |

Only the two limits can reject a record. The target merely decides when to spend
a request trying to do better, and is covered under
[Which endpoint answered changes the TLE age](#sec-endpoint-freshness).

The remote ceiling exists because a silent stale fallback is unsafe. For an
observation on 2026-08-11 SatChecker's empty bulk response fell through to
per-satellite lookups and returned records around 31 days old; propagating those
alongside fresh ones gave roughly 9,663 km of separation for the ISS and 4 km for
a sampled NAVSTAR object. TLE age has strongly orbit-dependent consequences, but
a month-old LEO TLE must not be accepted without the user knowing.

**Three days is a provisional emergency ceiling, not a claim of three-day
positional accuracy.** It is well below the seven days an earlier proposal used
and inside the range the catalogue investigation actually evaluated. The
calibrated, observation-specific replacement — which may reject records younger
than three days, or accept older ones where justified — is
[issue #101](https://github.com/epfl-radio-astro/tabascal/issues/101); the
emergency ceiling remains a separate backstop underneath it.

Whatever the ceiling, every accepted remote record's provenance is logged:

```text
TLE remote records     : 2 accepted (limit 3 d)
  20452: managed catalogue [celestrak] epoch 2023-02-21T13:04:12 UTC, offset +0.0224 d, age 0.0224 d
  38833: managed catalogue [celestrak] epoch 2023-02-21T09:51:02 UTC, offset -0.1104 d, age 0.1104 d
```

The epoch is parsed locally from TLE line 1 — a provider's own epoch field is
never trusted — and compared against the actual mean observation epoch. Large
satellite lists get a grouped summary instead, with the oldest records still
named; set `TABASCAL_TLE_LOG_DETAIL=1` for the full per-satellite listing.

Your own `extra_tle_dir` files are exempt from the remote ceiling, so the exact
replay workflow below keeps working no matter how old the saved run is.

(sec-endpoint-freshness)=
## Which endpoint answered changes the TLE age

SatChecker's two endpoints do **not** return the same element set for the same
satellite at the same epoch, because they answer two different questions:

| Endpoint | Selects |
|---|---|
| `tles-at-epoch` (bulk) | the **newest record at or before** the requested epoch, within a 14-day lookback |
| `get-nearest-tle` (per-satellite) | the record with the **smallest \|Δt\|**, in either direction |

This is deliberate upstream — the bulk endpoint exists to answer "what elements
were current at time *T*", which look-ahead would defeat — but it is not what a
trajectory prior wants, which is "what best describes the orbit at time *T*".
Measured over the 32 GNSS satellites of the bundled performance benchmark, at
observation epoch 2023-02-21T08:03 UTC with both endpoints queried at the same
canonical bucket (09:00 UTC):

| Path | Median age | Max age | Matched the nearest available record |
|---|---:|---:|---:|
| Bulk `tles-at-epoch` | 0.595 d | 2.160 d | 10 / 32 |
| Per-satellite `get-nearest-tle` | 0.280 d | 1.063 d | **32 / 32** |

For 19 of the 32 objects the bulk catalogue's record was older than the one the
same service would return on request — by up to **1.9 days** (NORAD 40294: 1.952 d
from the bulk endpoint, 0.043 d per-satellite, a 45× difference). The
per-satellite endpoint reproduced the Space-Track `gp_history` record exactly for
all 32.

This is a property of the service, not of the bucket policy: both figures above
were taken at the *same* canonical epoch, so bucketing (≤ 1 h) cannot account for
a multi-day gap. Every one of the 22 objects where the two endpoints disagreed
had its nearest record *after* the requested epoch, exactly as the one-sided
selection rule predicts.

**The penalty is bounded**, even though the ratio is not. A bulk record's age is
at most one gap between consecutive records for that object, and at most 14 days
absolutely — beyond the lookback the object is dropped from the response rather
than returned stale. Against the nearest available record the expected cost is
about 2× in both the median and the worst case, which is what the table shows
(0.595/0.280 = 2.13, 2.160/1.063 = 2.03). The 45× outlier is one target that
landed just before a new record, not a separate effect. Well-tracked LEO objects
should fare *better* than the GNSS objects measured here, since their higher
update cadence means shorter gaps.

The selection rule is undocumented upstream; a documentation request is filed as
[iausathub/satchecker#247](https://github.com/iausathub/satchecker/issues/247).
Full analysis, including the source references it was derived from, is in
`investigations/satchecker-endpoint-freshness.md` in the repository.

**What TABASCAL does about it.** The bulk catalogue is fetched first — one
multi-megabyte download instead of one request per satellite — and then any record
older than `remote_tle_target_age_days` (default 1 day) is re-requested from the
per-satellite endpoint, in the same pass that fills IDs the bulk response missed.
On the benchmark above that costs 11 requests instead of 32 and brings the worst
case to 1.06 d, matching what querying every satellite individually would achieve.

An upgrade can only improve a record. If the per-satellite endpoint has nothing
fresher, cannot answer, or offers something beyond `remote_tle_max_age_days`, the
existing record is kept — a declined upgrade can never turn a complete resolution
into a failure. Upgraded records are cached alongside the snapshot, so the
requests are paid once per canonical epoch, not once per run.

Setting `remote_tle_target_age_days: null` disables the pass and restores
pure bulk-first behaviour. To bypass both endpoints entirely, supply the elements
yourself through `extra_tle_dir` (see [below](#sec-manual-tles)).

## Caching behaviour by scenario

The managed cache lives in the platform user-cache directory (e.g.
`~/.cache/tle-cache` on Linux, `~/Library/Caches/tle-cache` on macOS), or
`TLE_CACHE_DIR` if set. (`TLE_CACHE_DIR` relocates that *storage*; it is not an
additional TLE source, which is what `--extra-tle-dir` is.) Snapshots are keyed
by a *canonical epoch*: the midpoint of the fixed UTC bucket containing the
observation, so the snapshot used depends only on the observation time and bucket
width — never on what happens to be cached already.

| Scenario | What happens | Later runs |
|---|---|---|
| Settled epoch (older than `tle_catalogue_settle_days`), catalogue available | Full catalogue downloaded once, stored as `catalogue-<stamp>.json`. Its records may be older than `get-nearest-tle` would return — see [above](#sec-endpoint-freshness) | Served from cache **forever** — deterministic, never refreshed |
| Unsettled epoch (newer than `tle_catalogue_settle_days`, or in the future) | Catalogue downloaded and stored only as `catalogue-<stamp>-provisional.json` | Reused for `tle_provisional_cache_hours` (default 12 h), then refetched. **Never promoted** to a stable snapshot: once the epoch settles, the next run downloads a fresh one |
| Recent epoch beyond SatChecker's data horizon (catalogue empty) | Per-satellite fallback; records stored as `catalogue-<stamp>-extra-provisional.json`; **no snapshot is stored** | The catalogue is re-attempted on **every** run, and the fallback records expire after `tle_provisional_cache_hours`; once SatChecker backfills the epoch, the snapshot is fetched and takes precedence |
| Catalogue contains malformed rows or fewer than 99% of the expected rows remain valid | Malformed rows are rejected. An incomplete catalogue is **not cached**, and the requested satellites are fetched individually | The full catalogue is re-attempted on every run |
| Satellite missing from a settled epoch's snapshot | Per-satellite lookup for that ID, cached in the `-extra` file | Reused from the `-extra` cache; **not refreshed** even if SatChecker later adds the satellite to the catalogue |
| Snapshot record older than `remote_tle_target_age_days` | Per-satellite lookup replaces it if the service has something nearer, cached in the same `-extra` file | Reused from cache — the upgrade requests are paid once per canonical epoch, not once per run |
| Service reachable but the catalogue response is unusable (malformed, truncated, or an HTTP 4xx) | The requested satellites are fetched individually instead | The full catalogue is re-attempted on every run |
| Service rate-limits (HTTP 429) or fails server-side (HTTP 5xx) | Run fails fast — answering either with one request per satellite would make it worse | — |
| Service unreachable, snapshot cached | Cache hit — run proceeds offline | — |
| Service unreachable, no snapshot | Run fails fast with a clear error (no satellite-by-satellite retry storm) | — |
| Cache directory read-only or out of quota | A warning names the path; the run proceeds on the validated records it already fetched | Nothing is cached, so the next run fetches again |

Several consequences are worth understanding:

- **Recent catalogues are provisional.** SatChecker's ingest was observed ramping
  from 9 rows at an observation age of 16 days to 20,458 at 17 days, 26,600 at 18
  days and 31,108 at 30 days. A response can therefore be complete against its own
  `total_results` while the upstream catalogue is still filling, so transport
  completeness is not evidence that a recent catalogue is settled. The 45-day
  default is an **observed defensive policy**, not a SatChecker API guarantee —
  and it says nothing about whether a 45-day-old TLE is scientifically usable,
  which is `remote_tle_max_age_days`' separate job. Per-satellite fallback
  records follow the same policy as the snapshot they stand in for: an unsettled
  epoch reaches the per-satellite endpoint *precisely because* its bulk catalogue
  is empty, so caching those responses permanently would exempt the one result
  the settling policy exists to revisit. (TLE age is measured against the fixed
  observation epoch, so nothing else would ever make such a record stale.)
- **Determinism wins over freshness.** Once a *stable* snapshot exists for a
  bucket it is never refreshed, even if SatChecker later serves better
  (closer-epoch) records for that time. This is deliberate: rerunning an analysis
  gives the same trajectory priors. To force a refresh, delete the relevant
  `catalogue-<stamp>.json` / `catalogue-<stamp>-extra.json` files from the cache
  directory (or point `TLE_CACHE_DIR` at a fresh directory) — the next run
  re-downloads. The exact `<stamp>` is printed in the run log wherever a
  catalogue epoch is mentioned.
- **The empty-catalogue fallback self-heals.** Because an empty catalogue is
  never stored, a recent observation first processed with per-satellite TLEs will
  automatically pick up the proper catalogue snapshot once SatChecker's ingest
  catches up — improving the priors on a rerun. If you need the *original* run's
  priors instead, use the saved run TLEs (next section).
- **Invalid cache files do not become trusted inputs.** Managed snapshots and
  per-satellite fallback files are checked for their schema, canonical epoch,
  stable/provisional state, record counts, completeness, TLE syntax, and matching
  satellite identities. A corrupt, incomplete, wrong-epoch or wrong-state file is
  treated as a cache miss and the service is consulted again. The cache schema
  version was bumped for the provisional policy, so snapshots written by earlier
  builds are ignored rather than becoming trusted by aging in place.

In a distributed TABASCAL run, process 0 performs the entire resolution — both
the preflight check and the element fetch — and broadcasts the resulting TLE
lines to every worker, so the service sees exactly one fetch per run and every
process models the same satellites. This matters in two ways: without it every
rank would download the catalogue independently on a cache miss (or whenever the
shared cache cannot be written, where workers find nothing to read), and ranks
could reach *different* coverage verdicts, leaving some to exit while others go
on to a collective and hang. If process 0's resolution fails, every process stops
with the same error. Only process 0 writes the shared
`used_tles` result, and any synthetic satellite entries added solely for device
padding are excluded from that file. Cache files are written atomically; however,
unrelated TABASCAL jobs writing the same per-satellite fallback file at the same
time are not serialized with each other.

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
resolves from the saved file and no service call is made. Replay is deliberately
exempt from `remote_tle_max_age_days`: the remote ceiling is a policy about what
the *service* may hand you, and it must never stop you reproducing a run you
already have the records for.

(sec-manual-tles)=
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
   the columns `NORAD_CAT_ID`, `TLE_LINE1` and `TLE_LINE2`. Every `*.json` file
   in the directory is considered; the filename does not need to contain a
   date. Space-Track's JSON output already uses these column names (the full
   column set is documented below); orbital-element columns are ignored —
   TABASCAL parses the elements locally from the two TLE lines. Both lines must
   form a valid TLE pair and must encode the same satellite as `NORAD_CAT_ID`.
   Unreadable files, files without the required columns, and invalid records are
   skipped; an unresolved satellite then falls through to the managed cache and
   SatChecker. Space-Track also covers objects SatChecker does not: full-catalogue
   comparisons at four representative epochs found 92–96% of Space-Track's IDs
   present in SatChecker within three days of the epoch, and SatChecker exposed
   no Alpha-5 objects in any sampled catalogue while Space-Track had 163–295
   recent ones. Optional built-in Space-Track support is
   [issue #101](https://github.com/epfl-radio-astro/tabascal/issues/101); until
   then this manual route is how you fill those gaps.

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
