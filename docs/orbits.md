# Satellite orbit records

TABASCAL obtains satellite orbital elements from the
[IAU CPS SatChecker](https://satchecker.cps.iau.org/) service. No account or
credentials are required.

The transport, record parsing, validation, and cache storage live in the
[satchecker-client](https://satchecker-client.readthedocs.io/) package,
extracted from TABASCAL and documented on its own. This page documents what
TABASCAL decides on top of it: which source a satellite resolves from, how
stale a record may be, and what happens when coverage is incomplete.

## Two archives, one handover

SatChecker keeps satellite orbits in two formats, and which one you get depends
on when your observation was.

| Archive | Endpoint | Covers |
|---|---|---|
| TLE | `get-nearest-tle` | up to **2026-07-11**, frozen |
| OMM | `get-nearest-omm` | from **2026-07-12** onwards, growing |

The two do not overlap: the last TLE and the first OMM are about twelve hours
apart, and the TLE archive will never gain another record. A **TLE** (Two-Line
Element set) encodes an orbit in two fixed-width 69-column lines; an **OMM**
(Orbit Mean-Elements Message) carries the same orbital elements as named
numeric fields. Both describe the same SGP4 model, and TABASCAL derives the
same element set from either, so nothing downstream of resolution cares which
one a satellite resolved to. The
[satchecker-client guide](https://satchecker-client.readthedocs.io/en/latest/usage.html#two-archives-one-handover)
covers the split, and why it happened, in more detail.

You do not choose between them. TABASCAL asks the archive your observation
epoch falls in, and falls back to the other if that one has nothing usable.

## Resolution order

Each configured NORAD ID is resolved independently, in this order:

1. `extra_orbit_dir`: explicit user or replay files, of either kind. The valid
   record closest to the observation is selected and checked against
   `extra_orbit_max_age_days`.
2. Managed per-NORAD cache: the validated cached record closest to the
   observation is selected. If its age is within `cache_reuse_max_age_days`, no
   network request is made.
3. SatChecker: TABASCAL requests the nearest record at the exact observation
   epoch. Cache misses run with at most five requests in flight, and valid
   responses are added to the per-NORAD cache.

TABASCAL does not use SatChecker's full-catalogue endpoints. Those select the
newest record at or before the requested epoch, whereas TABASCAL requires the
record whose epoch is closest on either side.

### Which endpoint, and the fallback

The observation epoch picks the endpoint to ask first — `get-nearest-omm` for an
observation on or after 2026-07-12, `get-nearest-tle` before it. In the common
case that is the whole story: one request per satellite, answered from the right
archive.

If that request yields nothing acceptable, the other endpoint is asked for the
IDs the first archive did not answer with an acceptable record. Note that this
is not the same as the IDs left *unresolved*: a satellite whose cached record is
stale but still inside the hard ceiling is already resolved from the cache
before any request goes out, and something closer is the entire point of asking.
It reaches the fallback too. The fallback exists because **neither endpoint
reports that it has nothing near the epoch you asked for**. Ask `get-nearest-omm` for a
2021 epoch and it returns its earliest 2026 record — years off, with nothing in
the response to say so. Ask `get-nearest-tle` for a 2027 epoch and it returns
the last TLE ever published. TABASCAL's age ceiling rejects both, and that
rejection is exactly the signal that the record you want lives in the other
archive. For an observation within a few days of the handover this is the normal
path, not an exceptional one.

That is also why the handover date is a *hint* rather than a cutoff. SatChecker
now sources OMM from Space-Track as well as Celestrak, and Space-Track's OMM
history runs years deep, so OMM may yet appear for earlier epochs. A hardcoded
cutoff would keep silently preferring a stale TLE; with the fallback, a backfill
costs one extra request instead of a worse answer. The date is not a
configuration key — it is a property of the service, not of your run.

A **failure** is different from an unusable answer. If the service cannot be
reached, returns HTTP 429, or rejects every request alike, TABASCAL does not try
the other endpoint: the service is down, and asking a down service a different
question is still asking a down service.

## Being a considerate client

TABASCAL talks to SatChecker through satchecker-client, whose
[batching](https://satchecker-client.readthedocs.io/en/latest/usage.html#batches)
is written to be considerate of a free public service: requests go out at most
five at a time, submitted incrementally as earlier ones land; an unreachable
service or an HTTP 429 stops the batch at a cost of at most five requests no
matter how many satellites are configured; a malformed reply or a 4xx rejection
of one request is one satellite's problem rather than the batch's; and ten
identical consecutive rejections with no success in between are recognised as a
service-level wall rather than ten absent satellites. TABASCAL adds its own
restraint on top: after any such service-level failure there is no second round
against the other endpoint (as above), and every response that can be reused is
written to the local cache so a later run does not ask again.

Whatever the cause, every configured ID is reported, and the resulting error
distinguishes a satellite with no record from one TABASCAL could not ask about.
When the service supplies a `Retry-After` hint it is carried into that error, so
you know when the run is worth repeating — TABASCAL reports the wait rather than
sleeping through it, so an unattended preflight never blocks for an interval it
did not choose.

Every configured satellite must resolve to an acceptable record. The check runs
during preflight, before the visibilities are read. A missing or over-age record
stops the run rather than silently shrinking the RFI model.

## Cache policy

The managed cache normally lives in the platform user-cache directory, such as
`~/.cache/orbit-cache` on Linux or `~/Library/Caches/orbit-cache` on macOS. Set
`ORBIT_CACHE_DIR` to relocate it. This variable controls managed storage; it is
not an additional source like `--extra-orbit-dir`.

The store itself — an atomically written, versioned `orbit-<NORAD ID>.json`
envelope per satellite, validated on every read — is satchecker-client's
[`TextOrbitCache`](https://satchecker-client.readthedocs.io/en/latest/usage.html#caching).
Records are keyed by their contents and epoch rather than by the observation
that originally requested them, so one record can serve multiple nearby runs.

**One file holds both kinds.** Around the handover a satellite will typically
have both its last TLEs and its first OMM records in the same file; which one a
given observation uses is decided by epoch distance, not by format.

Two age settings have intentionally different jobs:

| Setting | Purpose | Default |
|---|---|---:|
| `cache_reuse_max_age_days` | Avoid a request when the nearest cached record is already this close to the observation | 1 day |
| `remote_max_age_days` | Hard safety ceiling for every SatChecker or managed-cache record | 3 days |

If a cached record is older than the reuse threshold but still inside the hard
ceiling, TABASCAL asks SatChecker for something closer — including the fallback
archive, if the first one has nothing to offer for that satellite. A response
replaces the cached record only when it is strictly closer to the observation,
so a refresh can never downgrade what you already hold. If the request fails or
returns nothing usable, the acceptable cached record is used as an offline
fallback. Records outside the hard ceiling are never accepted automatically.

`cache_reuse_max_age_days: null` means always reuse the nearest acceptable
cached record. `remote_max_age_days: null` is a separate expert opt-out that
removes the safety ceiling. When both are numeric, the reuse threshold must not
exceed the ceiling.

Cache reads validate the schema, NORAD identity and every field consumed
downstream — see [Validation](#validation) for what that means per kind. A
missing, partial, corrupt, or incompatible file is treated as a cache miss; a
file that exists but cannot be used is also reported, so a cache that never
takes hold does not silently cost a request every run. Cache-write failures are
warnings: TABASCAL continues with validated records in memory, but a later run
will need to fetch them again.

Cache files written by a TABASCAL that predates OMM support (schema version 1)
are reported as unusable and replaced by the next fetch. Nothing needs
converting; you may see one warning per satellite on the first run after
upgrading.

(validation)=
## Validation

Record validation is satchecker-client's job, and its guide spells out
[what is guaranteed per kind](https://satchecker-client.readthedocs.io/en/latest/usage.html#what-validation-guarantees-per-kind):
the element range and finiteness checks both kinds share, a TLE's modulo-10
checksums and embedded-identity cross-check, and the epoch plausibility window
that partly stands in for both on a checksum-less OMM. The short version is
that the two formats do not offer the same guarantees, an OMM's are weaker,
and that is a property of the format rather than of the handling.

One consequence matters enough to repeat here. A TLE's epoch is always
re-derived from line 1; a provider's own epoch field is never trusted for
acceptance. An OMM has no lines to re-derive from, so its `EPOCH` field must be
taken at face value — and, as described above, a clamped `get-nearest-omm`
response is precisely the case where a wrong epoch would otherwise be
invisible. The age ceiling is therefore doing more work for OMM records than it
ever did for TLEs.

Logs report each selected provider, epoch, signed offset, absolute age, and
which endpoint answered.

## Record age and suitability

The three-day default is a provisional emergency backstop, not a claim that a
three-day-old element set is scientifically adequate. Position error depends
strongly on orbit, maneuvers, baseline, wavelength, and the intended phase
accuracy. The observation-specific replacement is tracked in
[issue #101](https://github.com/epfl-radio-astro/tabascal/issues/101).

## Exact replay

Every run saves the exact records it used to
`<sim_dir>/results/used_orbits_<name>.json`. To reproduce those trajectory
priors, copy or retain that file and pass its directory to a later run:

```bash
tabascal run -c path/to/config.yaml -ms path/to/data.ms --extra-orbit-dir /path/to/saved-run
```

The default `extra_orbit_max_age_days: null` deliberately exempts explicit
replay files from the remote age ceiling.

The file records each entry in whatever form it needs to be read back as itself:
a TLE's two lines, or an OMM's epoch and elements. Derived quantities are not
stored — they are recomputed on every read, so a stored copy could only ever
drift out of agreement with the elements it came from.

## Supplying records manually

Use `--extra-orbit-dir` when SatChecker lacks an object or an acceptable
historical record. Every `*.json` file in the directory is considered. Files
must be JSON tables carrying either kind's required columns.

For a TLE:

- `NORAD_CAT_ID`
- `TLE_LINE1`
- `TLE_LINE2`

For an OMM:

- `NORAD_CAT_ID`
- `EPOCH` (ISO 8601)
- `INCLINATION`, `RA_OF_ASC_NODE`, `ECCENTRICITY`, `ARG_OF_PERICENTER`,
  `MEAN_ANOMALY`, `MEAN_MOTION`, `BSTAR`

`OBJECT_NAME` and `OBJECT_ID` are optional. A JSON array of objects is the
clearest format to use:

```json
[
  {
    "NORAD_CAT_ID": 25544,
    "OBJECT_NAME": "ISS (ZARYA)",
    "TLE_LINE1": "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "TLE_LINE2": "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
  },
  {
    "NORAD_CAT_ID": 25544,
    "OBJECT_NAME": "ISS (ZARYA)",
    "EPOCH": "2026-08-13T03:34:14.082240",
    "INCLINATION": 51.6324,
    "RA_OF_ASC_NODE": 18.1827,
    "ECCENTRICITY": 0.0007533,
    "ARG_OF_PERICENTER": 41.6914,
    "MEAN_ANOMALY": 318.4648,
    "MEAN_MOTION": 15.49426097,
    "BSTAR": 7.5606e-05
  }
]
```

You do not need to say which kind a record is. TABASCAL infers it: a record with
TLE lines is a TLE, a record with the element columns is an OMM. (A record may
declare `RECORD_KIND` explicitly, and files TABASCAL writes itself do, but no
external export carries such a field and none needs to.) A record carrying both
— as Space-Track's exports do — is read as a TLE, because its lines are the
stronger thing to validate against.

A directory that does not exist is reported as a warning rather than silently
searched, so a typo in the path cannot quietly leave you modelling SatChecker's
satellites instead of your own.

A file may contain multiple records for the same object, records for multiple
objects, and a mixture of kinds. TABASCAL validates each record as described
under [Validation](#validation) and selects the valid record closest to the
observation independently for each requested ID — by epoch distance, regardless
of format. Invalid rows do not poison unrelated satellites; the unresolved ID
falls through to the managed cache and SatChecker.

The column-oriented JSON written by `pandas.DataFrame.to_json()` is also
accepted. In particular, TABASCAL's `used_orbits_<name>.json` replay files can
be placed directly in this directory. Other files and subdirectories are
ignored; only `*.json` files immediately inside `extra_orbit_dir` are read.

### Obtaining compatible JSON from Space-Track

Space-Track's `gp` and `gp_history` JSON responses include `NORAD_CAT_ID`,
`OBJECT_NAME`, `TLE_LINE1`, `TLE_LINE2` and the OMM element fields, so their
response bodies can be saved directly in `extra_orbit_dir` without conversion.
Use `gp` for the current element set and `gp_history` for historical element
sets. Space-Track requires a free account and authenticated requests; see its
[API documentation](https://www.space-track.org/documentation#api) and
[GP field definition](https://www.space-track.org/basicspacedata/modeldef/class/gp/format/html).

This is the practical route to a pre-handover epoch that SatChecker's OMM
archive cannot serve, and to a post-handover object with no TLE representation.

For example, the following logs in, downloads ISS element sets whose epochs fall
between 20 and 22 February 2023, and logs out. The object, epoch range and
output file are set once at the top so the rest can be reused unchanged. The
password is prompted for at run time and cleared afterwards rather than being
hard-coded or written into the output file:

```bash
# --- parameters -------------------------------------------------------------
NORAD_ID='25544'              # numeric NORAD catalogue ID (25544 = ISS)
EPOCH_START='2023-02-20'      # UTC
EPOCH_STOP='2023-02-22'       # UTC
OUTPUT='iss-history.json'
SPACETRACK_USER='your-email@example.com'
# ----------------------------------------------------------------------------

BASE='https://www.space-track.org/basicspacedata/query'
QUERY_URL="${BASE}/class/gp_history/norad_cat_id/${NORAD_ID}/EPOCH/${EPOCH_START}--${EPOCH_STOP}/orderby/EPOCH%20asc/format/json"

printf 'Space-Track password: '
read -r -s SPACETRACK_PASSWORD
printf '\n'

cookie_jar=$(mktemp)
curl --fail --silent --show-error \
  --cookie-jar "$cookie_jar" \
  --data-urlencode "identity=${SPACETRACK_USER}" \
  --data-urlencode "password=${SPACETRACK_PASSWORD}" \
  https://www.space-track.org/ajaxauth/login

curl --fail --silent --show-error \
  --cookie "$cookie_jar" \
  "$QUERY_URL" \
  --output "$OUTPUT"

curl --fail --silent --show-error \
  --cookie "$cookie_jar" \
  https://www.space-track.org/ajaxauth/logout
rm -f "$cookie_jar"
unset SPACETRACK_PASSWORD
```

Put `iss-history.json` in the directory passed to `--extra-orbit-dir`.

Notes for adapting this into a script of your own:

- `EPOCH_START` and `EPOCH_STOP` are UTC and both ends are inclusive. A bare
  date means midnight, so the window above ends at `2023-02-22 00:00:00`; append
  a time as `2023-02-22%2012:00:00` for finer control, keeping the `%20`
  encoding since the value sits in the URL path.
- `NORAD_ID` also accepts a comma-separated list — `'25544,48274'` — to fetch
  several objects into one file. TABASCAL reads multi-object files, and
  `OUTPUT` is set independently so the filename need not track the IDs.
- To obtain the current element set instead of a historical range, leave
  everything else alone and use the `gp` class, which takes no epoch range:

  ```bash
  QUERY_URL="${BASE}/class/gp/norad_cat_id/${NORAD_ID}/format/json"
  ```

- If you save this as a script rather than pasting it into a shell, add
  `set -eu` at the top so a failed login does not fall through to the query.
  It is deliberately absent above, because `set -e` in an interactive shell
  closes the terminal on the first error.

Space-Track's published usage policy says not to use `gp_history` for current
ephemerides and to download a historical object or range once and retain it
locally. For many objects or large date ranges, use the historical bulk files
Space-Track provides rather than repeatedly querying `gp_history`.

This is a manual interoperability path only; TABASCAL does not store
Space-Track credentials or query Space-Track itself.

## Distributed runs

Only process 0 resolves satellites and performs network requests. Its complete
resolution — including the chosen records, provenance, epochs, and coverage
decision — is broadcast to every process, and workers derive the orbital
elements locally. This prevents duplicate requests and ensures that all
processes either use the same satellite set or fail coherently.

For a TLE, "locally" means every rank re-parses the same two lines with the same
parser, so the results are bit-identical by construction. An OMM has no lines to
re-parse, so its element values themselves cross the broadcast, as JSON numbers
rather than as text — which round-trips exactly in Python 3. Ranks that
disagreed here would hold subtly different trajectory priors with nothing
raising anywhere, so this is asserted by test as exact equality rather than
approximate.
