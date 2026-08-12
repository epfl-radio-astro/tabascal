# Two-Line Elements (TLEs)

TABASCAL obtains historical satellite orbital elements from the
[IAU CPS SatChecker](https://satchecker.cps.iau.org/) service. No account or
credentials are required.

## Resolution order

Each configured NORAD ID is resolved independently, in this order:

1. `extra_tle_dir`: explicit user or replay files. The valid record closest to
   the observation is selected and checked against `extra_tle_max_age_days`.
2. Managed per-NORAD cache: the validated cached record closest to the
   observation is selected. If its age is within
   `tle_cache_reuse_max_age_days`, no network request is made.
3. SatChecker `get-nearest-tle`: TABASCAL requests the nearest record at the
   exact observation epoch. Cache misses run with at most five requests in
   flight, and valid responses are added to the per-NORAD cache.

TABASCAL does not use SatChecker's full-catalogue endpoint. That endpoint selects
the newest TLE at or before the requested epoch, whereas TABASCAL requires the
TLE whose epoch is closest on either side.

Every configured satellite must resolve to an acceptable TLE. The check runs
during preflight, before the visibilities are read. A missing or over-age record
stops the run rather than silently shrinking the RFI model.

## Cache policy

The managed cache normally lives in the platform user-cache directory, such as
`~/.cache/tle-cache` on Linux or `~/Library/Caches/tle-cache` on macOS. Set
`TLE_CACHE_DIR` to relocate it. This variable controls managed storage; it is not
an additional source like `--extra-tle-dir`.

Each `tle-<NORAD ID>.json` file is an atomically written, versioned envelope
containing the validated immutable TLE records already learned for that object.
Records are keyed by their contents and TLE epoch rather than by the observation
that originally requested them, so one record can serve multiple nearby runs.

Two age settings have intentionally different jobs:

| Setting | Purpose | Default |
|---|---|---:|
| `tle_cache_reuse_max_age_days` | Avoid a request when the nearest cached record is already this close to the observation | 1 day |
| `remote_tle_max_age_days` | Hard safety ceiling for every SatChecker or managed-cache record | 3 days |

If a cached record is older than the reuse threshold but still inside the hard
ceiling, TABASCAL asks SatChecker for something closer. If that request fails or
returns nothing usable, the acceptable cached record is used as an offline
fallback. Records outside the hard ceiling are never accepted automatically.

`tle_cache_reuse_max_age_days: null` means always reuse the nearest acceptable
cached record. `remote_tle_max_age_days: null` is a separate expert opt-out that
removes the safety ceiling. When both are numeric, the reuse threshold must not
exceed the ceiling.

Cache reads validate the schema, NORAD identity, TLE line width and checksums,
embedded line identities, epoch, and all orbital fields consumed downstream. A
missing, partial, corrupt, or incompatible file is treated as a cache miss.
Cache-write failures are warnings: TABASCAL continues with validated records in
memory, but a later run will need to fetch them again.

## TLE age and suitability

The TLE epoch is always parsed locally from line 1; SatChecker's separate epoch
metadata is not trusted for acceptance. Logs report each selected provider,
epoch, signed offset, and absolute age.

The three-day default is a provisional emergency backstop, not a claim that a
three-day-old TLE is scientifically adequate. Position error depends strongly on
orbit, maneuvers, baseline, wavelength, and the intended phase accuracy. The
observation-specific replacement is tracked in
[issue #101](https://github.com/epfl-radio-astro/tabascal/issues/101).

## Exact replay

Every run saves the exact TLE pairs it used to
`<sim_dir>/results/used_tles_<name>.json`. To reproduce those trajectory priors,
copy or retain that file and pass its directory to a later run:

```bash
tabascal run -c path/to/config.yaml -ms path/to/data.ms --extra-tle-dir /path/to/saved-run
```

The default `extra_tle_max_age_days: null` deliberately exempts explicit replay
files from the remote age ceiling.

## Supplying TLEs manually

Use `--extra-tle-dir` when SatChecker lacks an object or an acceptable historical
record. Every `*.json` file in the directory is considered. Files must be
JSON tables containing:

- `NORAD_CAT_ID`
- `TLE_LINE1`
- `TLE_LINE2`

`OBJECT_NAME` is optional. A JSON array of objects is the clearest format to use:

```json
[
  {
    "NORAD_CAT_ID": 25544,
    "OBJECT_NAME": "ISS (ZARYA)",
    "TLE_LINE1": "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "TLE_LINE2": "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
  }
]
```

A file may contain multiple records for the same object and records for multiple
objects. TABASCAL validates both 69-column lines, checks their modulo-10
checksums and embedded NORAD IDs, parses the line-1 epoch, and selects the valid
record closest to the observation independently for each requested ID. The
numeric `NORAD_CAT_ID` must agree with the identifier encoded in both lines.
Invalid rows do not poison unrelated satellites; the unresolved ID falls
through to the managed cache and SatChecker.

The column-oriented JSON written by `pandas.DataFrame.to_json()` is also
accepted. In particular, TABASCAL's `used_tles_<name>.json` replay files can be
placed directly in this directory. Other files and subdirectories are ignored;
only `*.json` files immediately inside `extra_tle_dir` are read.

### Obtaining compatible JSON from Space-Track

Space-Track's `gp` and `gp_history` JSON responses include `NORAD_CAT_ID`,
`OBJECT_NAME`, `TLE_LINE1`, and `TLE_LINE2`, so their response bodies can be
saved directly in `extra_tle_dir` without conversion. Use `gp` for the current
element set and `gp_history` for historical element sets. Space-Track requires a
free account and authenticated requests; see its
[API documentation](https://www.space-track.org/documentation#api) and
[GP field definition](https://www.space-track.org/basicspacedata/modeldef/class/gp/format/html).

For example, the following logs in, downloads ISS element sets whose epochs fall
between 20 and 22 February 2023, and logs out. Credentials are read from
environment variables rather than written into the command or output file:

```bash
export SPACETRACK_USER='your-email@example.com'
read -r -s -p 'Space-Track password: ' SPACETRACK_PASSWORD
export SPACETRACK_PASSWORD

cookie_jar=$(mktemp)
curl --fail --silent --show-error \
  --cookie-jar "$cookie_jar" \
  --data-urlencode "identity=${SPACETRACK_USER}" \
  --data-urlencode "password=${SPACETRACK_PASSWORD}" \
  https://www.space-track.org/ajaxauth/login

curl --fail --silent --show-error \
  --cookie "$cookie_jar" \
  'https://www.space-track.org/basicspacedata/query/class/gp_history/norad_cat_id/25544/EPOCH/2023-02-20--2023-02-22/orderby/EPOCH%20asc/format/json' \
  --output iss-history.json

curl --fail --silent --show-error \
  --cookie "$cookie_jar" \
  https://www.space-track.org/ajaxauth/logout
rm -f "$cookie_jar"
unset SPACETRACK_PASSWORD
```

Put `iss-history.json` in the directory passed to `--extra-tle-dir`. To obtain
the current element set instead, replace the query URL with:

```text
https://www.space-track.org/basicspacedata/query/class/gp/norad_cat_id/25544/format/json
```

Replace `25544` with the required numeric NORAD catalogue ID. Space-Track's
published usage policy says not to use `gp_history` for current ephemerides and
to download a historical object or range once and retain it locally. For many
objects or large date ranges, use the historical bulk files Space-Track provides
rather than repeatedly querying `gp_history`.

This is a manual interoperability path only; TABASCAL does not store
Space-Track credentials or query Space-Track itself.

## Distributed runs

Only process 0 resolves TLEs and performs network requests. Its complete
resolution—including the chosen lines, provenance, epochs, and coverage
decision—is broadcast to every process. Workers parse the elements locally from
the identical TLE lines. This prevents duplicate requests and ensures that all
processes either use the same satellite set or fail coherently.
