# Investigation brief: SatChecker returns different TLEs per endpoint

**Status:** open — reproduced against the live service, cause not yet identified
**Raised by:** PR #92 (SatChecker TLE retrieval), 2026-08-12
**Relevant issue:** [#101](https://github.com/epfl-radio-astro/tabascal/issues/101) (calibrated TLE suitability)
**Audience:** whoever picks this up next, human or agent. It assumes no prior
context on this repository.

This brief is deliberately outside `docs/`: the documentation build runs with
`-W` (warnings as errors), so every file under `docs/` must appear in a toctree,
and this is working material rather than user documentation.

---

## 1. The observation

For the same satellite, at the same epoch, the IAU CPS SatChecker service returns
**different element sets depending on which endpoint you ask**:

- `GET /tools/tles-at-epoch/` — bulk catalogue, one record per object near an epoch
- `GET /tools/get-nearest-tle/` — single object, nearest record to an epoch

`get-nearest-tle` is consistently equal to or fresher than `tles-at-epoch`, never
worse. Measured over 32 GNSS satellites at observation epoch
`2023-02-21T08:03:26 UTC`, both endpoints queried at the same canonical bucket
epoch `2023-02-21T09:00:00 UTC`:

| Path | Median age | Max age | Matched nearest-available |
|---|---:|---:|---:|
| `tles-at-epoch` (bulk) | 0.595 d | 2.160 d | 10 / 32 |
| `get-nearest-tle` (per-ID) | 0.280 d | 1.063 d | 32 / 32 |

19 of 32 objects were staler from the bulk endpoint. Worst cases:

| NORAD | bulk age | per-ID age | ratio |
|---|---:|---:|---:|
| 40294 | 1.952 d | 0.043 d | 45× |
| 21890 | 2.160 d | 0.450 d | 4.8× |
| 40105 | 1.325 d | 0.171 d | 7.7× |
| 43057 | 2.034 d | 0.899 d | 2.3× |
| 43565 | 1.696 d | 0.651 d | 2.6× |

The per-satellite endpoint reproduced the Space-Track `gp_history` record
**byte-for-byte for all 32** (compared against `tabascal/data/tles/*.json`, which
were pulled from Space-Track). So `get-nearest-tle` appears to be returning the
true nearest record, and `tles-at-epoch` something else.

## 2. What has already been ruled out

Do not re-derive these; they cost live requests.

- **Not the canonical bucket.** Both figures above come from the *same* epoch
  argument (09:00 UTC). Bucketing can contribute at most `interval/2` = 1 h; the
  observed gaps reach 1.9 days.
- **Not tabascal's record selection.** When a bulk response carries several rows
  for one NORAD ID, `_select_from_records` picks the row nearest the reference
  epoch. The bulk responses here carried a *single* row per object, so there was
  nothing to select between.
- **Not tabascal's parsing.** Ages are computed from TLE line 1 columns 19–32 by
  `tabascal/satchecker/tle_parse.py:tle_epoch_jd`, the same function on both
  paths. The comparison is on raw `TLE_LINE1` strings, not derived elements.
- **Not a provider split.** Every record on both paths reported
  `data_source: spacetrack`. This is not CelesTrak-vs-Space-Track.
- **Not a stale local cache.** The measurement used a throwaway `TLE_CACHE_DIR`.

## 3. Reproduction

Requires network. Roughly 35 requests plus one ~5 MB catalogue download; the
script throttles per-ID calls at 1 s as the client does.

```bash
# Test data comes from the HuggingFace dataset the pipeline tests use:
#   epfl-radio-astro/rfi-simulations, the 32SAT / 96A simulation
export TLE_CACHE_DIR=$(mktemp -d)
pixi run -e dev python - <<'PY'
import time
from tabascal import satchecker
from tabascal.satchecker.cache import canonical_epoch_jd
from tabascal.satchecker.tle_parse import tle_epoch_jd

OBS   = 2459996.8357189964          # 2023-02-21T08:03:26.121 UTC
CANON = canonical_epoch_jd(OBS)     # 2023-02-21T09:00:00 UTC

bulk = satchecker.fetch_full_catalogue(CANON).records
bulk = bulk.set_index(bulk["NORAD_CAT_ID"].astype(int))

for nid in (40294, 21890, 40105, 43057, 43565):
    b = abs(tle_epoch_jd(bulk.loc[nid, "TLE_LINE1"]) - OBS)
    time.sleep(1.0)
    rec = satchecker.fetch_nearest_tle(nid, CANON)
    p = abs(tle_epoch_jd(rec["TLE_LINE1"].iloc[0]) - OBS)
    print(f"{nid}: bulk {b:.3f} d, per-ID {p:.3f} d, ratio {b/p:.1f}x")
PY
```

Expect roughly the table in §1. To confirm the endpoint difference *without*
tabascal in the path, hit the HTTP API directly — this is the form the client
uses (`tabascal/satchecker/client.py`), note `epoch` is a bare Julian date:

```bash
curl -s 'https://satchecker.cps.iau.org/tools/get-nearest-tle/?id=40294&id_type=catalog&epoch=2459996.875'
curl -s 'https://satchecker.cps.iau.org/tools/tles-at-epoch/?epoch=2459996.875&format=json&per_page=1&page=1'
```

## 4. Hypotheses to test, most likely first

1. **`tles-at-epoch` is a materialised/precomputed snapshot.** It may be built on
   a schedule (nightly? per-epoch job?) from whatever was ingested at build time,
   rather than queried live. That would explain a systematic lag and why the lag
   varies per object.
2. **Different query semantics.** `get-nearest-tle` may do `ORDER BY abs(epoch -
   target) LIMIT 1`, while `tles-at-epoch` does something cheaper over the whole
   catalogue — e.g. "most recent record with `epoch <= target`", or a join
   against a per-object "current TLE" table, or a coarse date-bucket match.
3. **Different source tables.** The bulk path may read a summary/latest table
   while the per-ID path reads full history.
4. **Pagination or dedup collapsing.** The bulk path must emit one row per
   object; if the collapse picks arbitrarily (e.g. `DISTINCT ON` with a
   non-epoch sort key) rather than by nearest epoch, this is exactly the symptom.
5. **Intentional.** It may be a documented performance trade-off. Check the API
   docs and release notes before filing anything.

## 5. Where to look

Source: <https://github.com/iausathub/satchecker> — Python, Flask, `src/` layout.

`src/api/` uses a ports-and-adapters structure. From the directory listing (I did
not read the code, so treat these as starting points rather than facts):

| Directory | Likely relevance |
|---|---|
| `entrypoints/` | Flask route definitions — find the `/tools/tles-at-epoch/` and `/tools/get-nearest-tle/` handlers here first |
| `services/` | The logic each route calls; the two probably diverge here |
| `adapters/` | Data access / repositories — the actual queries. **This is where hypotheses 2–4 resolve.** |
| `domain/` | Models and any epoch-selection rules |
| `migrations/` | Table shapes; reveals whether a summary/latest table exists (hypothesis 3) |
| `celery_app.py` | Background jobs — a scheduled catalogue build would live here or be triggered from it (hypothesis 1) |

Suggested order:

1. Locate both route handlers in `entrypoints/`; note the service function each calls.
2. Follow both into `services/`, then into `adapters/`. Diff the two query paths.
3. Answer: **does `tles-at-epoch` select by nearest epoch, or by something else?**
4. If a background job builds the catalogue, find its schedule and its input cut-off.
5. Check `docs/` and the repo's issue tracker for an existing description of this.

## 6. Questions this investigation should answer

1. Is `tles-at-epoch` intentionally coarser, or is it a bug?
2. Is the lag bounded? Our worst observation is 1.9 d over one epoch and 32 GNSS
   objects. Is it worse for LEO, for recent epochs, or for larger catalogues?
3. Does it depend on how far in the past the requested epoch is? (Related: the
   ingest ramp already documented in `docs/tles.md`, where a catalogue keeps
   filling for weeks after the epoch.)
4. Is there a request parameter that makes `tles-at-epoch` resolve by nearest
   epoch?
5. Should this be reported upstream? If yes, §1 and §3 are the report.

## 7. What changes in tabascal, depending on the answer

Nothing has been changed on the strength of this yet. The source precedence lives
in `tabascal/tle.py:resolve_tles`; the relevant tests are
`tests/test_tle_policy.py`.

| Finding | Action |
|---|---|
| Upstream bug, gets fixed | Nothing. Re-measure after the fix and delete this brief. |
| Intentional and bounded | Document the bound. Possibly lower `remote_tle_max_age_days` so more satellites are routed to the per-ID upgrade. |
| Intentional and unbounded | Reconsider precedence: bulk for coverage, then per-ID *upgrade* for any record older than a threshold well below the ceiling. Cost is one request per upgraded satellite at a 1 s throttle, so it needs a cap for large lists. |
| Worse for LEO / recent epochs | Blocks the issue #101 calibration: "TLE age" would depend on which endpoint answered, so the suitability model must carry provenance, not just age. |

Whatever the outcome, note the constraint that shaped the current design: the
bulk endpoint is **one** download for the whole catalogue, whereas per-ID is one
request per satellite. For a 32-satellite benchmark that is ~32 s; for a
Starlink-scale list it is not viable. Any change must keep that in view.
