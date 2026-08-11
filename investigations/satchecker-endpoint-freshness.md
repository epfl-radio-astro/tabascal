# Investigation brief: SatChecker returns different TLEs per endpoint

**Status:** resolved — cause identified in upstream source, read at
`iausathub/satchecker@3cec98c`. Not a bug: the two endpoints answer two
different questions. Behaviour is bounded and undocumented.
**Raised by:** PR #92 (SatChecker TLE retrieval), 2026-08-12
**Relevant issue:** [#101](https://github.com/epfl-radio-astro/tabascal/issues/101) (calibrated TLE suitability)
**Related upstream issue:** [iausathub/satchecker#246](https://github.com/iausathub/satchecker/issues/246) (partial-catalogue ingest lag — separate defect, still open)
**Filed upstream from this investigation:** [iausathub/satchecker#247](https://github.com/iausathub/satchecker/issues/247) — documentation request for the selection rule, 2026-08-12
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

## 2. The cause

**`tles-at-epoch` selects the newest record at or before the requested epoch.
`get-nearest-tle` selects the record with the smallest `|epoch − target|`, in
either direction.** Every discrepancy in §1 is a satellite whose nearest record
lies *after* the requested epoch, and which the bulk endpoint therefore cannot
see.

Both routes are thin; the services are pass-throughs; the divergence is entirely
in the repository layer.

| Layer | Bulk | Per-ID |
|---|---|---|
| Route | `entrypoints/v1/routes/tools_routes.py:526` | `tools_routes.py:691` |
| Service | `services/tools_service.py:523` | `services/tools_service.py:176` |
| Repository | `adapters/repositories/tle_repository.py:361` | `tle_repository.py:713` → `adapters/repositories/orbital_data_lookup_mixin.py:157` |

Bulk, `tle_repository.py:457` (with `end_date = epoch`, `start_date = epoch − 14 d`
from line 375):

```sql
SELECT DISTINCT ON (sat_id) ...
FROM tle
WHERE epoch BETWEEN :start_date AND :end_date
  AND sat_id = ANY(:satellite_ids)
ORDER BY sat_id, epoch DESC          -- newest record at or before the epoch
```

Per-ID, `orbital_data_lookup_mixin.py:182` — no window, no direction constraint:

```python
.order_by(func.abs(func.extract("epoch", model.epoch)
                 - func.extract("epoch", epoch_param)))
.first()                              # true nearest, may be after the epoch
```

### This is by design, not an accident of dedup

- The bulk endpoint was requested (`iausathub/satchecker#108`) as *"a snapshot of
  all the current TLEs at a given time"* — i.e. what a user would have held at
  that moment. Look-ahead would defeat that.
- The lookback window has been deliberately tuned: `06876cb` (Oct 2024) widened
  it from one week to two, *"to be the upper bound of what is useful orbit
  data"*.
- The unreleased rewrite at `tle_repository.py:541`
  (`_get_all_tles_at_epoch_experimental`) keeps the identical semantics —
  `ORDER BY t.epoch DESC LIMIT 1` over the same two-week window. Upstream is not
  changing this.

So hypothesis 2 from the original brief was right; hypotheses 3 and 4 were wrong
(one table, and the `DISTINCT ON` sort key *is* epoch); hypothesis 5 holds —
intentional, but undocumented.

### The observed numbers are exactly what this mechanism predicts

For a target epoch falling in a gap of length Δ between consecutive records,
"newest at or before" gives an age uniform on (0, Δ]; "nearest" gives
min(x, Δ−x). Predicted ratios: **2× in the median, 2× in the max**. Observed:
0.595 / 0.280 = 2.13, and 2.160 / 1.063 = 2.03. The 45× outlier on 40294 is not
a separate effect — it is one target that happened to land just *before* a new
record, so the ratio is large while the absolute bulk age (1.95 d) stays inside
the same envelope as everything else.

The practical reading: **the ratio is unbounded, the absolute penalty is not.**
Bulk age is bounded by one inter-record gap for that object, hard-capped at 14
days.

### Hypothesis 1 (precomputed snapshot) — false for historical epochs

There is a cache, but it applies only when the requested epoch is in the future
or within the last 3 hours (`tle_repository.py:388`). It is refreshed by
`services/cache_service.py:168` with a 3 h TTL, and it stores the result of the
*same* query at `now`. Historical epochs like the 2023 one in §1 always hit the
database live. Recent-epoch requests can lag by at most the 3 h TTL.

## 3. What has already been ruled out

Do not re-derive these; they cost live requests.

- **Not the canonical bucket.** Both figures in §1 come from the *same* epoch
  argument (09:00 UTC). Bucketing can contribute at most `interval/2` = 1 h; the
  observed gaps reach 1.9 days.
- **Not tabascal's record selection.** When a bulk response carries several rows
  for one NORAD ID, `_select_from_records` picks the row nearest the reference
  epoch. The bulk responses here carried a *single* row per object — necessarily,
  given `DISTINCT ON (sat_id)` — so there was nothing to select between.
- **Not tabascal's parsing.** Ages are computed from TLE line 1 columns 19–32 by
  `tabascal/satchecker/tle_parse.py:tle_epoch_jd`, the same function on both
  paths. The comparison is on raw `TLE_LINE1` strings, not derived elements.
- **Not a provider split.** Every record on both paths reported
  `data_source: spacetrack`. This is not CelesTrak-vs-Space-Track.
- **Not a stale local cache.** The measurement used a throwaway `TLE_CACHE_DIR`.

## 4. Reproduction

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

Expect roughly the table in §1. Note the sign test that confirms §2 without any
further requests: in every discrepant case the per-ID record's epoch is *later*
than the queried epoch.

To confirm the endpoint difference *without* tabascal in the path, hit the HTTP
API directly — this is the form the client uses
(`tabascal/satchecker/client.py`), note `epoch` is a bare Julian date:

```bash
curl -s 'https://satchecker.cps.iau.org/tools/get-nearest-tle/?id=40294&id_type=catalog&epoch=2459996.875'
curl -s 'https://satchecker.cps.iau.org/tools/tles-at-epoch/?epoch=2459996.875&format=json&per_page=1&page=1'
```

## 5. Answers to the questions in the original brief

1. **Is `tles-at-epoch` intentionally coarser, or is it a bug?** Intentional, and
   not really "coarser" — it answers a different question ("what was current at
   time T") from `get-nearest-tle` ("what best describes the orbit at time T").
   For our purpose, only the second question is the right one. The behaviour is
   **undocumented**: `docs/source/tools_tle.rst:73` says only "fetches all TLEs at
   a specific epoch date", mentioning neither the one-sidedness nor the two-week
   window.
2. **Is the lag bounded?** Yes, twice over. Bulk age ≤ one inter-record gap for
   that object, and ≤ 14 days absolutely — past 14 days the object is *dropped
   from the response entirely* rather than returned stale. Expected penalty
   relative to nearest is 2× in both median and worst case.
3. **Does it depend on how far in the past the epoch is?** Not through this
   mechanism — it depends only on the object's record spacing around the target.
   Note the *separate* ingest-lag defect (upstream #246): for epochs younger than
   ~30 days the catalogue is still filling, and between 12–16 days old it returns
   a near-empty catalogue that is indistinguishable from a complete one. That is
   a coverage problem, not a freshness one, and it is the more dangerous of the
   two.
   Also, for LEO the penalty should be *smaller*, not larger: update cadence is
   higher, so gaps are shorter. The 32 GNSS objects in §1 are closer to a worst
   case than a typical one among well-tracked satellites. Poorly-tracked debris
   would be worse.
4. **Is there a parameter that makes `tles-at-epoch` resolve by nearest epoch?**
   No. The route accepts only `epoch`, `page`, `per_page`, `format`
   (`tools_routes.py:622`); unlisted query parameters are silently dropped by
   `validate_parameters`. The repository's `constellation`,
   `data_source_limit` and `use_generated_tles` arguments are not wired to this
   route, and none of them would change the selection rule anyway. No bulk
   endpoint with nearest semantics exists.
5. **Should this be reported upstream?** Not as a bug. Filed as a documentation
   request, [#247](https://github.com/iausathub/satchecker/issues/247): state on
   `tles-at-epoch` that selection is "newest record with epoch ≤ requested epoch,
   within a 14-day lookback", point users needing nearest-by-absolute-time at
   `get-nearest-tle`, and document the three side effects (objects silently
   omitted, `generated` records excluded, 3 h cache). The `documentation` label
   could not be applied — filing under an account without triage permission on
   the repo drops labels silently. The genuine defect (#246) was already filed.

## 6. Endpoint options for nearest-by-epoch retrieval

**There is no bulk-by-ID endpoint.** All 23 routes were enumerated; every
endpoint with nearest semantics is single-object, and none accepts a list of IDs.
`extract_parameters` (`services/validation_service.py:22`) reads each parameter
with `request.args.get(param, None)`, so a repeated `?id=A&id=B` silently takes
only the first. Retrieving nearest records for a set of NORAD IDs therefore costs
one request per satellite, whichever endpoint is used.

| Endpoint | Returns | Fit for our use |
|---|---|---|
| `get-nearest-tle` | single nearest record | Exact semantics, smallest payload. Currently used. |
| `get-adjacent-tles` | records immediately before and after | Nearest is whichever is closer; also yields the bracketing pair |
| `get-tles-around-epoch` | `count_before` / `count_after` records | Same, sized by count |
| `get-tle-data` | all records in `start_date_jd`–`end_date_jd` | One request covers a whole time span |
| `tles-at-epoch` | one record per object, newest-before | Only single-request-for-everything option; wrong selection rule (§2) |

### The 1 s client throttle is ~33× more conservative than the published limit

All four per-ID endpoints carry `@limiter.limit("100 per second, 2000 per
minute")` (`tools_routes.py:257, 693, 795, 897`), matching the global default in
`entrypoints/extensions.py:68`. The limiter is keyed on forwarded address with a
moving window, backed by Redis. There is no additional `limit_req` or
`limit_conn` in `nginx/`.

Our client throttles at 1 s ≈ 60 requests/min against a published 2000/min. At
the published ceiling, 32 satellites take under a second rather than 32 s, and a
full ~17,800-object catalogue takes ~9 minutes — slow, but not the hard barrier
assumed when the current design was chosen.

Two cautions before exploiting that headroom. The limiter is configured with
`swallow_errors=True`, so if Redis is unavailable it **fails open** — the
published ceiling is not a guarantee that the service can absorb that rate. And
this is a shared public service run by IAU CPS; sustained near-limit traffic is
antisocial regardless of what the decorator permits. A throttle in the low
hundreds of ms is defensible; saturating 2000/min for a Starlink-scale list is
not, without asking first.

## 7. What this means for tabascal

Nothing has been changed on the strength of this yet. The source precedence lives
in `tabascal/tle.py:resolve_tles`; the relevant tests are
`tests/test_tle_policy.py`.

This lands on the **"intentional and bounded"** row of the original decision
table. Recommended:

1. **Document the bound** where `resolve_tles` chooses its source: bulk records
   carry an age of up to one update interval, capped at 14 days, and are
   systematically ~2× older than the best available record.
2. **Revisit the throttle before revisiting the architecture.** The per-ID upgrade
   path was scoped around a 1 s throttle that §6 shows to be ~33× more
   conservative than the service permits. Raising it is a one-line change that
   makes per-ID viable for far longer lists than assumed, and it should be tried
   before adding threshold logic to avoid requests. Bulk remains **one** download
   for the whole catalogue against one request per satellite, so the asymmetry
   still argues for bulk-as-coverage at Starlink scale — but the crossover sits
   much further out than the current design assumes.
3. **Consider `get-tle-data` over `get-nearest-tle` for the upgrade path.** At one
   request per satellite either way, a date-range fetch returns every record in
   the window, so one request serves *all* epochs in an observation rather than
   one, and it supplies the neighbouring records that issue #101 needs for a
   suitability judgement. Unmeasured — this follows from the API surface, not
   from a benchmark.
4. **Carry provenance, not just age**, into the issue #101 suitability model. Two
   records of equal nominal age mean different things depending on which endpoint
   answered, because the bulk record is always *behind* the epoch while the per-ID
   record may straddle it. SGP4 error grows with |Δt| regardless of sign, so the
   sign matters for interpretation, not for the error budget.

One cheap option worth evaluating before spending requests on upgrades: because
bulk selection is "newest ≤ epoch", querying the bulk endpoint at a canonical
epoch shifted *forward* of the observation lets it see post-observation records,
recovering much of the nearest-endpoint quality at no extra request cost. The
shift would need tuning against typical record spacing — too large and it
overshoots — and it interacts with both the canonical bucketing and the ingest
lag of #246. Measure before adopting.
