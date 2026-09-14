# Antenna reuse with four baseline workers

This construction minimizes the largest worker antenna set for the complete
512-antenna cross-correlation set, when each baseline belongs to one of four
workers and that worker materializes both antennas. A deterministic host-side planner is implemented in
`tabascal/baseline_partition.py`; it is not wired into the runner and is not a
claim of optimal total runtime. The current fixes retain
the data's baseline order.

Split the antennas into disjoint sets A, B, C of 102 antennas each, and D of
206. There are 130816 baselines, so each worker must own 32704. Here AA means
cross-correlations within A, excluding autocorrelations.

| Worker | Baselines | Count | Real antennas |
|---|---|---:|---:|
| 0 | AB, AC, BC; 498 AA, 497 BB, 497 CC | 31212 + 1492 = 32704 | 306 |
| 1 | AD; remaining AA; 7039 DD | 21012 + 4653 + 7039 = 32704 | 308 |
| 2 | BD; remaining BB; 7038 DD | 21012 + 4654 + 7038 = 32704 | 308 |
| 3 | CD; remaining CC; remaining 7038 DD | 21012 + 4654 + 7038 = 32704 | 308 |

The three DD subsets are disjoint and exhaust its 21115 baselines. Every
baseline occurs exactly once. Each worker has exactly the same number of
baseline evaluations, with no ghost baselines. A common local antenna width
of 308 requires only two dark, unused antenna slots on worker 0. Compared with
materializing all 512 antennas on each worker, this reduces the padded
per-antenna sample count by 39.84375%. It does not imply that total runtime
falls by that percentage: baseline products, GP evaluation, communication and
other stages remain.

Implementation would require both a baseline permutation and compact local
antenna tables. The permutation must apply consistently to observations,
flags, resolved noise, baseline coordinates and sky parameters, with an
inverse at export. Compact antenna gathers must remain inside the sharded
calculation so the transpose scatters gradients to the correct global antennas.
The current flat FFI argument ordering does not itself perform this remapping.

A useful next experiment would compare this construction against contiguous
ownership with the same kernel, scratch budget and 512A/8ch dataset, checking
values, gradients, exported ordering, peak memory and full optimization time.
The construction applies to one all-analytic group; mixed sampling groups and
incomplete baseline sets require separate load balancing.

A direct enumeration on mini verified all 130816 pairs occur exactly once,
with worker counts `[32704, 32704, 32704, 32704]` and antenna counts
`[306, 308, 308, 308]`. This checks the construction, not its runtime.

## Why 308 is the minimum common antenna width

For each antenna, write down the subset of workers that use it. Any two of
these subsets intersect: the worker owning their baseline uses both antennas.
Let M be the largest worker's antenna count, and n the total antenna count.

- If an antenna belongs to only one worker, every other antenna must occur on
  that worker too, giving M = n.
- If every antenna belongs to at least three workers, 4M >= 3n.
- Otherwise consider the worker pairs belonging to antennas that occur on
  exactly two workers. Pairwise-intersecting pairs form either a star (all
  contain one common worker) or a triangle. To see this, take pairs ab and ac:
  a pair not containing a must be bc; any further pair must then belong to
  that triangle.
- In the triangle case, every antenna must occur on at least two of its three
  workers, so 3M >= 2n.
- In the star case, let Y be the number of antennas absent from the center
  worker. They must occur on all three other workers; all two-worker antenna
  subsets contain the center. The center has n-Y antennas, so Y >= n-M.
  Across the other three workers there are at least n+2Y antenna occurrences.
  Hence 3M >= n+2Y >= 3n-2M, giving M >= 3n/5.

All cases give M >= ceil(3n/5). For n=512 that is 308. The balanced construction
above achieves it, so it attains the lower bound as well as equal baseline
counts. The claim concerns the common antenna width of this computation model;
it does not cover exchanging precomputed antenna samples between workers,
missing baselines, or mixed analytic/quadrature groups.

## Deterministic planning API

```python
from tabascal.baseline_partition import optimal_four_worker_partition

plan = optimal_four_worker_partition(a1, a2, n_devices=4)
packed = vis[plan.baseline_indices]  # example with baseline as first axis
restored = packed.reshape((-1,) + vis.shape[1:])[plan.inverse_indices]
```

Pass the configured baseline mesh width after mesh setup. The function checks
for four workers and a complete cross-correlation graph. It rejects unsupported
sizes rather than claiming a general optimal partition. Supported sizes must
both attain `ceil(3*n_ant/5)` and permit equal baseline counts without ghosts.

Sort antenna IDs ascending; assign the first `s = floor(n_ant/5)` to A, the next
s to B, the next s to C, and the remainder to D. Enumerate each within-group
and cross-group pair list lexicographically. Worker 0 takes AB, AC, BC, then
`extra = n_baselines/4 - 3*s*s` within-group pairs. Divide extra by three;
assign remainder pairs to A first, then B. Workers 1–3 take their respective
AD/BD/CD lists, the remaining AA/BB/CC pairs, then consecutive DD slices to
fill their equal capacities. No random seed or input-row ordering is involved.

The plan exposes packed-to-input row indices, an inverse permutation, sorted
per-worker antenna IDs, a padding mask, and local endpoints preserving the
original orientation. Padded antenna IDs are -1: gather only valid IDs and
zero the unused slots; never index a signal array directly with the sentinel.
The planner does not reorder any model state itself.

Correctness tests on mini cover complete coverage, equal loads, endpoint and
data-order round trips, arbitrary antenna IDs, shuffled rows, reversed pairs,
and rejection of unsupported device counts and invalid graphs. Production
activation still requires the consistent state permutation and differentiable
local gathers described above, followed by GPU value/gradient verification.
