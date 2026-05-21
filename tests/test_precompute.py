"""Unit tests for the static/live component classification and the precompute
equivalence (``tabascal.config.classify_live_components``).

Uses lightweight stub components with plain-Python forwards, so no JAX arrays,
MeasurementSet, or component setup is needed. The end-to-end pipeline tests are
the exactness guard for the real (jitted) Model.
"""

from tabascal.config import classify_live_components


def stub(name, forward, reads=None, writes=None, accumulates=None, params=False):
    """A minimal stub component: I/O declarations + a plain-Python forward."""
    return type(
        name,
        (),
        {
            "reads": reads or {},
            "writes": writes or {},
            "accumulates": accumulates or {},
            "init_params_base": {"p": 0} if params else {},
            "build_forward": lambda self, _f=forward: _f,
        },
    )()


def run_chain(comps, params, state):
    state = dict(state)
    for comp in comps:
        state = comp.build_forward()(params, state, {})
    return state


def example_stack():
    """A fixed-orbit + learnable-RFI + fixed-sky + gains stack of stubs.

    Mirrors the FixedPointSky e2e case: the orbit and sky chains are static
    (no params), the RFI chain is live (RFISignal has params), and gains is
    live because it reads the live vis_rfi.
    """
    orbit = stub("Orbit", lambda p, s, c: {**s, "rfi_xyz": 1.0},
                 writes={"rfi_xyz": ("n",)})
    rfi_sig = stub("RFISignal", lambda p, s, c: {**s, "rfi_A": p["rfi_A_base"]},
                   writes={"rfi_A": ("n",)}, params=True)
    rfi_vis = stub("RFIVis", lambda p, s, c: {**s, "vis_rfi": s["vis_rfi"] + s["rfi_A"]},
                   reads={"rfi_A": ("n",)}, accumulates={"vis_rfi": ("n",)})
    sky = stub("Sky", lambda p, s, c: {**s, "ast_I": 2.0},
               writes={"ast_I": ("n",)})
    ast_vis = stub("AstVis", lambda p, s, c: {**s, "vis_ast": s["vis_ast"] + s["ast_I"]},
                   reads={"ast_I": ("n",)}, accumulates={"vis_ast": ("n",)})
    gains = stub("Gains", lambda p, s, c: {**s, "vis_obs": s["vis_rfi"] + s["vis_ast"]},
                 reads={"vis_rfi": ("n",), "vis_ast": ("n",)},
                 writes={"vis_obs": ("n",)})
    return [orbit, rfi_sig, rfi_vis, sky, ast_vis, gains]


def test_classify_taints_through_state():
    comps = example_stack()
    live = classify_live_components(comps)
    # Orbit, Sky, AstVis static; RFISignal, RFIVis, Gains live.
    assert live == [False, True, True, False, False, True]


def test_classify_fully_static_when_no_params():
    """With no learnable params anywhere, everything is static."""
    comps = [c for c in example_stack() if type(c).__name__ != "RFISignal"]
    # Replace the param-bearing signal with a static writer of rfi_A.
    rfi_sig = stub("FixedRFI", lambda p, s, c: {**s, "rfi_A": 3.0},
                   writes={"rfi_A": ("n",)})
    comps = [comps[0], rfi_sig, *comps[1:]]
    assert classify_live_components(comps) == [False] * len(comps)


def test_static_then_live_matches_running_all():
    """Precompute equivalence: static-first + live == running every component."""
    comps = example_stack()
    seed = {"vis_ast": 0.0, "vis_rfi": 0.0}
    params = {"rfi_A_base": 5.0}

    full = run_chain(comps, params, seed)

    live = classify_live_components(comps)
    static_comps = [c for c, lv in zip(comps, live) if not lv]
    live_comps = [c for c, lv in zip(comps, live) if lv]
    baseline = run_chain(static_comps, {}, seed)   # params unused by static comps
    split = run_chain(live_comps, params, baseline)

    assert split == full
    assert full["vis_obs"] == 7.0  # 5 (rfi) + 2 (sky)
