"""Regression tests for AnalyticVisCalculation (analytic RFI-visibility path).

The analytic component replaces the oversample-and-average with a sub-windowed
closed-form Fresnel fringe factor. These tests exercise the full production stack on a
self-contained synthetic near-field geometry (a fast LEO over a few-km array): the
host-side near-field fringe-parameter fit, the K-sizing, the interleaved edge/centre
grid, and the component assembly. They check that the component reproduces a high-N
oversample oracle of the same envelope x geometric phase to the envelope-linearisation
floor, that gradients flow to the GP envelope and match finite differences, and that the
oversample default path is untouched.

cerf is exercised at large |z|, so the module requires double precision.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from types import SimpleNamespace

from tabascal.interferometry import (
    C,
    fit_nearfield_fringe_freq_poly_numpy,
    fringe_params_at_offsets,
    size_subwindows,
)
from tabascal.components.rfi_vis import AnalyticVisCalculation, RiemannVisCalculation

pytestmark = pytest.mark.requires_double

FREQ = 1.5e9
DT = 2.0
N_TIME = 6
N_ANT = 5
ELL = 24.0           # GP correlation time (s) -> envelope floor (DT/ELL)^2
N_FIT = 16

# --- analytic near-field geometry (closed-form in t, so oracle == component phase) ---
_BASE = np.array([5109360.0, 2006852.0, -3238948.0])
_OMEGA = 7.292e-5
_rng = np.random.default_rng(7)
_ANT_OFF = np.concatenate([np.zeros((1, 3)), _rng.uniform(-4000, 4000, (N_ANT - 1, 3))])
_ANTS0 = _BASE + _ANT_OFF
_R0 = _BASE / np.linalg.norm(_BASE) * (np.linalg.norm(_BASE) + 550e3)
_V = np.array([7500.0, 0.0, 0.0])


def sat_xyz(t):
    t = np.atleast_1d(t)
    return _R0[None, :] + _V[None, :] * t[:, None]  # (T, 3)


def ants_xyz(t):
    t = np.atleast_1d(t)
    th = _OMEGA * t
    cz, sz = np.cos(th), np.sin(th)
    ax = _ANTS0[:, 0][:, None] * cz[None, :] - _ANTS0[:, 1][:, None] * sz[None, :]
    ay = _ANTS0[:, 0][:, None] * sz[None, :] + _ANTS0[:, 1][:, None] * cz[None, :]
    az = np.repeat(_ANTS0[:, 2][:, None], len(t), axis=1)
    return np.stack([ax, ay, az], axis=-1)  # (N_ANT, T, 3)


def ants_w(t):
    t = np.atleast_1d(t)
    return 1e-3 * _ANT_OFF[:, 0][:, None] * np.cos(0.01 * t)[None, :]  # (N_ANT, T)


def phase_ant(t):
    """Near-field geometric phase phi_p(t) = -2pi(|ant-sat|+w)/lambda (N_ANT, T)."""
    lam = C / FREQ
    dist = np.linalg.norm(ants_xyz(t) - sat_xyz(t)[None], axis=-1)
    return -2 * np.pi * (dist + ants_w(t)) / lam


# --- SE-GP envelope (the differentiable parameter) ---
_T_IND = np.arange(-DT, N_TIME * DT + ELL, ELL)
_N_IND = len(_T_IND)


def _se(x, y):
    return np.exp(-0.5 * ((np.atleast_1d(x)[:, None] - y[None, :]) / ELL) ** 2)


_KII = np.linalg.inv(_se(_T_IND, _T_IND) + 1e-8 * np.eye(_N_IND))


def resample(t):
    return _se(t, _T_IND) @ _KII  # (T, N_IND)


_THETA = _rng.standard_normal((N_ANT, _N_IND)) + 1j * _rng.standard_normal((N_ANT, _N_IND))


def env_ant(t, theta=_THETA):
    return (resample(np.atleast_1d(t)) @ theta.T).T  # (N_ANT, T)


@pytest.fixture(scope="module")
def setup():
    a1, a2 = np.triu_indices(N_ANT, 1)
    n_bl = len(a1)
    win_c = np.arange(N_TIME) * DT

    # fringe params (mirror config.setup_analytic_sampling)
    dt_fit = DT / N_FIT
    t_fit = win_c[0] - DT / 2 + (np.arange(N_TIME * N_FIT) + 0.5) * dt_fit
    coeffs = fit_nearfield_fringe_freq_poly_numpy(
        t_fit, FREQ, sat_xyz(t_fit)[None], ants_xyz(t_fit), ants_w(t_fit), a1, a2, N_TIME
    )
    K, fddot_max = size_subwindows(coeffs, DT, resid_tol=3e-4)
    dt_sub = DT / K
    centre_off = -DT / 2 + (np.arange(K) + 0.5) * dt_sub
    f_ref, fdot_ref = fringe_params_at_offsets(coeffs, centre_off)

    # interleaved edge/centre grid (mirror _set_freqs_times)
    t0 = win_c[0] - DT / 2
    n_e, n_c = N_TIME * K + 1, N_TIME * K
    grid = np.empty(n_e + n_c)
    grid[0::2] = t0 + np.arange(n_e) * dt_sub
    grid[1::2] = t0 + (np.arange(n_c) + 0.5) * dt_sub

    edge_gather = (np.arange(N_TIME)[:, None] * K + np.arange(K + 1)[None, :]).astype(np.int32)
    cfg = SimpleNamespace(
        a1=jnp.asarray(a1), a2=jnp.asarray(a2), n_time=N_TIME, n_bl=n_bl, n_freq=1,
        n_rfi=1, vis_method="analytic", analytic_K=int(K), analytic_dt_sub=float(dt_sub),
        analytic_f=f_ref, analytic_fdot=fdot_ref,
        analytic_freq_scale=np.array([1.0]), analytic_edge_gather=edge_gather,
    )
    fdt = np.abs(f_ref).max() * DT
    return SimpleNamespace(
        cfg=cfg, a1=a1, a2=a2, n_bl=n_bl, K=int(K), grid=grid, win_c=win_c, fdt=fdt
    )


def _state(grid, theta=_THETA):
    A = env_ant(grid, theta)        # (N_ANT, n_grid)
    P = phase_ant(grid)             # (N_ANT, n_grid)
    return {
        "rfi_A": jnp.asarray(A[None, :, None, :]),
        "rfi_phase": jnp.asarray(P[None, :, None, :]),
        "vis_rfi": jnp.zeros((len(np.triu_indices(N_ANT, 1)[0]), 1, N_TIME), complex),
    }


def _run(setup, state=None):
    comp = AnalyticVisCalculation()
    comp.setup(setup.cfg)
    fwd = comp.build_forward()
    consts = {f"{comp.prefix}/{k}": v for k, v in comp.build_constants().items()}
    if state is None:
        state = _state(setup.grid)
    return np.asarray(fwd({}, state, consts)["vis_rfi"]), comp, fwd, consts


def _oracle(setup):
    a1, a2, win_c = setup.a1, setup.a2, setup.win_c
    n_bl = setup.n_bl
    V = np.zeros((n_bl, N_TIME), complex)
    N = 8193
    wts = np.ones(N); wts[1:-1:2] = 4; wts[2:-1:2] = 2
    for win in range(N_TIME):
        s = np.linspace(-DT / 2, DT / 2, N); t = win_c[win] + s
        A = env_ant(t); P = phase_ant(t)
        for b in range(n_bl):
            w = A[a1[b]] * np.conj(A[a2[b]]) * np.exp(1j * (P[a1[b]] - P[a2[b]]))
            V[b, win] = (DT / (N - 1) / 3) * np.sum(wts * w) / DT
    return V


def test_exercises_fresnel_regime(setup):
    """The synthetic geometry must reach K>1 / f*dt>>1 so the Fresnel path is tested."""
    assert setup.K > 1
    assert setup.fdt > 5.0


def test_output_shape(setup):
    V, *_ = _run(setup)
    assert V.shape == (setup.n_bl, 1, N_TIME)


def test_forward_parity_to_envelope_floor(setup):
    """Reproduces the oversample oracle to the envelope-linearisation floor (DT/ELL)^2."""
    V_an = _run(setup)[0][:, 0, :]
    V_or = _oracle(setup)
    wscale = np.mean(np.abs(env_ant(setup.grid)) ** 2)  # ~ |w| envelope scale
    abserr = np.abs(V_an - V_or) / wscale
    floor = (DT / ELL) ** 2
    assert abserr.max() < 3 * floor, f"max abs err {abserr.max():.2e} vs floor {floor:.2e}"


def test_accumulates_into_vis_rfi(setup):
    """vis_rfi is added to, not overwritten."""
    base = _state(setup.grid)
    seed = jnp.ones_like(base["vis_rfi"])
    out0, comp, fwd, consts = _run(setup, state={**base, "vis_rfi": jnp.zeros_like(seed)})
    out1 = np.asarray(fwd({}, {**base, "vis_rfi": seed}, consts)["vis_rfi"])
    assert np.allclose(out1 - 1.0, out0, atol=1e-10)


def test_grad_flows_and_matches_fd(setup):
    """d sum|V|^2 / d(theta_real) is finite and matches a finite difference."""
    _, comp, fwd, consts = _run(setup)
    grid = setup.grid
    R = jnp.asarray(resample(grid))
    theta_i = jnp.asarray(_THETA.imag)
    base_vis = jnp.zeros((setup.n_bl, 1, N_TIME), complex)
    P = jnp.asarray(phase_ant(grid)[None, :, None, :])

    def loss(theta_r):
        A = (R @ (theta_r + 1j * theta_i).T).T
        st = {"rfi_A": A[None, :, None, :], "rfi_phase": P, "vis_rfi": base_vis}
        return jnp.sum(jnp.abs(fwd({}, st, consts)["vis_rfi"]) ** 2)

    theta_r = jnp.asarray(_THETA.real)
    g = jax.grad(loss)(theta_r)
    assert bool(jnp.all(jnp.isfinite(g)))
    eps = 1e-6
    j = (0, _N_IND // 2)
    fd = (loss(theta_r.at[j].add(eps)) - loss(theta_r)) / eps
    rel = abs(float(g[j]) - float(fd)) / (abs(float(fd)) + 1e-30)
    assert rel < 1e-4, f"grad {float(g[j]):.6e} vs FD {float(fd):.6e} (rel {rel:.2e})"


def test_jit_compiles(setup):
    _, comp, fwd, consts = _run(setup)
    state = _state(setup.grid)
    out = jax.jit(lambda s: fwd({}, s, consts))(state)
    assert np.all(np.isfinite(np.asarray(out["vis_rfi"])))


def test_requires_analytic_method(setup):
    """Component refuses to set up against a non-analytic config."""
    bad = SimpleNamespace(**{**vars(setup.cfg), "vis_method": "oversample"})
    with pytest.raises(RuntimeError):
        AnalyticVisCalculation().setup(bad)
