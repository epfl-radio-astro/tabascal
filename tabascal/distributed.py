import os
import sys
from contextlib import contextmanager
from functools import lru_cache
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:  # public since jax 0.8; the experimental path covers the 0.6/0.7 range of our pin
    from jax import shard_map
except ImportError:  # pragma: no cover - exercised only on older jax
    from jax.experimental.shard_map import shard_map


# ---------------------------------------------------------------------------
# Initialization / capability checks
# ---------------------------------------------------------------------------

def _world_size() -> int:
    """Number of processes in this launch, across known launchers; 1 if single-process.

    Reads the world-size variable exported by whichever launcher started us -- SLURM,
    OpenMPI, MPICH/Intel MPI (PMI) or ``torchrun``/torch-elastic. We cannot infer
    multi-process from ``CUDA_VISIBLE_DEVICES`` alone: it tells us a GPU was pinned, not
    that peers exist or how to reach the coordinator. Returns 1 when no such variable is
    set, i.e. a plain ``python`` invocation.
    """
    for var in ("SLURM_NTASKS", "SLURM_STEP_NUM_TASKS", "SLURM_NPROCS",
                "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "WORLD_SIZE"):
        val = os.environ.get(var)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return 1


def _process_rank() -> int:
    """This process's rank, across known launchers; 0 if unset."""
    for var in ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK"):
        val = os.environ.get(var)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


def init_distributed() -> None:
    """Bring up the JAX distributed runtime when launched multi-process.

    Calls :func:`jax.distributed.initialize` **only** when the launcher reports more
    than one process (:func:`_world_size`), or when ``TABASCAL_FORCE_DISTRIBUTED`` is
    set. Outside a multi-process launch -- a plain ``python`` invocation or a
    single-process multi-device test -- this is a no-op, so we never block waiting for a
    coordinator that will not appear.

    Under SLURM (or when no MPI-style coordinator address is exported) we let JAX
    auto-detect the coordinator, process count and id. For other launchers that export
    ``MASTER_ADDR``/``MASTER_PORT`` (``torchrun`` and friends) we pass explicit
    coordinates. After this call ``jax.device_count()`` reports the *global* device
    count, so :func:`sharding_enabled` and everything downstream turn on automatically.

    Must be called before any JAX array is created (it initializes the device backend);
    the CLI calls it in ``run_tabascal._run_cmd`` before even importing the run
    implementation module.
    """
    if not (os.environ.get("TABASCAL_FORCE_DISTRIBUTED") or _world_size() > 1):
        return

    # SLURM auto-detects; so does JAX when no MPI-style coordinator is exported.
    if os.environ.get("SLURM_NTASKS") or "MASTER_ADDR" not in os.environ:
        jax.distributed.initialize()
    else:
        jax.distributed.initialize(
            coordinator_address=f"{os.environ['MASTER_ADDR']}:{os.environ.get('MASTER_PORT', '1234')}",
            num_processes=_world_size(),
            process_id=_process_rank(),
        )


def sharding_enabled() -> bool:
    """True when more than one global device is visible, so we should shard.

    ``jax.device_count()`` is the *global* device count once :func:`init_distributed`
    has run (it is the *local* count before that). So in the one-GPU-per-process layout
    this returns ``True`` on every process after init, even though each process owns a
    single device.
    """
    return jax.device_count() > 1


def is_process_0() -> bool:
    """True on the single process responsible for logging and writing results."""
    return jax.process_index() == 0


def process_count() -> int:
    """Number of processes in this run; 1 outside a multi-process launch."""
    return jax.process_count()

# ---------------------------------------------------------------------------
# RFI-axis sharding
# ---------------------------------------------------------------------------

# Array names whose *leading* axis is the RFI-source axis and which should therefore be
# split across devices. Everything else (per-baseline data, per-antenna gains, ast
# params, GP operators) is replicated. Constants are keyed "<prefix>/<name>" in the
# model, so matching happens on the part after the last "/". Membership here is not
# sufficient on its own: shard_pytree additionally requires leaf.shape[0] == n_rfi, so
# e.g. L_rfi_A -- deliberately absent, its leading dim is n_rfi_times -- could not be
# sharded even if a future rename made it match.
RFI_AXIS_NAMES = frozenset({
    # latent params (optimized)
    "rfi_k_r_base", "rfi_k_i_base",
    "rfi_r_induce_base", "rfi_i_induce_base",
    "rfi_orbit_base",
    # state buffers
    "rfi_A", "rfi_phase", "rfi_xyz", "elements",
    # constants
    "mu_rfi_k", "mu_rfi_A", "mu_rfi_orbit", "L_rfi_orbit",
})


@lru_cache(maxsize=None)
def rfi_mesh() -> Mesh:
    """The single 1-D device mesh used for RFI-axis sharding.

    Spans *all* global devices with one axis named ``"rfi"``, which covers both the
    single-process multi-GPU layout and the multi-process one-GPU-per-process layout
    uniformly. Cached: every caller must use the same mesh object so shardings compare
    equal and jit avoids spurious recompiles.
    """
    return Mesh(np.array(jax.devices()), ("rfi",))


def rfi_sharding() -> NamedSharding:
    """Sharding that splits an array's leading (RFI-source) axis across the mesh."""
    return NamedSharding(rfi_mesh(), P("rfi"))


def replicated_sharding() -> NamedSharding:
    """Sharding that replicates an array on every device of the mesh."""
    return NamedSharding(rfi_mesh(), P())


def padded_rfi_count(n_rfi: int) -> int:
    """Smallest multiple of the device count >= ``n_rfi``; ``n_rfi`` when not sharding.

    The RFI axis must divide evenly across the mesh, so TabConfig pads the satellite
    list up to this count with dark dummy sources (zero prior mean and zero init on
    their signal latents, hence exactly zero signal and zero gradient forever).
    """
    if not sharding_enabled():
        return n_rfi
    n_dev = jax.device_count()
    return n_rfi + (-n_rfi) % n_dev


def make_global(x, sharding: NamedSharding) -> jax.Array:
    """Build a (possibly multi-process) global array from a full per-process host copy.

    Every process holds the complete array (they all read the same MS / run the same
    setup), so the callback just slices the requested shard out of the local copy. This
    is the supported construction path in both single-process multi-device and
    multi-process layouts, unlike ``jax.device_put`` which historically rejects
    cross-process shardings.
    """
    x_np = np.asarray(x)
    return jax.make_array_from_callback(x_np.shape, sharding, lambda idx: x_np[idx])


def _wants_rfi_axis(key: str, leaf, n_rfi: int) -> bool:
    name = key.rsplit("/", 1)[-1]
    return name in RFI_AXIS_NAMES and np.ndim(leaf) >= 1 and np.shape(leaf)[0] == n_rfi


def shard_pytree(tree: dict, n_rfi: int) -> dict:
    """Device-put a flat dict of arrays: RFI-axis entries sharded, the rest replicated.

    Identity when sharding is off. Leaves that already carry the requested sharding
    (e.g. placeholders created via :func:`sharded_rfi_zeros`) are passed through
    untouched -- rebuilding them from a host copy would defeat their purpose of never
    existing as a full single-device array.
    """
    if not sharding_enabled():
        return tree

    out = {}
    for key, leaf in tree.items():
        want = rfi_sharding() if _wants_rfi_axis(key, leaf, n_rfi) else replicated_sharding()
        if isinstance(leaf, jax.Array) and leaf.sharding == want:
            out[key] = leaf
        else:
            out[key] = make_global(leaf, want)
    return out


def sharded_rfi_zeros(shape: tuple, dtype) -> jax.Array:
    """Zeros with the leading (RFI) axis sharded, without a full single-device copy.

    Used for the big ``rfi_A``/``rfi_phase`` state placeholders: each device only ever
    allocates its own shard, which is what lets a run hold more RFI sources than one
    GPU fits. Plain ``jnp.zeros`` when sharding is off.
    """
    if not sharding_enabled():
        return jnp.zeros(shape, dtype=dtype)
    # Canonicalize like jnp.zeros would (None -> default float; complex -> complex64
    # under f32): numpy would otherwise hand the callback f64/c128 buffers that
    # disagree with the array dtype jax expects when x64 is off.
    dtype = jnp.zeros((), dtype=dtype).dtype
    sharding = rfi_sharding()
    return jax.make_array_from_callback(
        tuple(shape), sharding, lambda idx: np.zeros(_index_shape(shape, idx), dtype=dtype)
    )


def _index_shape(shape, idx) -> tuple:
    """Shape of ``zeros(shape)[idx]`` without materializing the full array."""
    return tuple(
        len(range(*s.indices(dim))) for s, dim in zip(idx, shape)
    )


def constrain_rfi_state(state: dict, n_rfi: int) -> dict:
    """Pin per-RFI state entries to the RFI sharding inside traced code.

    Called between component forwards so XLA keeps ``rfi_A``/``rfi_phase`` (the
    fine-grid memory hogs) split across devices instead of ever materializing a
    replicated copy. No-op when sharding is off.
    """
    if not sharding_enabled():
        return state
    sharding = rfi_sharding()
    out = dict(state)
    for key, leaf in state.items():
        if _wants_rfi_axis(key, leaf, n_rfi):
            out[key] = lax.with_sharding_constraint(leaf, sharding)
    return out


def psum_over_rfi(local_fn: Callable) -> Callable:
    """Map a per-RFI-shard visibility function over the mesh and sum across shards.

    ``local_fn(rfi_A, rfi_phase) -> vis`` must accept any leading RFI count and return
    an array with **no** RFI axis (its local sources already summed). Under sharding it
    runs per device on the local shard via ``shard_map`` -- which is also what lets the
    FFI custom op participate, since GSPMD cannot partition a custom call -- and the
    small coarse-grid results are ``psum``-ed into a replicated total. Unsharded it is
    ``local_fn`` itself, keeping the single-device path bitwise identical.
    """
    if not sharding_enabled():
        return local_fn

    def summed(rfi_A, rfi_phase):
        return lax.psum(local_fn(rfi_A, rfi_phase), "rfi")

    # Varying-axis type checking must be off: the FFI kernel's custom JVP/transpose
    # rules produce cotangents without the {V:rfi} annotation, which trips the check
    # under value_and_grad. The out_specs still hold -- the psum makes the result
    # replicated -- the checker just cannot prove it for custom primitives.
    kwargs = dict(mesh=rfi_mesh(), in_specs=(P("rfi"), P("rfi")), out_specs=P())
    try:
        return shard_map(summed, check_vma=False, **kwargs)
    except TypeError:  # pragma: no cover - jax < 0.7 spells it check_rep
        return shard_map(summed, check_rep=False, **kwargs)


# ---------------------------------------------------------------------------
# Multi-process coordination
# ---------------------------------------------------------------------------

def barrier(name: str) -> None:
    """Block until every process reaches this point; no-op single-process."""
    if jax.process_count() == 1:
        return
    from jax.experimental import multihost_utils
    multihost_utils.sync_global_devices(name)


def broadcast_bytes_from_rank0(payload, name: str) -> bytes:
    """Broadcast process 0's *payload* bytes to every process, verbatim.

    Used to share a result that only process 0 may compute -- the resolved TLE set
    -- so workers never repeat the work. Sharing the *result* rather than merely
    ordering the processes is what makes the outcome coherent when the shared
    filesystem cache cannot be written: there is then nothing on disk for a worker
    to read, and re-deriving it would mean one provider request per process.

    ``broadcast_one_to_all`` requires identical shapes on every process, so the
    length is broadcast first and the payload second. Non-zero processes pass
    ``None``. Single-process: the payload straight back.
    """
    if jax.process_count() == 1:
        return bytes(payload or b"")

    from jax.experimental import multihost_utils

    rank0 = is_process_0()
    size = np.asarray(
        multihost_utils.broadcast_one_to_all(
            np.array([len(payload) if rank0 else 0], dtype=np.int64)
        )
    )
    buffer = np.zeros(int(size[0]), dtype=np.uint8)
    if rank0 and buffer.size:
        buffer[:] = np.frombuffer(payload, dtype=np.uint8)
    return np.asarray(multihost_utils.broadcast_one_to_all(buffer)).tobytes()


@contextmanager
def rank0_first(name: str):
    """Run the block on process 0 first, then on all other processes.

    Serializes shared-resource setup so workers find the resource already in place.
    Single-process: plain yield.

    Not used for the TLE fetch: sharing only the *ordering* leaves workers to
    re-derive the result from the cache, which is exactly what fails when the cache
    is unwritable. :func:`broadcast_bytes_from_rank0` shares the result itself.

    Process 0 releases the barrier from a ``finally``, so if its block raises (e.g. a
    network failure) the workers are woken instead of blocking until the coordinator
    timeout; they then fail independently on the missing resource. Turns a
    multi-process hang into a fast, symmetric error.
    """
    if jax.process_count() == 1:
        yield
        return
    if is_process_0():
        try:
            yield
        finally:
            barrier(f"rank0_first:{name}")
    else:
        barrier(f"rank0_first:{name}")
        yield


# ---------------------------------------------------------------------------
# Host materialization
# ---------------------------------------------------------------------------

def to_host(arr) -> np.ndarray:
    """Materialize a (possibly sharded) device array as a full host numpy array.

    With RFI-axis sharding the written results (per-baseline ``vis_*``, per-antenna
    ``gains``) are replicated and fully addressable on every process, so this is just
    ``np.asarray``; per-RFI arrays are not written to disk.
    """
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Process-0 IO guards
# ---------------------------------------------------------------------------

def print0(*args, **kwargs) -> None:
    """``print`` only on process 0; a no-op elsewhere."""
    if is_process_0():
        print(*args, **kwargs)


@contextmanager
def suppress_worker_stdout():
    """Silence stdout on non-zero processes for the duration of the block.

    Component ``setup`` and the model summary print a lot; without this every worker
    would duplicate it. Process 0 is untouched.
    """
    if is_process_0():
        yield
        return
    saved = sys.stdout
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = saved
