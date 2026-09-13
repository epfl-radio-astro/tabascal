"""Compare the JAX reference with a git revision in fresh Linux processes.

Run on the measurement host, from the worktree with the proposed change::

    python ci/measure_analytic_rfi_vis.py --baseline <revision> --output /tmp/analytic-scans

The 16 cases span float32/64, one/two sources, two/four channels and time
stencils of width three/five. Each process retains all executables, with no
cache clearing between cases. Compilation caches are disabled. Inputs and a
small backend warmup precede the first mapping count. Runtime samples follow
two warmup calls and wait for device completion. --derivatives also measures
amplitude JVPs and VJPs; those executables contribute to the mapping count.
"""

import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def inputs(np, jax, n_source, n_freq, half_width, double):
    from tabascal.poly_interp import interp_tables, monomial_tables

    real = np.float64 if double else np.float32
    complex_ = np.complex128 if double else np.complex64
    n_time = 2 * half_width + 3
    rng = np.random.default_rng(17)
    shape = (n_source, 3, n_freq, n_time)
    amp = (1 + .1 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))).astype(complex_)
    phase = rng.uniform(-np.pi, np.pi, shape).astype(real)
    delay = np.zeros((n_source, 3, n_time, 4), dtype=real)
    delay[..., 0] = rng.uniform(-.2, .2, delay.shape[:-1])
    # Include slow cells, fast winding and both signs of curvature and cubic.
    delay[:, 0, :, 1] = np.linspace(0., 1000.41, n_time) / 2800
    delay[:, 1, :, 1] = .7 / 2800
    delay[..., 2] = rng.uniform(-6., 6., delay.shape[:-1]) / 1400
    delay[..., 3] = rng.uniform(-.162, .162, delay.shape[:-1]) / 1400
    dnu = np.asarray([-.2, 0., .2], dtype=real)
    wf, sf = interp_tables(n_freq, 1, dnu)
    gt, st = monomial_tables(n_time, half_width)
    args = (amp, phase, delay, wf.astype(real), sf.astype(np.int32),
            gt.astype(real), st.astype(np.int32), dnu, np.asarray(2., dtype=real),
            np.arange(1400, 1400 + n_freq, dtype=real),
            np.asarray([0, 1, 0], dtype=np.int32), np.asarray([1, 0, 2], dtype=np.int32))
    tangent = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(complex_)
    vis_shape = (3, n_freq, n_time)
    cotangent = (rng.normal(size=vis_shape) + 1j * rng.normal(size=vis_shape)).astype(complex_)
    return jax.device_put((args, tangent, cotangent))


def map_counts():
    lines = Path('/proc/self/maps').read_text().splitlines()
    return {'total': len(lines), 'executable': sum('x' in line.split()[1] for line in lines)}


def worker(options):
    sys.path.insert(0, str(ROOT))
    import jax
    import jax.numpy as jnp
    import jaxlib
    import numpy as np

    jax.config.update('jax_enable_compilation_cache', False)
    jax.config.update('jax_default_matmul_precision', 'highest')
    spec = importlib.util.spec_from_file_location('measured_reference', options.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for double, n_source, n_freq, half_width in itertools.product(
        (False, True), (1, 2), (2, 4), (1, 2)
    ):
        jax.config.update('jax_enable_x64', double)
        data = inputs(np, jax, n_source, n_freq, half_width, double)
        jax.block_until_ready(data)
        cases.append((double, n_source, n_freq, half_width, data))

    warmups = []
    for double in (False, True):
        jax.config.update('jax_enable_x64', double)
        x = jnp.ones(())
        compiled = jax.jit(lambda x: x + 1).lower(x).compile()
        jax.block_until_ready(compiled(x))
        warmups.append(compiled)
    start_maps = map_counts()
    report = {
        'jax': jax.__version__, 'jaxlib': jaxlib.__version__,
        'devices': [str(d) for d in jax.devices()],
        'device_kind': [d.device_kind for d in jax.devices()],
        'xla_flags': os.environ.get('XLA_FLAGS', ''),
        'terms': options.terms, 'segments': options.segments,
        'cubic_terms': options.cubic_terms, 'repeats': options.repeats,
        'derivatives': options.derivatives,
        'maps_start': start_maps, 'cases': [],
    }
    arrays, executables = {}, []
    for index, (double, n_source, n_freq, half_width, data) in enumerate(cases):
        jax.config.update('jax_enable_x64', double)
        args, tangent, cotangent = data

        def call(*args):
            return module.analytic_rfi_vis(
                *args, terms=options.terms, segments=options.segments,
                cubic_terms=options.cubic_terms,
            )

        def jvp(tangent, *args):
            return jax.jvp(lambda amp: call(amp, *args[1:]), (args[0],), (tangent,))[1]

        def vjp(cotangent, *args):
            return jax.vjp(lambda amp: call(amp, *args[1:]), args[0])[1](cotangent)[0]

        variants = [('value', call, args)]
        if options.derivatives:
            variants += [('jvp', jvp, (tangent, *args)), ('vjp', vjp, (cotangent, *args))]
        row = {'precision': 64 if double else 32, 'sources': n_source,
               'channels': n_freq, 'half_width': half_width, 'variants': {}}
        for name, function, arguments in variants:
            before = map_counts()
            start = time.perf_counter()
            lowered = jax.jit(function).lower(*arguments)
            lower_seconds = time.perf_counter() - start
            hlo_bytes = len(lowered.as_text().encode())
            start = time.perf_counter()
            compiled = lowered.compile()
            compile_seconds = time.perf_counter() - start
            executables.append(compiled)
            for _ in range(2):
                result = jax.block_until_ready(compiled(*arguments))
            after = map_counts()
            samples = []
            for _ in range(options.repeats):
                start = time.perf_counter()
                result = jax.block_until_ready(compiled(*arguments))
                samples.append(time.perf_counter() - start)
            arrays[f'case_{index}_{name}'] = np.asarray(result)
            memory = compiled.memory_analysis()
            compiled_text = compiled.as_text()
            code_bytes = getattr(memory, 'generated_code_size_in_bytes', None)
            row['variants'][name] = {
                'lower_seconds': lower_seconds, 'compile_seconds': compile_seconds,
                'stablehlo_bytes': hlo_bytes,
                'compiled_hlo_bytes': len(compiled_text.encode()) if compiled_text is not None else None,
                'generated_code_bytes': code_bytes if code_bytes and code_bytes > 0 else None,
                'runtime_median_seconds': statistics.median(samples),
                'runtime_samples_seconds': samples,
                'maps_growth': {key: after[key] - before[key] for key in before},
            }
        row['maps_cumulative_growth'] = {
            key: map_counts()[key] - start_maps[key] for key in start_maps
        }
        report['cases'].append(row)
        print(f'{options.label}: case {index + 1}/16 {row}', flush=True)
    report['maps_end'] = map_counts()
    report['maps_growth'] = {
        key: report['maps_end'][key] - start_maps[key] for key in start_maps
    }
    options.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez(options.output.with_suffix('.npz'), **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', help='Git revision containing the original reference')
    parser.add_argument('--output', type=Path, default=Path('/tmp/analytic-scans'))
    parser.add_argument('--terms', type=int, default=16)
    parser.add_argument('--segments', type=int, default=4)
    parser.add_argument('--cubic-terms', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--derivatives', action='store_true')
    parser.add_argument('--source', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--label', help=argparse.SUPPRESS)
    options = parser.parse_args()
    if not Path('/proc/self/maps').is_file():
        parser.error('Run this measurement on Linux; /proc/self/maps is required.')
    if min(options.terms, options.segments, options.repeats) < 1 or options.cubic_terms < 0:
        parser.error('terms, segments and repeats must be positive; cubic-terms must be nonnegative')
    if options.source:
        worker(options)
        return
    if not options.baseline:
        parser.error('--baseline is required')
    import numpy as np

    options.output.mkdir(parents=True, exist_ok=True)
    baseline = subprocess.check_output(
        ['git', 'rev-parse', '--verify', options.baseline + '^{commit}'], cwd=ROOT, text=True,
    ).strip()
    source = subprocess.check_output(
        ['git', 'show', f'{baseline}:tabascal/coarse_rfi_vis.py'], cwd=ROOT,
    )
    before_source = options.output / 'before.py'
    before_source.write_bytes(source)
    after_source = options.output / 'after.py'
    after_source.write_bytes((ROOT / 'tabascal/coarse_rfi_vis.py').read_bytes())
    for label, path in [('before', before_source), ('after', after_source)]:
        command = [sys.executable, str(Path(__file__).resolve()), '--source', str(path.resolve()),
                   '--label', label, '--output', str((options.output / label).resolve()),
                   '--terms', str(options.terms), '--segments', str(options.segments),
                   '--cubic-terms', str(options.cubic_terms), '--repeats', str(options.repeats)]
        if options.derivatives:
            command.append('--derivatives')
        subprocess.run(command, cwd=ROOT, check=True)
    before = json.loads((options.output / 'before.json').read_text())
    after = json.loads((options.output / 'after.json').read_text())
    comparisons = {}
    with np.load(options.output / 'before.npz') as old, np.load(options.output / 'after.npz') as new:
        for key in old.files:
            x, y = old[key], new[key]
            error = float(np.max(np.abs(x-y)))
            scale = float(np.max(np.abs(x)))
            eps = np.finfo(x.real.dtype).eps
            comparisons[key] = {
                'bitwise_equal': x.tobytes() == y.tobytes(), 'max_abs_difference': error,
                'difference_in_eps_times_peak': error / max(eps * scale, np.finfo(x.real.dtype).tiny),
                'within_32_eps': bool(np.all(np.isfinite(y)) and error <= 32 * eps * scale),
            }
    summary = {'baseline': baseline, 'before_maps_growth': before['maps_growth'],
               'after_maps_growth': after['maps_growth'], 'comparisons': comparisons}
    for metric in ('stablehlo_bytes', 'compiled_hlo_bytes', 'generated_code_bytes',
                   'lower_seconds', 'compile_seconds', 'runtime_median_seconds'):
        summary[metric] = {}
        for label, report in [('before', before), ('after', after)]:
            values = [v[metric] for row in report['cases'] for v in row['variants'].values()]
            summary[metric][label] = sum(values) if all(v is not None for v in values) else None
    (options.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if not all(row['within_32_eps'] for row in comparisons.values()):
        raise SystemExit('Numerical comparison failed; inspect summary.json.')


if __name__ == '__main__':
    main()
