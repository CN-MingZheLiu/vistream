#!/usr/bin/env python3
"""Inspect the restored runtime inside the generation namespace; no model run.

CUDA driver/system ABI libraries are host prerequisites. CUDA computation
libraries, Torch and the native Python extensions must resolve inside /bundle.
The process records actual /proc/self/maps after imports and device discovery.
"""
import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

BUNDLE = Path('/bundle')
ENVIRONMENT = BUNDLE / 'runtime/envs/lingbot-map'
COMPUTE_PREFIXES = (
    'libcudart.', 'libcublas', 'libcufft', 'libcurand', 'libcusolver',
    'libcusparse', 'libcudnn', 'libnccl', 'libnvrtc', 'libnvJitLink',
    'libcupti', 'libnvToolsExt', 'libtorch', 'libc10',
)
DRIVER_PREFIXES = ('libcuda.', 'libnvidia-', 'libnvcuvid.')
MODULE_NAMES = (
    'torch', 'numpy', 'scipy', 'cv2', 'pypose', 'lietorch',
    'lietorch_backends', 'lietorch_extras', 'droid_backends',
    'cuda_ba', 'cuda_corr',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def mapped_libraries():
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5].startswith('/'):
            paths.add(fields[5])
    compute, drivers = [], []
    for raw in sorted(paths):
        deleted = raw.endswith(' (deleted)')
        value = raw[:-10] if deleted else raw
        name = Path(value).name
        group = compute if name.startswith(COMPUTE_PREFIXES) else drivers if name.startswith(DRIVER_PREFIXES) else None
        if group is not None:
            resolved = Path(value).resolve()
            group.append({'path': value, 'resolved': str(resolved), 'deleted': deleted,
                          'inside_bundle': resolved.is_relative_to(BUNDLE)})
    return compute, drivers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new JSON receipt path inside the run directory')
    args = parser.parse_args()
    report = {'schema': 'vistream-runtime-preflight-v1', 'status': 'FAILED',
              'scope': 'Imports, native extension origins, CUDA device discovery and loaded-library paths; no model inference or accuracy evaluation',
              'python': str(Path(sys.executable).resolve()), 'modules': {}, 'issues': []}
    try:
        if not Path(sys.executable).resolve().is_relative_to(ENVIRONMENT):
            raise ValueError('Python must run from the packaged environment inside /bundle')
        for name in MODULE_NAMES:
            module = importlib.import_module(name)
            origin = getattr(module, '__file__', None)
            if not origin:
                raise ValueError('Module has no inspectable origin: ' + name)
            path = Path(origin).resolve()
            if not path.is_relative_to(BUNDLE):
                raise ValueError('Module resolves outside package: ' + name + ': ' + str(path))
            report['modules'][name] = {'path': str(path), 'sha256': sha(path)}
        import torch
        report['versions'] = {'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                              'numpy': sys.modules['numpy'].__version__,
                              'scipy': sys.modules['scipy'].__version__, 'opencv': sys.modules['cv2'].__version__}
        report['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        report['cuda_available'] = torch.cuda.is_available()
        if not report['cuda_available']:
            raise RuntimeError('NVIDIA CUDA device unavailable in the generation namespace')
        report['device'] = torch.cuda.get_device_name(0)
        compute, drivers = mapped_libraries()
        report['compute_libraries'] = compute
        report['driver_libraries'] = drivers
        escaped = [x['path'] for x in compute if not x['inside_bundle'] or x['deleted']]
        if escaped:
            raise ValueError('Computation libraries resolve outside the package or were deleted: ' + ', '.join(escaped))
        names = [Path(x['path']).name for x in compute]
        for prefix in ('libcudart.', 'libtorch_cpu.', 'libtorch_cuda.', 'libc10.'):
            if not any(name.startswith(prefix) for name in names):
                raise ValueError('Required loaded library missing from process maps: ' + prefix)
        report['host_library_policy'] = 'Host CUDA driver and system ABI libraries are allowed; CUDA computation and Torch libraries must be bundled.'
        report['status'] = 'PASS'
    except Exception as error:
        report['issues'].append(repr(error))
        try:
            report['compute_libraries'], report['driver_libraries'] = mapped_libraries()
        except OSError:
            pass
    text = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            handle.write(text)
    print(text, end='', flush=True)
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
