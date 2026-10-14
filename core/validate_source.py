#!/usr/bin/env python3
"""Hash and syntax checks only; never import models or run inference."""
from pathlib import Path
import ast
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent


def main():
    selection = json.loads((ROOT / 'provenance/SELECTION_MANIFEST.json').read_text())
    hash_failures = []
    for row in selection['includedOriginalFiles']:
        p = ROOT / row['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
            hash_failures.append(row['path'])
    parsed = {'core': 0, 'third_party': 0}
    failures = []
    imported_modules = set()
    for p in sorted(ROOT.rglob('*.py')):
        rel = p.relative_to(ROOT).as_posix()
        scope = 'core' if rel.startswith(('source/original_project/VIStream/', 'scripts/', 'evaluation/', 'native_stage/')) or rel == 'validate_source.py' else 'third_party'
        try:
            tree = ast.parse(p.read_bytes(), filename=rel)
            parsed[scope] += 1
            if scope == 'core':
                for n in ast.walk(tree):
                    if isinstance(n, ast.Import):
                        imported_modules.update(a.name.split('.')[0] for a in n.names)
                    elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                        imported_modules.add(n.module.split('.')[0])
        except SyntaxError as e:
            failures.append({'path': rel, 'scope': scope, 'line': e.lineno, 'message': e.msg})
    known_legacy = {
        'source/third_party/DROID-SLAM/thirdparty/lietorch/eigen/scripts/relicense.py',
        'source/third_party/DROID-SLAM/thirdparty/tartanair_tools/download_training.py',
    }
    unexpected = [f for f in failures if f['path'] not in known_legacy]
    report = {
        'status': 'PASS' if not hash_failures and not unexpected else 'FAIL',
        'scope': 'Original-file SHA256 verification, Python AST parsing and static import inventory only. No imports, GPU inference, native compilation, dataset access or accuracy claim.',
        'python_parser_version': sys.version.split()[0],
        'original_files_checked': len(selection['includedOriginalFiles']),
        'hash_failures': hash_failures,
        'python_files_parsed': parsed,
        'preserved_legacy_python2_utilities': [f for f in failures if f['path'] in known_legacy],
        'legacy_note': 'The two unchanged upstream Python 2 auxiliary utilities are not current inference entry points; their Python 3 parse incompatibility is reported rather than edited or hidden.',
        'unexpected_syntax_failures': unexpected,
        'core_import_roots': sorted(imported_modules),
        'import_inventory_note': 'An import name is not a verified installed dependency. Optional scripts also reference external and historical test modules.',
    }
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
