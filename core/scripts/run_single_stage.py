#!/usr/bin/env python3
"""Use unmodified release generators on one explicitly selected KITTI sequence."""
import argparse
import importlib.util
from pathlib import Path
import sys

BUNDLE = Path('/bundle')
SCRIPTS = BUNDLE/'source/original_project/VIStream/scripts'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sequence',required=True)
    ap.add_argument('--script',required=True)
    args, remaining=ap.parse_known_args()
    if remaining and remaining[0]=='--':remaining=remaining[1:]
    source=SCRIPTS/(args.script+'.py')
    sys.path.insert(0,str(SCRIPTS))
    spec=importlib.util.spec_from_file_location('selected_generator',source)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    from source_attestation import apply_attestation
    apply_attestation(module, args.script, BUNDLE)
    if hasattr(module,'SEQUENCES'):module.SEQUENCES=(args.sequence,)
    sys.argv=[str(source)]+remaining
    if hasattr(module,'main'):module.main()
    else:module.generate(module.parse_args())


if __name__=='__main__':main()
