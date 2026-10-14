#!/usr/bin/env python3
"""Run the packaged expert with an explicitly audited shutdown-only native overlay."""
import sys,json,hashlib,importlib.util,os
from pathlib import Path
PROJECT=Path('/data2/mingzheliu/vistream_code_zip_validation_20261008/VIStream_Code_Reproduction_20261008/source/original_project/VIStream')
sys.path.insert(0,str(PROJECT))
spec=importlib.util.spec_from_file_location('orb_expert',PROJECT/'scripts/training_free_orbslam3_expert.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
args=m.parse_args()
record=Path('/repair/NATIVE_BUILD.json'); build=json.loads(record.read_text())
if m.sha256(args.orb_library)!=build['patched_library_sha256']:
    raise ValueError('Native fix SHA256 differs from recorded build')
loaded={v['name']:v for v in m.dynamic_libraries(args.binary)}
if loaded['libORB_SLAM3.so']['sha256']!=build['patched_library_sha256']:
    raise ValueError('Native process would load the old shared library')
original_run=m.run_orbslam3

def checked_run(*a,**kw):
    result=original_run(*a,**kw)
    text=result[2].read_text()
    ready=text.find('VIStream shutdown barrier: complete after ')
    export=text.find('Saving trajectory to FrameTrajectory.txt')
    if ready<0 or export<ready:
        raise ValueError('Trajectory was exported before shutdown barrier completed')
    return result
m.run_orbslam3=checked_run
m.generate(args)
p=args.output_dir/'manifest_before_gt.json'
x=json.loads(p.read_text());x['native_overlay']={'build_record':m.artifact(record),'library':m.artifact(args.orb_library),'patch':m.artifact(Path('/repair/orb_shutdown_barrier.patch')),'scope':'Shutdown synchronization only; original native objects and settings otherwise unchanged'}
p.write_text(json.dumps(x,indent=2)+'\n')
