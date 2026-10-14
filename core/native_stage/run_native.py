#!/usr/bin/env python3
"""Generate one fresh native ORB candidate using the exact target shutdown patch.

No GT argument exists. A completed native export rejected by the original
validator is explicit UNAVAILABLE; crashes, dependency failures and timeouts
are FAILED and cannot be silently turned into a model fallback.
"""
import argparse, json, os, signal, subprocess, time, traceback
from pathlib import Path
from types import SimpleNamespace
from portable_common import ROOT, artifact, project_module, runtime_paths, save, sha

VALIDATOR_FAILURE_PREFIXES=(
    'expected EuRoC trajectory [N,8]', 'ORB-SLAM3 trajectory is empty or non-finite',
    'ORB-SLAM3 timestamps are not strictly increasing', 'ORB-SLAM3 returned a zero quaternion',
    'ORB-SLAM3 timestamp does not match KITTI input:', 'multiple ORB-SLAM3 poses map to one KITTI frame',
    'ORB-SLAM3 did not produce exact target frames:',
)
def export_is_complete(raw):
    log=raw/'run.log'
    if not log.is_file():return False
    text=log.read_text(errors='replace')
    barrier=text.find('VIStream shutdown barrier: complete after ')
    export=text.find('Saving trajectory to FrameTrajectory.txt')
    return (barrier>=0 and export>barrier and 'End of saving trajectory' in text[export:]
        and (raw/'FrameTrajectory.txt').is_file() and (raw/'KeyFrameTrajectory.txt').is_file())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','runtime','sequence-dir','trajectory','settings','output-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--timeout-seconds',type=int,default=7200)
    p.add_argument('--pose-key',default='predicted_c2w')
    a=p.parse_args()
    if a.timeout_seconds<=0:p.error('timeout must be positive')
    a.output_dir=a.output_dir.resolve()
    if a.output_dir.exists():raise FileExistsError('A new native output directory is required')
    a.output_dir.mkdir(parents=True)
    receipt={'schema':'VIStream_native_stage_receipt/v1','status':'RUNNING','started_unix':time.time(),
             'gt_accessed':False,'native_returncode':None,'native_started':False,'native_export_complete':False,
             'source':artifact(__file__),'runtime':artifact(a.runtime)}
    raw=a.output_dir/'raw_before_gt'
    try:
        record,paths=runtime_paths(a.runtime)
        m=project_module(a.project_root,'scripts/training_free_orbslam3_expert.py','frozen_orb_expert')
        import numpy as np
        with np.load(a.trajectory,allow_pickle=False) as z:
            if any('ground_truth' in k.lower() or k.lower().startswith('gt') for k in z.files):
                raise ValueError('Generation support must not contain GT members')
        build=json.loads(paths['build_record'].read_text())
        if sha(paths['orb_library'])!=build['patched_library_sha256']:raise ValueError('Unexpected patched native library')
        loaded={x['name']:x for x in m.dynamic_libraries(paths['binary'])}
        if loaded.get('libORB_SLAM3.so',{}).get('sha256')!=build['patched_library_sha256']:
            raise ValueError('ldd resolves a different ORB shared library')
        receipt['runtime_artifacts']={k:artifact(v) for k,v in paths.items()}
        receipt['inputs']={k:artifact(v) for k,v in {'trajectory':a.trajectory,'settings':a.settings,'times':a.sequence_dir/'times.txt','calibration':a.sequence_dir/'calib.txt'}.items()}
        # Metadata replacement only: the bundle carries complete immutable source
        # plus source manifests, rather than relying on the original .git location.
        def source_state(args):
            return {'bundled_source_manifest':artifact(paths['source_manifest']),
                    'native_build':artifact(paths['build_record']),
                    'source_attestation_mode':'relocated bundle; no legacy absolute git paths'},paths['source_diff'].read_bytes()
        m.validate_source_state=source_state
        def controlled_native(binary,vocabulary,settings,sequence_dir,raw_dir,timeout_seconds):
            raw_dir.mkdir(parents=True,exist_ok=False)
            cmd=[str(Path(x).resolve()) for x in [binary,vocabulary,settings,sequence_dir]]
            receipt['native_started']=True;receipt['native_argv']=cmd
            with (raw_dir/'run.log').open('wb') as log:
                proc=subprocess.Popen(cmd,cwd=raw_dir,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                receipt['native_pid']=proc.pid
                try:rc=proc.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait();receipt['native_returncode']=proc.returncode
                    raise TimeoutError('External native timeout; refuses an unfinished export')
            receipt['native_returncode']=rc;receipt['native_export_complete']=export_is_complete(raw_dir)
            if rc:raise subprocess.CalledProcessError(rc,cmd)
            if not receipt['native_export_complete']:raise RuntimeError('Native export did not follow completed shutdown barrier')
            return raw_dir/'FrameTrajectory.txt',raw_dir/'KeyFrameTrajectory.txt',raw_dir/'run.log',cmd
        m.run_orbslam3=controlled_native
        args=SimpleNamespace(trajectory=a.trajectory.resolve(),sequence_dir=a.sequence_dir.resolve(),calibration=(a.sequence_dir/'calib.txt').resolve(),
            settings=a.settings.resolve(),output_dir=a.output_dir,pose_key=a.pose_key,timeout_seconds=a.timeout_seconds,
            timestamp_tolerance=5e-7,intrinsics_tolerance=1e-3,**{k:paths[k] for k in ['binary','vocabulary','orb_library','dbow2_library','g2o_library']})
        # generate requires an empty output directory; receipt is written afterward.
        m.generate(args)
        manifest_path=a.output_dir/'manifest_before_gt.json';manifest=json.loads(manifest_path.read_text())
        manifest['native_overlay']={'build_record':artifact(paths['build_record']),'library':artifact(paths['orb_library']),
            'patch':artifact(paths['shutdown_patch']),'scope':'Exact target shutdown synchronization patch; no tracking/optimization rule change'}
        save(manifest_path,manifest);receipt.update(status='SUCCEEDED',validation_status='ACCEPTED',manifest=artifact(manifest_path))
    except Exception as e:
        known=(isinstance(e,ValueError) and str(e).startswith(VALIDATOR_FAILURE_PREFIXES)
               and receipt['native_returncode']==0 and receipt['native_export_complete']
               and not (a.output_dir/'manifest_before_gt.json').exists())
        receipt.update(status='UNAVAILABLE' if known else 'FAILED',validation_status='REJECTED' if known else 'NOT_COMPLETED',error=repr(e),traceback=traceback.format_exc())
    receipt['native_completed']=bool(receipt['native_returncode']==0 and receipt['native_export_complete'])
    receipt['export_completed']=receipt['native_export_complete']
    receipt['validation_reason']=('timestamps_not_strictly_increasing' if 'timestamps are not strictly increasing' in receipt.get('error','') else receipt.get('validation_status','unknown').lower())
    receipt['finished_unix']=time.time()
    receipt['raw_artifacts']={x.name:artifact(x) for x in raw.glob('*') if x.is_file()} if raw.exists() else {}
    save(a.output_dir/'native_stage_receipt.json',receipt)
    print(json.dumps({'status':receipt['status'],'native_returncode':receipt['native_returncode'],'output_dir':str(a.output_dir)}),flush=True)
    if receipt['status']=='FAILED':raise SystemExit(1)
if __name__=='__main__':main()
