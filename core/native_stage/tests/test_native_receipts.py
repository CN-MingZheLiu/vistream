#!/usr/bin/env python3
"""Exercise administration with a synthetic executable, never the ORB binary."""
import argparse,contextlib,io,json,sys,tempfile,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from portable_common import artifact,save,sha,project_module
import run_native

def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    checks={};original_module=run_native.project_module;original_runtime=run_native.runtime_paths;old_argv=sys.argv
    try:
      with tempfile.TemporaryDirectory(prefix='vistream_native_receipt_tests_') as td:
        temp=Path(td);seq=temp/'sequence';seq.mkdir();(seq/'image_2').mkdir()
        times=np.arange(5,dtype=float);np.savetxt(seq/'times.txt',times)
        (seq/'calib.txt').write_text('P2: 100 0 50 0 0 100 40 0 0 0 1 0\n')
        for i in range(5):(seq/'image_2'/f'{i:06d}.png').write_bytes(b'SYNTHETIC TEST ONLY; never decoded')
        settings=temp/'settings.yaml';settings.write_text('Camera1.fx: 100\nCamera1.fy: 100\nCamera1.cx: 50\nCamera1.cy: 40\n')
        visual=temp/'visual.npz';poses=np.repeat(np.eye(4)[None],5,axis=0);poses[:,0,3]=times
        np.savez_compressed(visual,predicted_c2w=poses,frame_indices=np.arange(5),timestamps=times)
        runtime=temp/'runtime.json';save(runtime,{'test_fixture':True})
        dummy=temp/'dummy';dummy.write_text('SYNTHETIC TEST ONLY')
        build=temp/'build.json';save(build,{'patched_library_sha256':sha(dummy)})
        paths={k:dummy for k in ['vocabulary','orb_library','dbow2_library','g2o_library','source_manifest','source_diff','shutdown_patch']};paths['build_record']=build
        for case in ['accepted','duplicate','crash','timeout']:
            executable=temp/(case+'.py')
            prelude='#!/usr/bin/env python3\nfrom pathlib import Path\nimport time,sys\n'
            if case=='crash':body='raise SystemExit(2)\n'
            elif case=='timeout':body='time.sleep(5)\n'
            else:
                rows='\n'.join(f'{(i if case=="accepted"else min(i,3))*1e9:.0f} {i} 0 0 0 0 0 1'for i in range(5))+'\n'
                body="print('VIStream shutdown barrier: complete after 0 seconds',flush=True)\nprint('Saving trajectory to FrameTrajectory.txt',flush=True)\nPath('FrameTrajectory.txt').write_text("+repr(rows)+")\nPath('KeyFrameTrajectory.txt').write_text('')\nprint('End of saving trajectory',flush=True)\n"
            executable.write_text(prelude+body);executable.chmod(0o755);paths['binary']=executable
            m=project_module(a.project_root,'scripts/training_free_orbslam3_expert.py','frozen_expert_'+case)
            m.dynamic_libraries=lambda binary:[{'name':'libORB_SLAM3.so','path':str(dummy),'resolved_path':str(dummy),'sha256':sha(dummy)}]
            run_native.project_module=lambda *args,**kwargs:m
            run_native.runtime_paths=lambda *args:({'test_fixture':True},dict(paths))
            out=temp/('out_'+case)
            sys.argv=['run_native.py','--project-root',str(a.project_root),'--runtime',str(runtime),'--sequence-dir',str(seq),'--trajectory',str(visual),'--settings',str(settings),'--output-dir',str(out),'--timeout-seconds','1']
            with contextlib.redirect_stdout(io.StringIO()):
                try:run_native.main();exitcode=0
                except SystemExit as e:exitcode=e.code
            receipt=json.loads((out/'native_stage_receipt.json').read_text())
            if case=='accepted':assert exitcode==0 and receipt['status']=='SUCCEEDED' and receipt['native_completed']
            elif case=='duplicate':
                assert exitcode==0 and receipt['status']=='UNAVAILABLE' and receipt['native_returncode']==0 and receipt['native_completed'] and receipt['validation_reason']=='timestamps_not_strictly_increasing'
                assert not (out/'manifest_before_gt.json').exists()
                assert (out/'raw_before_gt/FrameTrajectory.txt').read_text()==rows
            elif case=='crash':assert exitcode==1 and receipt['status']=='FAILED' and receipt['native_returncode']==2
            else:assert exitcode==1 and receipt['status']=='FAILED' and receipt['native_returncode']==-9 and not receipt['export_completed']
            checks[case]={'status':receipt['status'],'exitcode':exitcode,'native_returncode':receipt['native_returncode']}
    finally:
        run_native.project_module=original_module;run_native.runtime_paths=original_runtime;sys.argv=old_argv
    save(a.report,{'schema':'VIStream_native_administration_tests/v1','status':'PASS','checks':checks,'native_executable':'Synthetic Python emitter; not ORB','dataset_or_GT_read':False,'real_native_inference_executed':False,'source':artifact(__file__)})
    print(json.dumps({'status':'PASS','cases':len(checks),'report':str(a.report)}))
if __name__=='__main__':main()
