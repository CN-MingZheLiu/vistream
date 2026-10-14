#!/usr/bin/env python3
"""Synthetic contract tests only; no dataset, archived trajectory, or GT input."""
import argparse,importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from portable_common import artifact,save,project_module,sha
from run_native import export_is_complete,VALIDATOR_FAILURE_PREFIXES

def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    original=project_module(a.project_root,'scripts/training_free_orbslam3_expert.py','original_expert_test')
    from admission_reference import admission_diagnostic
    from geometry_admission import apply_geometry_admission
    n=80;base=np.repeat(np.eye(4)[None],n,axis=0);base[:,0,3]=np.arange(n,dtype=float)
    rigid=base.copy();rigid[:,0,3]=2*base[:,0,3]+4
    native={'gate':{'route_enabled':True,'selected_head':'dense_native_open_road','selected_expert':'dpvo'}}
    direction={'gate':{'route_enabled':True,'selected_head':'environment_gated_loop_translation','selected_expert':'dpvo'}}
    warped=base.copy();warped[:,0,3]=np.arange(n,dtype=float)**2
    checks={}
    diag=admission_diagnostic(base,rigid,native);assert diag['valid'] and abs(diag['local_scale_spread']-1)<1e-12
    checks['uniform_scale_native_reference_admitted']=True
    diag=admission_diagnostic(base,warped,native);assert not diag['valid'] and diag['reason']=='nonuniform_relative_motion_scale'
    checks['nonuniform_progress_rejected_without_GT']=True
    diag=admission_diagnostic(base,warped,direction);assert diag['valid'] and not diag['applicable']
    checks['direction_transfer_head_not_used_as_native_scale_reference']=True
    assert admission_diagnostic(base,None,native)['reason']=='orb_unavailable'
    gate=apply_geometry_admission({'orb_weight':0.5},admission_diagnostic(base,warped,native));assert gate['orb_weight']==0 and gate['base_weight']==1
    checks['rejected_candidate_routes_to_unmodified_base']=True
    with tempfile.TemporaryDirectory(prefix='vistream_native_stage_tests_') as td:
        temp=Path(td);raw=temp/'raw';raw.mkdir()
        rows=np.zeros((3,8));rows[:,0]=[0,1e9,1e9];rows[:,7]=1
        np.savetxt(raw/'FrameTrajectory.txt',rows);before=sha(raw/'FrameTrajectory.txt')
        try:original.parse_euroc_trajectory(raw/'FrameTrajectory.txt');raise AssertionError('duplicate timestamps were admitted')
        except ValueError as e:assert str(e)=='ORB-SLAM3 timestamps are not strictly increasing'
        assert sha(raw/'FrameTrajectory.txt')==before;checks['duplicate_timestamps_rejected_without_deduplication']=True
        (raw/'KeyFrameTrajectory.txt').write_text('');(raw/'run.log').write_text('Saving trajectory to FrameTrajectory.txt\nVIStream shutdown barrier: complete after 0 seconds\nEnd of saving trajectory\n')
        assert not export_is_complete(raw)
        (raw/'run.log').write_text('VIStream shutdown barrier: complete after 0 seconds\nSaving trajectory to FrameTrajectory.txt\nEnd of saving trajectory\n')
        assert export_is_complete(raw);checks['export_order_enforced_empty_keyframe_file_not_silently_removed']=True
        # Integration: create an explicitly synthetic new-run support, native receipt,
        # and original manifests. Full CLI uses the pinned original fusion module.
        for name,candidate,reference,expected in [('uniform',rigid,native,'uniform'),('drift',warped,native,'base')]:
            case=temp/name;case.mkdir();b=case/'base';b.mkdir();orbroot=case/'orb';o=orbroot/'01';o.mkdir(parents=True)
            visual=case/'visual.npz';np.savez_compressed(visual,predicted_c2w=base,frame_indices=np.arange(n),timestamps=np.arange(n,dtype=float))
            bc=b/'candidate.npz';np.savez_compressed(bc,c2w=base,frame_indices=np.arange(n))
            bd=b/'diagnostics.json';save(bd,reference)
            entry={'sequence':'01','candidate':str(bc),'candidate_sha256':sha(bc),'diagnostics':str(bd),'diagnostics_sha256':sha(bd),'visual':str(visual),'visual_sha256':sha(visual)}
            save(b/'manifest_before_gt.json',{'entries':[entry]})
            raw=o/'raw_before_gt';raw.mkdir();(raw/'run.log').write_text('Creation of new map with id 0\nVIStream shutdown barrier: complete after 0 seconds\nSaving trajectory to FrameTrajectory.txt\nEnd of saving trajectory\n')
            rr=np.zeros((n,8));rr[:,0]=np.arange(n)*1e9;rr[:,7]=1;rr[:,1]=candidate[:,0,3]
            np.savetxt(raw/'FrameTrajectory.txt',rr);(raw/'KeyFrameTrajectory.txt').write_text('0 0 0 0 0 0 0 1\n')
            oc=o/'candidate_before_gt.npz';np.savez_compressed(oc,orbslam3_c2w=candidate,frame_indices=np.arange(n))
            od=o/'orbslam3_before_gt.json';save(od,{'tracking_completeness':1.0,'orbslam3':{'endpoint_to_path_ratio':1.0}})
            mp=o/'manifest_before_gt.json';save(mp,{'artifacts':{k:artifact(v)for k,v in {'candidate':oc,'diagnostics':od,'raw_frame_trajectory':raw/'FrameTrajectory.txt','run_log':raw/'run.log'}.items()}})
            save(o/'native_stage_receipt.json',{'status':'SUCCEEDED','native_returncode':0,'native_export_complete':True,'raw_artifacts':{v.name:artifact(v)for v in raw.iterdir()},'manifest':artifact(mp),'test_fixture':True})
            result=subprocess.run([sys.executable,str(ROOT/'run_fusion.py'),'--project-root',str(a.project_root.resolve()),'--base-root',str(b),'--orb-root',str(orbroot),'--sequence','01','--output-dir',str(case/'out')],capture_output=True,text=True)
            assert result.returncode==0,result.stderr
            with np.load(case/'out/01/candidate_before_gt.npz') as z:got=z['c2w']
            if expected=='base':assert np.array_equal(got,base)
            else:assert np.allclose(got[:,0,3],2*np.arange(n)) and np.allclose(got[:,:3,:3],np.eye(3))
            checks['full_fusion_CLI_'+name]=True
        # project_module already compared every admission source against the
        # fixed target SHA; no numerical source is modified by this test.
        checks['target_admission_files_byte_identical_to_side']=True
    save(a.report,{'schema':'VIStream_native_stage_synthetic_tests/v1','status':'PASS','checks':checks,'test_count':len(checks),'dataset_or_GT_read':False,'native_inference_executed':False,'full_native_rebuild_executed':False,'source':artifact(__file__)})
    print(json.dumps({'status':'PASS','test_count':len(checks),'report':str(a.report)}))
if __name__=='__main__':main()
