#!/usr/bin/env python3
"""Seal complete generation, score with external GT, or verify numerical reproduction.

No inference is performed. Subcommands seal/verify never open GT files. Evaluate
opens GT only after all33 predictions/support and generation evidence are sealed.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
METRICS_SHA = 'd59cc4d8af7b675dfc83a27b0c62212d3055a86885b9e5be918425d88f7bc8f6'
TARGET_SHA = '32c8f29eec646e1a74a5617b22a2a2a761f29eb93de58d13f6e8d2da79c35284'
SEQUENCES = tuple(f'{i:02d}' for i in range(11))
STAGES = ('visual','raw_imu','loop_geometry','translation','router','dpvo','droid','base','orb_off','fusion_off','orb_on','fusion_on')
METHODS = ('visual','loop_off','loop_on')
PRODUCERS = {'visual':'visual','loop_off':'fusion_off','loop_on':'fusion_on'}
INPUT_KINDS = {'sensor_manifest','checkpoint_manifest','code_manifest','runtime_manifest','policy','generated_artifact'}
RESOURCE_ROLES = {'sensor','checkpoint','code','runtime','policy'}
CONTRACT_NAME = 'GENERATION_CONTRACT.json'
SEAL_NAME = 'PREDICTIONS_SEALED_BEFORE_GT.json'
TOLERANCE = 1e-8

def require(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def digest_valid(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def read_json(path):
    with Path(path).open() as f:
        return json.load(f)

def write_json_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def inside(root, relative):
    require(isinstance(relative,str) and bool(relative), 'Artifact path must be a nonempty string')
    p=Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'External/traversal artifact: '+relative)
    q=root
    for part in p.parts:
        q=q/part
        require(not q.is_symlink(), 'Symlink artifact is not allowed: '+relative)
    require(q.is_file() and root in q.resolve().parents, 'Missing/non-file/outside artifact: '+relative)
    return q

def target():
    p=HERE/'target_results.json'
    require(sha(p)==TARGET_SHA, 'Target fixture changed; do not redefine the expected result')
    d=read_json(p)
    require(d['schema']=='vistream-kitti-side-repaired-target-v1', 'Wrong target schema')
    require(d['numeric_match_absolute_tolerance_m']==TOLERANCE, 'Target tolerance changed')
    require([r['sequence'] for r in d['rows']]==list(SEQUENCES), 'Incomplete target sequences')
    require(d['metric_sha256']==METRICS_SHA, 'Metric target changed')
    return d

def validate_generation(root, allow_audit_review=False, load_poses=True):
    root=Path(root).resolve()
    tracked={}
    def register(relative, expected=None):
        p=inside(root,relative);h=sha(p)
        if expected is not None:
            require(digest_valid(expected) and h==expected, 'SHA mismatch: '+relative)
        if relative in tracked:
            require(tracked[relative]==h, 'Input changed while validating: '+relative)
        tracked[relative]=h
        return p
    def referenced(ref):
        require(isinstance(ref,dict) and 'path' in ref and 'sha256' in ref, 'Missing path/SHA reference')
        return register(ref['path'],ref['sha256'])
    cp=register(CONTRACT_NAME);c=read_json(cp)
    require(c.get('schema')=='vistream-complete-generation-v1','Wrong generation schema')
    mode=c.get('mode');require(mode in {'fresh_from_sensors','reference_recompute'},'Explicit fresh/reference mode required')
    require(c.get('sequences')==list(SEQUENCES),'All11 sequences in canonical order required')
    require(isinstance(c.get('run_id'),str) and c['run_id'],'Unique nonempty run_id required')
    require(c.get('gt_evaluation_performed') is False,'Contract already declares GT evaluation')
    sm=read_json(referenced(c['source_manifest']))
    require(sm.get('schema')=='vistream-source-manifest-v1' and sm.get('mode')==mode,'Source manifest mode/schema mismatch')
    resources=sm.get('resources')
    require(isinstance(resources,list) and resources,'Empty source manifest')
    for resource in resources:
        require(isinstance(resource,dict) and resource.get('path') and digest_valid(resource.get('sha256')),'Resource path/SHA missing')
        allowed=RESOURCE_ROLES if mode=='fresh_from_sensors' else RESOURCE_ROLES|{'reference_prediction'}
        require(resource.get('role') in allowed,'Forbidden/unknown source role: '+str(resource.get('role')))
    expected={(seq,stage) for seq in SEQUENCES for stage in STAGES}
    receipts=[];producers={};audit_status='NOT_APPLICABLE_REFERENCE';evidence_mode='reference'
    if mode=='fresh_from_sensors':
        require(c.get('historical_prediction_dependencies')==[] and sm.get('historical_prediction_dependencies')==[],
                'Fresh generation has missing/declared historical prediction dependencies')
        receipts=read_json(referenced(c['stage_receipts']))
        require(isinstance(receipts,list) and len(receipts)==132,'Fresh generation needs132 final receipts')
        pairs=[(r.get('sequence'),r.get('stage')) for r in receipts]
        require(len(set(pairs))==132 and set(pairs)==expected,'Missing/duplicate/unexpected generation stage')
        for r in receipts:
            pair=(r['sequence'],r['stage'])
            require(r.get('run_id')==c['run_id'],'Receipt from a different run: '+str(pair))
            require(isinstance(r.get('started_unix'),(int,float)) and isinstance(r.get('finished_unix'),(int,float))
                    and math.isfinite(r['started_unix']) and math.isfinite(r['finished_unix'])
                    and r['finished_unix']>=r['started_unix'],'Invalid stage times: '+str(pair))
            require(isinstance(r.get('argv'),list) and r['argv'] and all(isinstance(x,str) for x in r['argv']),'Missing executed command: '+str(pair))
            iso=r.get('isolation',{})
            require(iso.get('gt_visible') is False and iso.get('historical_predictions_visible') is False
                    and iso.get('network_disabled') is True,'Generation isolation failed/unknown: '+str(pair))
            require(isinstance(r.get('artifacts'),dict) and r['artifacts'],'Empty artifact record: '+str(pair))
            for rel,h in r['artifacts'].items():
                register(rel,h)
                require(rel not in producers,'An artifact has multiple producers: '+rel)
                producers[rel]=r
            referenced(r['log'])
            require(isinstance(r.get('inputs'),list) and r['inputs'],'Missing stage input provenance: '+str(pair))
            for i in r['inputs']:
                require(i.get('kind') in INPUT_KINDS,'Forbidden/unknown input dependency: '+str(i.get('kind')))
                referenced(i)
            if r.get('status')=='SUCCEEDED':
                require(r.get('returncode')==0,'Successful stage has nonzero returncode: '+str(pair))
            elif r.get('status')=='UNAVAILABLE':
                require(r['stage'] in {'orb_off','orb_on'} and r.get('returncode') in {0,1},'Only genuine ORB validation rejection may be unavailable')
                e=r.get('unavailable_evidence',{})
                require(e.get('native_returncode')==0 and e.get('native_completed') is True
                        and e.get('export_completed') is True and e.get('validation_status')=='REJECTED'
                        and e.get('reason')=='timestamps_not_strictly_increasing','Incomplete/unsupported ORB unavailable evidence')
                native=read_json(referenced(e['receipt']))
                require(native.get('status')=='UNAVAILABLE' and native.get('native_returncode')==0
                        and native.get('native_completed') is True and native.get('export_completed') is True
                        and native.get('validation_status')=='REJECTED'
                        and native.get('validation_reason')==e['reason'],'Native receipt does not support admitted rejection')
                raw=native.get('raw_artifacts',{})
                require(isinstance(raw,dict) and {'run.log','FrameTrajectory.txt','KeyFrameTrajectory.txt'}.issubset(raw),
                        'Rejected native candidate lacks raw export evidence')
                for name in ['run.log','FrameTrajectory.txt','KeyFrameTrajectory.txt']:
                    item=raw[name]
                    require(isinstance(item,dict) and digest_valid(item.get('sha256')) and isinstance(item.get('bytes'),int),
                            'Invalid raw native artifact metadata')
                    matches=[rel for rel,h in r['artifacts'].items() if Path(rel).name==name and h==item['sha256']]
                    require(len(matches)==1,'Raw native export must be uniquely sealed in the same stage: '+name)
                    raw_path=register(matches[0],item['sha256'])
                    require(raw_path.stat().st_size==item['bytes'],'Raw native export byte size changed')
                    if name!='KeyFrameTrajectory.txt':require(item['bytes']>0,'Empty required native export: '+name)
            else:
                raise ValueError('Unfinished/failed stage: '+str(pair)+' '+str(r.get('status')))
        for r in receipts:
            for i in r['inputs']:
                if i['kind']=='generated_artifact':
                    require(i['path'] in producers,'Cached/external output used as generated input: '+i['path'])
                    pr=producers[i['path']]
                    require(pr['sequence']==r['sequence'] and pr['stage']!=r['stage']
                            and pr['finished_unix']<=r['started_unix'],'Generated dependency not from an earlier same-run stage')
        audit=read_json(referenced(c['access_audit']))
        require(audit.get('schema')=='vistream-generation-access-audit-v1' and audit.get('run_id')==c['run_id'],'Wrong access-audit linkage')
        coverage=audit.get('stage_coverage',[])
        pairs=[(r.get('sequence'),r.get('stage')) for r in coverage]
        require(len(pairs)==132 and len(set(pairs))==132 and set(pairs)==expected,'Audit does not cover all132 stages')
        require(audit.get('namespace_checks_complete') is True and audit.get('forbidden_inputs')==[], 'Observed forbidden inputs or incomplete namespace audit')
        evidence_mode=audit.get('evidence_mode')
        require(evidence_mode in {'namespace_and_trace','namespace_mounts'},'Audit must state actual evidence mode')
        require(isinstance(audit.get('evidence_files'),dict) and audit['evidence_files'],'Audit evidence cannot be empty')
        for rel,h in audit['evidence_files'].items():
            require(register(rel,h).stat().st_size>0,'Empty access evidence: '+rel)
        audit_status=audit.get('status')
        unresolved=audit.get('unresolved_records');issues=audit.get('issues')
        require(isinstance(unresolved,list) and isinstance(issues,list),'Audit must preserve issues/unresolved arrays')
        if audit_status=='PASS':
            require(not unresolved and not issues,'Audit PASS contains unresolved evidence')
        else:
            require(audit_status=='REVIEW_REQUIRED' and allow_audit_review and unresolved,
                    'Unresolved audit requires explicit --allow-audit-review-required')
            require(issues==unresolved,'Non-decoding/other audit errors cannot be bypassed')
    else:
        require(not c.get('stage_receipts'),'Reference recompute must not claim fresh generation receipts')

    predictions=c.get('predictions');require(isinstance(predictions,list) and len(predictions)==33,'Exactly33 predictions required')
    pred_pairs=[(p.get('sequence'),p.get('method')) for p in predictions]
    require(len(set(pred_pairs))==33 and set(pred_pairs)=={(s,m) for s in SEQUENCES for m in METHODS},'Missing/duplicate prediction')
    targets=target();support={r['sequence']:r for r in targets['rows']};arrays={};summaries=[]
    for p in predictions:
        key=(p['sequence'],p['method']);seq=p['sequence']
        require(p.get('stage')==PRODUCERS[p['method']],'Wrong prediction producer')
        path=referenced(p)
        if mode=='fresh_from_sensors':
            require(p['path'] in producers and producers[p['path']]['sequence']==seq
                    and producers[p['path']]['stage']==p['stage'],'Prediction absent from producing stage artifact record')
        require(p.get('pose_key')==('predicted_c2w' if p['method']=='visual' else 'c2w')
                and p.get('frame_indices_key')=='frame_indices','Wrong canonical pose/support keys')
        with np.load(path,allow_pickle=False) as z:
            indices=z[p['frame_indices_key']].copy();poses=z[p['pose_key']].copy()
        expected_indices=np.asarray(support[seq]['frame_indices'],dtype=np.int64)
        require(np.issubdtype(indices.dtype,np.integer) and np.array_equal(indices,expected_indices),'Wrong exact stride6+last support: '+str(key))
        require(hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()==support[seq]['frame_indices_sha256_int64_le'],'Frame-index SHA mismatch')
        if poses.shape==(len(indices),3,4):
            expanded=np.repeat(np.eye(4,dtype=poses.dtype)[None],len(indices),axis=0)
            expanded[:,:3,:]=poses;poses=expanded
        require(poses.shape==(len(indices),4,4) and np.isfinite(poses).all(),'Nonfinite/invalid poses: '+str(key))
        require(np.allclose(poses[:,3,:],[0,0,0,1],atol=1e-5,rtol=0),'Invalid homogeneous poses: '+str(key))
        if load_poses: arrays[key]=(poses,indices)
        summaries.append({'sequence':seq,'method':p['method'],'path':p['path'],'sha256':p['sha256'],
                          'evaluation_frames':len(indices),'frame_indices_sha256_int64_le':support[seq]['frame_indices_sha256_int64_le']})
    require(sha(HERE/'metrics_reference.py')==METRICS_SHA,'Historical metric code has changed')
    # Detect file changes during the complete validation pass.
    for rel,h in tracked.items():
        require(sha(inside(root,rel))==h,'Artifact changed during validation: '+rel)
    return {'contract':c,'tracked_files':tracked,'predictions':summaries,'arrays':arrays,'target':targets,
            'access_audit_status':audit_status,'access_evidence_mode':evidence_mode}

def seal_run(root, allow_audit_review=False):
    root=Path(root).resolve();require(not (root/SEAL_NAME).exists(),'Never overwrite an existing prediction seal')
    v=validate_generation(root,allow_audit_review,load_poses=False)
    seal={'schema':'vistream-prediction-seal-v1','mode':v['contract']['mode'],'run_id':v['contract']['run_id'],
          'created_unix':time.time(),'gt_opened_by_this_tool':False,'gt_evaluation_performed':False,
          'stage_count':132 if v['contract']['mode']=='fresh_from_sensors' else 0,'prediction_count':33,
          'files':v['tracked_files'],'predictions':v['predictions'],'target_sha256':TARGET_SHA,
          'metrics_sha256':METRICS_SHA,'access_audit_status':v['access_audit_status'],'access_evidence_mode':v['access_evidence_mode']}
    write_json_new(root/SEAL_NAME,seal)
    return {'status':'SEALED_NOT_SCORED','mode':seal['mode'],'stage_count':seal['stage_count'],'prediction_count':33,'seal_sha256':sha(root/SEAL_NAME)}

def validate_seal(root, allow_audit_review=False, load_poses=True):
    root=Path(root).resolve();seal=read_json(inside(root,SEAL_NAME))
    require(seal.get('schema')=='vistream-prediction-seal-v1' and seal.get('gt_opened_by_this_tool') is False
            and seal.get('gt_evaluation_performed') is False,'Wrong prediction seal')
    require(seal.get('target_sha256')==TARGET_SHA and seal.get('metrics_sha256')==METRICS_SHA,'Seal target/metric changed')
    require(seal.get('prediction_count')==33 and seal.get('stage_count')==(132 if seal.get('mode')=='fresh_from_sensors' else 0),
            'Seal stage/prediction counts are incomplete')
    require(isinstance(seal.get('created_unix'),(int,float)) and math.isfinite(seal['created_unix']), 'Seal creation time missing')
    v=validate_generation(root,allow_audit_review,load_poses)
    require(seal.get('files')==v['tracked_files'] and seal.get('predictions')==v['predictions'],'Generation changed since pre-GT seal')
    require(seal.get('mode')==v['contract']['mode'] and seal.get('run_id')==v['contract']['run_id'],'Seal run/mode mismatch')
    require(seal.get('access_audit_status')==v['access_audit_status'] and seal.get('access_evidence_mode')==v['access_evidence_mode'],'Seal audit mismatch')
    return seal,v

def load_metric():
    path=HERE/'metrics_reference.py';require(sha(path)==METRICS_SHA,'Historical metric changed')
    spec=importlib.util.spec_from_file_location('vistream_pinned_metrics',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.ate_rmse

def evaluate(root, gt_root, report_dir, allow_audit_review=False):
    root=Path(root).resolve();gt_root=Path(gt_root).resolve();report_dir=Path(report_dir).resolve()
    require(not report_dir.exists(),'Scoring requires a new report directory')
    require(root!=report_dir and root not in report_dir.parents,'Report directory must be outside sealed generation')
    require(root!=gt_root and root not in gt_root.parents,'GT must be an external directory')
    require(report_dir!=gt_root and report_dir not in gt_root.parents and gt_root not in report_dir.parents,'GT/report directories cannot contain each other')
    seal,v=validate_seal(root,allow_audit_review,load_poses=True)
    seal_hash=sha(root/SEAL_NAME);metric=load_metric();gt={};gt_hashes={};gt_started=time.time()
    # First GT read occurs only after every generation artifact and all33 arrays
    # have passed support/finite/hash checks above.
    for row in v['target']['rows']:
        seq=row['sequence'];path=gt_root/(seq+'.txt')
        require(path.is_file(),'Missing external GT: '+str(path))
        h=sha(path)
        require(h==row['external_gt_file_sha256'],'External GT differs from the target dataset file SHA: '+seq)
        values=np.loadtxt(path,dtype=np.float64)
        require(values.shape==(row['input_frames'],12) and np.isfinite(values).all(),'GT must contain exactly all raw frames with12 finite values per frame: '+seq)
        require(sha(path)==h,'GT changed while loading: '+seq)
        gt[seq]=values.reshape(-1,3,4);gt_hashes[seq]=h
    pred_lookup={(p['sequence'],p['method']):p for p in v['predictions']};rows=[]
    for seq in SEQUENCES:
        for method in METHODS:
            poses,indices=v['arrays'][(seq,method)]
            score=metric(poses[:,:3,3],gt[seq][indices,:3,3],'sim3')
            value=float(score['ate_rmse']);require(math.isfinite(value),'Nonfinite metric output')
            p=pred_lookup[(seq,method)]
            rows.append({'sequence':seq,'method':method,'ate_rmse_m':value,'sim3_scale':float(score['scale']),
                         'evaluation_frames':len(indices),'candidate_sha256':p['sha256'],
                         'frame_indices_sha256_int64_le':p['frame_indices_sha256_int64_le'],'gt_sha256':gt_hashes[seq]})
    for rel,h in seal['files'].items():require(sha(inside(root,rel))==h,'Sealed generation changed during scoring: '+rel)
    require(sha(root/SEAL_NAME)==seal_hash,'Seal changed during scoring')
    for seq,h in gt_hashes.items():require(sha(gt_root/(seq+'.txt'))==h,'GT changed during scoring: '+seq)
    result={'schema':'vistream-kitti-evaluation-v1','status':'SCORED','mode':seal['mode'],'run_id':seal['run_id'],
            'created_unix':time.time(),'first_gt_open_unix':gt_started,'prediction_count':33,
            'input_seal_sha256':seal_hash,'target_sha256':TARGET_SHA,'metrics_sha256':METRICS_SHA,
            'gt_opened_only_after_seal':True,'predictions_unchanged_after_scoring':True,
            'access_audit_status':v['access_audit_status'],'access_evidence_mode':v['access_evidence_mode'],
            'protocol':v['target']['metric'],'rows':rows,
            'macro_mean_ate_rmse_m':{m:sum(r['ate_rmse_m'] for r in rows if r['method']==m)/11 for m in METHODS},
            'numeric_match_not_yet_verified':True,
            'scope_note':'Reference mode only recomputes cached evidence and is not fresh inference. Fresh completion does not imply target numerical reproduction.'}
    report_dir.mkdir(parents=True,exist_ok=False);write_json_new(report_dir/'EVALUATION.json',result)
    with (report_dir/'EVALUATION.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return {'status':'SCORED','report':str(report_dir),'macro_mean_ate_rmse_m':result['macro_mean_ate_rmse_m']}

def verify(root, report_dir, allow_audit_review=False):
    root=Path(root).resolve();report_dir=Path(report_dir).resolve()
    require(not (report_dir/'VERIFICATION.json').exists(),'Never overwrite a verification report')
    seal,v=validate_seal(root,allow_audit_review,load_poses=False)
    score_path=report_dir/'EVALUATION.json';scores=read_json(score_path)
    require(scores.get('schema')=='vistream-kitti-evaluation-v1' and scores.get('status')=='SCORED','Wrong evaluation report')
    require(scores.get('mode')==seal['mode'] and scores.get('run_id')==seal['run_id'],'Scores belong to another run/mode')
    require(scores.get('input_seal_sha256')==sha(root/SEAL_NAME) and scores.get('target_sha256')==TARGET_SHA
            and scores.get('metrics_sha256')==METRICS_SHA,'Score provenance differs from sealed run')
    require(scores.get('gt_opened_only_after_seal') is True and scores.get('predictions_unchanged_after_scoring') is True,'Scoring did not maintain pre-GT seal')
    require(scores.get('access_audit_status')==v['access_audit_status'] and scores.get('access_evidence_mode')==v['access_evidence_mode'],'Score/audit mismatch')
    require(isinstance(scores.get('first_gt_open_unix'),(int,float)) and scores['first_gt_open_unix']>=seal['created_unix'],'GT scoring preceded seal creation')
    rows=scores.get('rows',[]);pairs=[(r.get('sequence'),r.get('method')) for r in rows]
    require(len(rows)==33 and len(set(pairs))==33 and set(pairs)=={(s,m) for s in SEQUENCES for m in METHODS},'Incomplete score coverage')
    pred={(p['sequence'],p['method']):p for p in v['predictions']};targets={r['sequence']:r for r in v['target']['rows']};comparison=[]
    for r in rows:
        seq,method=r['sequence'],r['method'];p=pred[(seq,method)]
        require(r.get('candidate_sha256')==p['sha256'] and r.get('frame_indices_sha256_int64_le')==p['frame_indices_sha256_int64_le']
                and r.get('evaluation_frames')==p['evaluation_frames'],'Scored prediction/support differs from seal')
        value=r.get('ate_rmse_m');require(isinstance(value,(int,float)) and math.isfinite(value) and value>=0,'Invalid stored score')
        require(digest_valid(r.get('gt_sha256')),'Missing GT hash in existing score report')
        require(r['gt_sha256']==targets[seq]['external_gt_file_sha256'],'Stored score used a different GT dataset file')
        expected=targets[seq]['targets_ate_rmse_m'][method];delta=value-expected
        comparison.append({'sequence':seq,'method':method,'expected_ate_rmse_m':expected,'actual_ate_rmse_m':value,
                           'difference_m':delta,'matches_absolute_1e_8_m':abs(delta)<=TOLERANCE,
                           'matches_display_2dp':format(value,'.2f')==format(expected,'.2f')})
    for seq in SEQUENCES:
        require(len({r['gt_sha256'] for r in rows if r['sequence']==seq})==1,'Methods use different GT for one sequence')
    means={m:sum(r['ate_rmse_m'] for r in rows if r['method']==m)/11 for m in METHODS}
    for m,value in means.items():
        require(m in scores.get('macro_mean_ate_rmse_m',{}) and math.isclose(value,scores['macro_mean_ate_rmse_m'][m],abs_tol=1e-12,rel_tol=1e-13),'Recorded macro mean differs from all11 score rows')
    matched=all(r['matches_absolute_1e_8_m'] for r in comparison)
    if seal['mode']=='reference_recompute':
        status='REFERENCE_RECOMPUTE_MATCH' if matched else 'REFERENCE_RECOMPUTE_MISMATCH'
    elif v['access_audit_status']!='PASS':
        status='REVIEW_REQUIRED_NUMERIC_MATCH' if matched else 'NOT_REPRODUCED_REVIEW_REQUIRED'
    else:
        status='REPRODUCED' if matched else 'NOT_REPRODUCED'
    result={'schema':'vistream-kitti-verification-v1','status':status,'mode':seal['mode'],
            'fresh_inference_claim':seal['mode']=='fresh_from_sensors','execution_complete':True,
            'numeric_target_match':matched,'all33_compared':True,'absolute_tolerance_m':TOLERANCE,
            'per_sequence_2dp_matches':sum(r['matches_display_2dp'] for r in comparison),
            'exact_tolerance_matches':sum(r['matches_absolute_1e_8_m'] for r in comparison),
            'macro_mean_ate_rmse_m':means,'target_macro_mean_ate_rmse_m':v['target']['macro_mean_ate_rmse_m'],
            'access_audit_status':v['access_audit_status'],'access_evidence_mode':v['access_evidence_mode'],
            'input_seal_sha256':sha(root/SEAL_NAME),'evaluation_sha256':sha(score_path),'rows':comparison,
            'limitations':'Matching values is a check of this run only, not a guarantee of future deterministic inference. Reference recomputation cannot establish fresh inference. Mount-only evidence is not exhaustive syscall tracing.'}
    write_json_new(report_dir/'VERIFICATION.json',result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ['seal','evaluate','verify']:
        p=sub.add_parser(name);p.add_argument('--run-root',type=Path,required=True)
        p.add_argument('--allow-audit-review-required',action='store_true')
        if name in {'evaluate','verify'}:p.add_argument('--report-dir',type=Path,required=True)
        if name=='evaluate':p.add_argument('--gt-root',type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.command=='seal':result=seal_run(args.run_root,args.allow_audit_review_required)
        elif args.command=='evaluate':result=evaluate(args.run_root,args.gt_root,args.report_dir,args.allow_audit_review_required)
        else:result=verify(args.run_root,args.report_dir,args.allow_audit_review_required)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2,allow_nan=False))
        if args.command=='verify' and result['status'] not in {'REPRODUCED','REFERENCE_RECOMPUTE_MATCH'}:return 3
        return 0
    except (ValueError,KeyError,TypeError,OSError,AssertionError) as error:
        print(json.dumps({'status':'FAILED','command':args.command,'error':str(error)},ensure_ascii=False),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
