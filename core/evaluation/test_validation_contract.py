#!/usr/bin/env python3
"""Synthetic format/integrity negative tests; no GT, inference or metric scoring."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import verify_results as v

class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='vistream-format-test-')
        self.base=Path(self.temp.name);self.root=self.base/'generation';self.root.mkdir()
        self.run_id='SYNTHETIC_FORMAT_TEST_NOT_A_MODEL_RUN'
        self.target=v.target();self.receipts=[];self.predictions=[]
        self.write('provenance/inputs.json',{'synthetic_fixture':True,'not_real_inference':True})
        self.write('logs/namespace.json',{'synthetic_fixture':True,'not_real_access_audit':True})
        for tr in self.target['rows']:
            seq=tr['sequence'];idx=np.asarray(tr['frame_indices'],dtype=np.int64)
            poses=np.repeat(np.eye(4)[None],len(idx),axis=0)
            poses[:,0,3]=idx
            previous=None
            for j,stage in enumerate(v.STAGES):
                rel=f'artifacts/{seq}/{stage}.json';self.write(rel,{'synthetic_stage':stage})
                artifacts={rel:v.sha(self.root/rel)}
                for method,producer in v.PRODUCERS.items():
                    if producer==stage:
                        relp=f'predictions/{seq}/{method}.npz';p=self.root/relp;p.parent.mkdir(parents=True,exist_ok=True)
                        pose_key='predicted_c2w' if method=='visual' else 'c2w'
                        np.savez(p,**{pose_key:poses,'frame_indices':idx})
                        artifacts[relp]=v.sha(p)
                        self.predictions.append({'sequence':seq,'method':method,'stage':stage,'path':relp,
                            'pose_key':pose_key,'frame_indices_key':'frame_indices','sha256':v.sha(p)})
                log=f'logs/{seq}_{stage}.log';self.write(log,{'synthetic_log':stage})
                inputs=[dict(kind='code_manifest',**self.ref('provenance/inputs.json'))]
                if previous:inputs.append(dict(kind='generated_artifact',**self.ref(previous)))
                self.receipts.append({'sequence':seq,'stage':stage,'status':'SUCCEEDED','returncode':0,
                    'run_id':self.run_id,'started_unix':float(j*2),'finished_unix':float(j*2+1),
                    'argv':['synthetic-test-only',stage],'inputs':inputs,'artifacts':artifacts,
                    'log':self.ref(log),'isolation':{'gt_visible':False,'historical_predictions_visible':False,'network_disabled':True}})
                previous=rel
        self.source={'schema':'vistream-source-manifest-v1','mode':'fresh_from_sensors',
            'historical_prediction_dependencies':[],
            'resources':[{'role':'code','path':'synthetic-does-not-exist','sha256':'1'*64}]}
        self.audit={'schema':'vistream-generation-access-audit-v1','run_id':self.run_id,'status':'PASS',
            'evidence_mode':'namespace_mounts','stage_coverage':[{'sequence':r['sequence'],'stage':r['stage']} for r in self.receipts],
            'namespace_checks_complete':True,'forbidden_inputs':[],'issues':[],'unresolved_records':[],
            'evidence_files':{'logs/namespace.json':v.sha(self.root/'logs/namespace.json')}}
        self.contract={'schema':'vistream-complete-generation-v1','mode':'fresh_from_sensors','run_id':self.run_id,
            'sequences':list(v.SEQUENCES),'gt_evaluation_performed':False,'historical_prediction_dependencies':[],
            'predictions':self.predictions}
        self.sync()

    def tearDown(self):self.temp.cleanup()
    def write(self,rel,data):
        p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
    def ref(self,rel):return {'path':rel,'sha256':v.sha(self.root/rel)}
    def sync(self):
        for name,body in [('stage_receipts.json',self.receipts),('SOURCE_MANIFEST.json',self.source),('GENERATION_ACCESS_AUDIT.json',self.audit)]:self.write(name,body)
        self.contract.update(stage_receipts=self.ref('stage_receipts.json'),source_manifest=self.ref('SOURCE_MANIFEST.json'),access_audit=self.ref('GENERATION_ACCESS_AUDIT.json'))
        self.write(v.CONTRACT_NAME,self.contract)
    def fake_score(self,delta=0):
        # Store synthetic target-number fixtures to test verification branching;
        # deliberately do not call evaluate(), loadtxt(), or any ATE function.
        seal=v.read_json(self.root/v.SEAL_NAME);pred={(p['sequence'],p['method']):p for p in seal['predictions']};rows=[]
        for tr in self.target['rows']:
            for m in v.METHODS:
                p=pred[(tr['sequence'],m)]
                rows.append({'sequence':tr['sequence'],'method':m,'ate_rmse_m':tr['targets_ate_rmse_m'][m],
                    'evaluation_frames':p['evaluation_frames'],'candidate_sha256':p['sha256'],
                    'frame_indices_sha256_int64_le':p['frame_indices_sha256_int64_le'],'gt_sha256':tr['external_gt_file_sha256']})
        rows[0]['ate_rmse_m']+=delta
        score={'schema':'vistream-kitti-evaluation-v1','status':'SCORED','mode':seal['mode'],'run_id':self.run_id,
            'input_seal_sha256':v.sha(self.root/v.SEAL_NAME),'target_sha256':v.TARGET_SHA,'metrics_sha256':v.METRICS_SHA,
            'gt_opened_only_after_seal':True,'predictions_unchanged_after_scoring':True,
            'first_gt_open_unix':seal['created_unix']+1,'access_audit_status':seal['access_audit_status'],
            'access_evidence_mode':seal['access_evidence_mode'],'rows':rows,
            'macro_mean_ate_rmse_m':{m:sum(r['ate_rmse_m'] for r in rows if r['method']==m)/11 for m in v.METHODS},
            'synthetic_fixture':True}
        report=self.base/'report';report.mkdir();(report/'EVALUATION.json').write_text(json.dumps(score));return report

    def test_complete_format_seals_without_scoring(self):
        result=v.seal_run(self.root);self.assertEqual(result['status'],'SEALED_NOT_SCORED');self.assertEqual(result['stage_count'],132)
    def test_missing_stage_fails(self):
        self.receipts.pop();self.sync()
        with self.assertRaisesRegex(ValueError,'132'):v.seal_run(self.root)
    def test_missing_prediction_fails(self):
        self.predictions.pop();self.sync()
        with self.assertRaisesRegex(ValueError,'33'):v.seal_run(self.root)
    def test_cached_dependency_fails(self):
        self.contract['historical_prediction_dependencies']=['old/base.npz'];self.sync()
        with self.assertRaisesRegex(ValueError,'historical'):v.seal_run(self.root)
    def test_cached_resource_role_fails(self):
        self.source['resources'][0]['role']='cached_prediction';self.sync()
        with self.assertRaisesRegex(ValueError,'source role'):v.seal_run(self.root)
    def test_external_generated_input_fails(self):
        self.receipts[0]['inputs'][0]['kind']='generated_artifact';self.sync()
        with self.assertRaisesRegex(ValueError,'Cached/external'):v.seal_run(self.root)
    def test_empty_audit_cannot_vacuously_pass(self):
        self.audit['stage_coverage']=[];self.audit['evidence_files']={};self.sync()
        with self.assertRaisesRegex(ValueError,'132'):v.seal_run(self.root)
    def test_failed_generator_cannot_be_fallback(self):
        r=next(r for r in self.receipts if r['stage']=='orb_off');r['status']='FAILED';r['returncode']=1;self.sync()
        with self.assertRaisesRegex(ValueError,'Unfinished/failed'):v.seal_run(self.root)
    def test_unsupported_unavailable_fails(self):
        r=next(r for r in self.receipts if r['stage']=='orb_off');r['status']='UNAVAILABLE';r['unavailable_evidence']={'reason':'timeout'};self.sync()
        with self.assertRaisesRegex(ValueError,'unavailable evidence'):v.seal_run(self.root)
    def test_artifact_mutation_after_seal_fails(self):
        v.seal_run(self.root);self.write('artifacts/00/raw_imu.json',{'changed':True})
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):v.validate_seal(self.root)
    def test_wrong_frame_support_fails(self):
        p=self.predictions[0];path=self.root/p['path']
        with np.load(path) as z:poses=z[p['pose_key']].copy();indices=z['frame_indices'].copy()
        indices[-1]-=1;np.savez(path,**{p['pose_key']:poses,'frame_indices':indices});p['sha256']=v.sha(path)
        self.receipts[0]['artifacts'][p['path']]=p['sha256'];self.sync()
        with self.assertRaisesRegex(ValueError,'stride6'):v.seal_run(self.root)
    def test_real_unavailable_schema_can_seal(self):
        r=next(r for r in self.receipts if r['sequence']=='00' and r['stage']=='orb_off');r['status']='UNAVAILABLE'
        raw={}
        for name in ['run.log','FrameTrajectory.txt','KeyFrameTrajectory.txt']:
            rel='raw/00/orb_off/'+name;path=self.root/rel;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('' if name=='KeyFrameTrajectory.txt' else 'SYNTHETIC raw export\n')
            raw[name]={'path':'/work/'+rel,'sha256':v.sha(path),'bytes':path.stat().st_size};r['artifacts'][rel]=v.sha(path)
        native={'status':'UNAVAILABLE','native_returncode':0,'native_completed':True,'export_completed':True,
            'validation_status':'REJECTED','validation_reason':'timestamps_not_strictly_increasing','raw_artifacts':raw}
        rel='raw/00/orb_off/native_stage_receipt.json';self.write(rel,native);r['artifacts'][rel]=v.sha(self.root/rel)
        r['unavailable_evidence']={'native_returncode':0,'native_completed':True,'export_completed':True,
            'validation_status':'REJECTED','reason':'timestamps_not_strictly_increasing','receipt':self.ref(rel)}
        self.sync();self.assertEqual(v.seal_run(self.root)['status'],'SEALED_NOT_SCORED')
    def test_numeric_mismatch_not_pass(self):
        v.seal_run(self.root);report=self.fake_score(delta=1)
        result=v.verify(self.root,report);self.assertEqual(result['status'],'NOT_REPRODUCED');self.assertFalse(result['numeric_target_match'])
    def test_reference_not_mislabeled_fresh(self):
        self.contract['mode']='reference_recompute';self.source['mode']='reference_recompute'
        self.source['resources'][0]['role']='reference_prediction';self.sync();self.contract.pop('stage_receipts');self.write(v.CONTRACT_NAME,self.contract)
        v.seal_run(self.root);report=self.fake_score();result=v.verify(self.root,report)
        self.assertEqual(result['status'],'REFERENCE_RECOMPUTE_MATCH');self.assertFalse(result['fresh_inference_claim'])
    def test_audit_gaps_never_become_full_reproduction(self):
        self.audit.update(status='REVIEW_REQUIRED',issues=[{'gap':'synthetic'}],unresolved_records=[{'gap':'synthetic'}]);self.sync()
        with self.assertRaisesRegex(ValueError,'allow-audit'):v.seal_run(self.root)
        v.seal_run(self.root,True);report=self.fake_score();result=v.verify(self.root,report,True)
        self.assertEqual(result['status'],'REVIEW_REQUIRED_NUMERIC_MATCH')

if __name__=='__main__':unittest.main(verbosity=2)
