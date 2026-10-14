#!/usr/bin/env python3
"""Original mixture plus byte-exact side admission, from this run's outputs."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
from portable_common import ROOT, artifact, project_module, save, sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','base-root','orb-root','output-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sequence',required=True,choices=[f'{i:02d}' for i in range(11)])
    a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError('A new fusion output directory is required')
    started=time.time();orbdir=a.orb_root/a.sequence
    receipt_path=orbdir/'native_stage_receipt.json';receipt=json.loads(receipt_path.read_text())
    if receipt['status'] not in ['SUCCEEDED','UNAVAILABLE'] or receipt['native_returncode']!=0 or not receipt['native_export_complete']:
        raise ValueError('Missing completed native generation; refusing silent fallback')
    for value in receipt['raw_artifacts'].values():
        if sha(value['path'])!=value['sha256']:raise ValueError('Native raw output changed')
    if receipt['status']=='UNAVAILABLE' and (orbdir/'manifest_before_gt.json').exists():raise ValueError('Rejected expert must not acquire a candidate manifest')
    if receipt['status']=='SUCCEEDED' and sha(receipt['manifest']['path'])!=receipt['manifest']['sha256']:raise ValueError('Native manifest changed')
    m=project_module(a.project_root,'scripts/training_free_orb_consensus_moe.py','frozen_orb_fusion')
    from admission_reference import admission_diagnostic,apply_geometry_admission,POLICY
    import numpy as np
    m.SEQUENCES=(a.sequence,)
    original_base=m.load_base_entry;original_orb=m.load_orb_entry;original_gate=m.frozen_orb_gate;state={}
    def load_base(entry):
        with np.load(entry['visual'],allow_pickle=False) as z:
            if any('ground_truth' in k.lower() or k.lower().startswith('gt') for k in z.files):raise ValueError('Fresh visual support must exclude GT')
        result=original_base(entry);state['base']=result[0];state['reference']=json.loads(Path(entry['diagnostics']).read_text());return result
    def load_orb(*args,**kwargs):
        result=original_orb(*args,**kwargs);state['diagnostic']=admission_diagnostic(state['base'],result[0],state['reference']);return result
    def gate(features,policy):return apply_geometry_admission(original_gate(features,policy),state['diagnostic'])
    m.load_base_entry=load_base;m.load_orb_entry=load_orb;m.frozen_orb_gate=gate
    policy=a.project_root/'configs/training_free_orb_consensus_moe_fixed.json'
    if sha(policy)!='5b315e9f2d9857fd023faa35b0cf91a75bacdd47bf26942176e19e9f650c9fc9':raise ValueError('Original mixture policy changed')
    m.generate(SimpleNamespace(base_root=a.base_root.resolve(),orb_root=a.orb_root.resolve(),policy=policy.resolve(),output_dir=a.output_dir.resolve()))
    mp=a.output_dir/'manifest_before_gt.json';manifest=json.loads(mp.read_text())
    names=['admission_reference.py','geometry_admission_local.py','geometry_admission.py','run_fusion.py']
    manifest['repair_admission']={'policy':POLICY,'files':{n:sha(ROOT/n) for n in names},
        'description':'Target offline KITTI development repair; unchanged admission math and admitted mixture weights; distinct from historical paper configuration',
        'native_receipt':artifact(receipt_path)}
    save(mp,manifest)
    save(a.output_dir/'fusion_stage_receipt.json',{'schema':'VIStream_fusion_stage_receipt/v1','status':'SUCCEEDED','started_unix':started,'finished_unix':time.time(),'gt_accessed':False,'manifest':artifact(mp),'source':artifact(__file__),'base_manifest':artifact(a.base_root/'manifest_before_gt.json'),'native_receipt':artifact(receipt_path)})
if __name__=='__main__':main()
