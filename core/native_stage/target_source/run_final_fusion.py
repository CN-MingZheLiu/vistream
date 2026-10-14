#!/usr/bin/env python3
"""Final repair: original mixture plus reference-aware, GT-free admission."""
from pathlib import Path
import sys,importlib.util,json
PROJECT=Path('/data2/mingzheliu/vistream_code_zip_validation_20261008/VIStream_Code_Reproduction_20261008/source/original_project/VIStream')
sys.path[:0]=[str(PROJECT),'/repair']
from admission_reference import admission_diagnostic,apply_geometry_admission,POLICY
seq=sys.argv[1];sys.argv=sys.argv[:1]+sys.argv[2:]
spec=importlib.util.spec_from_file_location('original_fusion',PROJECT/'scripts/training_free_orb_consensus_moe.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.SEQUENCES=(seq,)
args=m.parse_args();original_base=m.load_base_entry;original_orb=m.load_orb_entry;original_gate=m.frozen_orb_gate;state={}
def load_base(entry):
    result=original_base(entry)  # verifies the base, visual and diagnostics hashes
    state['base']=result[0];state['reference']=json.loads(Path(entry['diagnostics']).read_text());return result
def load_orb(*a,**kw):
    result=original_orb(*a,**kw);state['diagnostic']=admission_diagnostic(state['base'],result[0],state['reference']);return result
def gate(features,policy):
    return apply_geometry_admission(original_gate(features,policy),state['diagnostic'])
m.load_base_entry=load_base;m.load_orb_entry=load_orb;m.frozen_orb_gate=gate
m.generate(args)
p=args.output_dir/'manifest_before_gt.json';x=json.loads(p.read_text())
names=['admission_reference.py','geometry_admission_local.py','geometry_admission.py','run_final_fusion.py']
x['repair_admission']={'policy':POLICY,'files':{name:m.sha256(Path('/repair')/name)for name in names},'applicability':'Existing consensus router selected its native DPVO head; visual-step direction heads are not native-scale references','description':'New offline KITTI development repair; admitted original mixture weights unchanged; distinct from the historical paper configuration'}
p.write_text(json.dumps(x,indent=2)+'\n')
