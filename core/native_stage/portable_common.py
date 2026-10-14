"""Portable administration only. Numerical functions remain in the frozen project."""
import hashlib, importlib.util, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PROJECT_SOURCE_HASHES = {
    'scripts/training_free_orbslam3_expert.py': 'a4f40926998d15d49012df332c1fae301a5b30488d08427b935d42d9ac0b60f2',
    'scripts/training_free_orb_consensus_moe.py': '31706bedb703fdc677252c9dceb6dc72d47098ca94f3221851be629764713a28',
    'vistream/metrics.py': 'd59cc4d8af7b675dfc83a27b0c62212d3055a86885b9e5be918425d88f7bc8f6',
}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def artifact(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
ADMISSION_HASHES = {
    'admission_reference.py':'9f377acd7be860ae1a2f52505e53a29edde7845f1b5618b2dedde0ac775e261e',
    'geometry_admission_local.py':'6ad95a38e0363ea50fac5847588a4a8bdadb1bf6779933b3329f2e8553e4c8b6',
    'geometry_admission.py':'5b13ddd85a06a3f7841fa842d6732ac0bb0e7e9331ae77e2070c8623570d2e30',
}
def project_module(project,relative,name):
    for rel,expected in ADMISSION_HASHES.items():
        if sha(ROOT/rel)!=expected:raise ValueError('Target admission source changed: '+rel)
    project=Path(project).resolve()
    for rel,expected in PROJECT_SOURCE_HASHES.items():
        if sha(project/rel)!=expected:raise ValueError('Frozen project source changed: '+rel)
    sys.path[:0]=[str(project),str(ROOT)]
    spec=importlib.util.spec_from_file_location(name,project/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def runtime_paths(path):
    path=Path(path).resolve();record=json.loads(path.read_text());paths={}
    if record.get('schema')!='VIStream_native_runtime/v1':raise ValueError('Wrong native runtime schema')
    for name in ['binary','vocabulary','orb_library','dbow2_library','g2o_library','build_record','source_manifest','source_diff','shutdown_patch']:
        item=record['artifacts'][name];p=Path(item['path'])
        if not p.is_absolute():p=path.parent/p
        p=p.resolve()
        if sha(p)!=item['sha256']:raise ValueError('Native runtime artifact changed: '+name)
        paths[name]=p
    dirs=[]
    for item in record.get('library_dirs',[]):
        p=Path(item);dirs.append(str((path.parent/p).resolve() if not p.is_absolute() else p))
    if dirs:os.environ['LD_LIBRARY_PATH']=':'.join(dirs+([os.environ['LD_LIBRARY_PATH']] if os.environ.get('LD_LIBRARY_PATH') else []))
    return record,paths
