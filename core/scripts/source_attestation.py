"""Relocated-source attestation. No synthesized git repository or code changes.

Only the legacy wrappers' *source provenance query* is replaced. The returned
upstream revision is a recorded source origin, not the commit of this package.
Every actual packaged file is checked before that origin is accepted.
"""
import hashlib,json
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def apply_attestation(module,name,bundle):
 key={'training_free_dpvo_expert':'DPVO','training_free_droid_expert':'DROID'}.get(name)
 if not key:return
 p=bundle/'deps/source_provenance.json';data=json.loads(p.read_text())
 component=data['components'][key];root=bundle/component['root']
 if not component['files']:raise ValueError('Empty source attestation')
 for rel,expected in component['files'].items():
  file=(root/rel).resolve()
  if not file.is_relative_to(root.resolve()) or sha(file)!=expected:raise ValueError('Source changed: '+str(file))
 def state(requested):
  if Path(requested).resolve()!=root.resolve():raise ValueError('Wrong source root')
  return tuple(data['state_DPVO']) if key=='DPVO' else data['state_DROID']
 setattr(module,'dpvo_source_state' if key=='DPVO' else 'droid_source_state',state)
 print(json.dumps({'source_attestation':'packaged file hashes; recorded upstream origin, no .git repository', 'component':key,'manifest_sha256':sha(p)}),flush=True)
