#!/usr/bin/env python3
"""Developer packaging tool: freeze operative asset bytes, never run inference."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
hashes={}
for f in sorted(ROOT.rglob('*')):
 rel=f.relative_to(ROOT).as_posix()
 if not f.is_file() or any(x in {'__pycache__','.git'} for x in f.relative_to(ROOT).parts) or f.suffix=='.pyc':continue
 if rel.startswith(('runtime/envs/','evaluation/','QA/','docs/')) or rel in {'ASSET_MANIFEST.json','FILE_MANIFEST.json','runtime/bootstrap_receipt.json'}:continue
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(8<<20),b''):h.update(b)
 hashes[rel]=h.hexdigest()
(ROOT/'ASSET_MANIFEST.json').write_text(json.dumps({'schema':'vistream-package-assets-v1','files':hashes},indent=2)+'\n')
print(json.dumps({'asset_files':len(hashes)}))
