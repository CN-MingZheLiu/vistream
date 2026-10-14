#!/usr/bin/env python3
"""Restore the complete packaged environment offline on Linux x86_64."""
import argparse,hashlib,json,os,platform,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def repair(prefix):
 mapping={
 '/home/mingzheliu/MZL_workplace/lingbot-map':'/bundle/source/original_project/lingbot-map',
 '/home/mingzheliu/MZL_workplace/pypose':'/bundle/source/third_party/pypose',
 '/home/mingzheliu/VIStream/third_party/DROID-SLAM':'/bundle/source/third_party/DROID-SLAM',
 '/data2/mingzheliu/vistream_code_zip_validation_20261008/VIStream_Code_Reproduction_20261008':'/bundle'}
 changed=[]
 for site in prefix.glob('lib/python*/site-packages'):
  for p in list(site.glob('__editable__*finder.py'))+list(site.glob('*.pth'))+list(site.glob('*.egg-link'))+list(site.glob('*.dist-info/direct_url.json')):
   t=p.read_text();u=t
   for old,new in mapping.items():u=u.replace(old,new)
   if u!=t:p.write_text(u);changed.append(str(p.relative_to(prefix)))
 return changed

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repair-existing',action='store_true',help='For the supplied development copy only; normal extraction requires a new prefix');a=p.parse_args()
 if platform.system()!='Linux' or platform.machine()!='x86_64':raise RuntimeError('Requires Linux x86_64. macOS is a download/extraction host only.')
 archive=ROOT/'runtime/environment/lingbot-map-linux-x86_64.tar.gz';prefix=ROOT/'runtime/envs/lingbot-map'
 expected=json.loads((ROOT/'ASSET_MANIFEST.json').read_text())['files']['runtime/environment/lingbot-map-linux-x86_64.tar.gz']
 if sha(archive)!=expected:raise ValueError('Environment archive checksum mismatch')
 if prefix.exists() and not a.repair_existing:raise FileExistsError('Environment already exists; preserve it or use a fresh package extraction.')
 extracted=not prefix.exists()
 if extracted:
  prefix.mkdir(parents=True)
  with tarfile.open(archive,'r:gz') as t:
   for m in t.getmembers():
    dest=(prefix/m.name).resolve()
    if not dest.is_relative_to(prefix.resolve()) or m.isdev():raise ValueError('Unsafe archive member '+m.name)
    if m.issym() or m.islnk():
     target=(dest.parent/m.linkname).resolve() if m.issym() else (prefix/m.linkname).resolve()
     if Path(m.linkname).is_absolute() or not target.is_relative_to(prefix.resolve()):raise ValueError('External archive link '+m.name)
   t.extractall(prefix)
  subprocess.run([str(prefix/'bin/python'),str(prefix/'bin/conda-unpack')],check=True)
 changed=repair(prefix)
 receipt={'status':'READY','archive_sha256':expected,'extracted_from_archive':extracted,'prefix':str(prefix),'editable_paths_are_namespace_paths':True,'modified_editable_files':changed,'network_used':False}
 (ROOT/'runtime/bootstrap_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
