#!/usr/bin/env python3
"""Record actual generation namespace, reject forbidden inputs, then exec."""
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);p.add_argument('argv',nargs=argparse.REMAINDER);a=p.parse_args()
cmd=a.argv[1:] if a.argv and a.argv[0]=='--' else a.argv
forbidden=['/sensors/poses','/bundle/evidence','/bundle/evaluation','/work/PREDICTIONS_SEALED_BEFORE_GT.json',
'/home/mingzheliu/vistream_kitti_independent_20261012_side', '/home/mingzheliu/vistream_kitti_fix_20261012_side']
visible=[x for x in forbidden if Path(x).exists()]
mountinfo=Path('/proc/self/mountinfo').read_text()
record={'schema':'vistream-namespace-v1','unix':time.time(),'pid':os.getpid(),'argv':cmd,
 'mountinfo':mountinfo,'forbidden_paths':forbidden,'visible_forbidden_paths':visible,
 'network_namespace':os.readlink('/proc/self/ns/net'),'pid_namespace':os.readlink('/proc/self/ns/pid'),
 'environment':dict(os.environ),'status':'PASS' if not visible else 'FAIL'}
Path(a.receipt).write_text(json.dumps(record,indent=2)+'\n')
if visible:raise RuntimeError('Forbidden generation input visible: '+str(visible))
if not cmd:raise ValueError('Missing stage command')
os.execv(cmd[0],cmd)
