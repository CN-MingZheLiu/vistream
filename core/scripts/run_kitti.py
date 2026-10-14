#!/usr/bin/env python3
"""Fresh KITTI generation. No GT option, no prediction cache, no network.

One immutable run directory; failures remain failures. The 12 original stages
per sequence are traced with receipts and mount evidence. Evaluation is a
separate command and is impossible before all requested generation is sealed.
"""
import argparse,concurrent.futures,hashlib,json,os,platform,queue,shutil,subprocess,sys,threading,time,uuid
from pathlib import Path
from stage_commands import stages
ROOT=Path(__file__).resolve().parents[1]
PY='/bundle/runtime/envs/lingbot-map/bin/python'
PROJECT='/bundle/source/original_project/VIStream'
SEQS=[f'{i:02d}' for i in range(11)]
STAGES=['visual','raw_imu','loop_geometry','translation','router','dpvo','droid','base','orb_off','fusion_off','orb_on','fusion_on']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.writing')
 tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)

def runtime_ld():
 n='/bundle/native_stage/prebuilt'
 return ':'.join([n+'/ORB_SLAM3/lib',n+'/ORB_SLAM3/Thirdparty/DBoW2/lib',n+'/ORB_SLAM3/Thirdparty/g2o/lib',n+'/Pangolin/lib',n+'/OpenCV-4.5.5/lib','/bundle/runtime/lib/orb-portable','/bundle/tools/lib'])

def sandbox(run,data,imu,gpu,preflight=False):
 bwrap=str(ROOT/'tools/bwrap') if (ROOT/'tools/bwrap').exists() else shutil.which('bwrap')
 if not bwrap:raise FileNotFoundError('bubblewrap is required; packaged tools/bwrap or Ubuntu bubblewrap package')
 cmd=[bwrap,'--die-with-parent','--unshare-pid','--unshare-net','--new-session','--clearenv',
 '--ro-bind','/usr','/usr','--symlink','usr/bin','/bin','--symlink','usr/lib','/lib',
 '--symlink','usr/lib64','/lib64','--symlink','usr/sbin','/sbin','--ro-bind','/sys','/sys',
 '--proc','/proc','--dev-bind','/dev','/dev','--tmpfs','/tmp','--dir','/etc',
 '--ro-bind','/etc/ld.so.cache','/etc/ld.so.cache','--bind',str(run),'/work','--chdir','/work']
 # Mount only operative resources. The environment tar, target table, QA, and
 # original historical evidence are NOT visible to generation processes.
 for rel in ['source','deps','scripts','native_stage','tools','runtime/envs/lingbot-map','runtime/weights','runtime/lib']:
  if not (ROOT/rel).is_dir():raise FileNotFoundError(ROOT/rel)
  cmd+=['--ro-bind',str(ROOT/rel),'/bundle/'+rel]
 # Original ELF RUNPATH values may mention these directories. They resolve
 # exclusively to this package; the original host checkout is never mounted.
 for name in ['ORB_SLAM3','OpenCV-4.5.5','Pangolin']:
  cmd+=['--ro-bind',str(ROOT/'native_stage/prebuilt'/name),'/home/mingzheliu/VIStream/third_party/'+name]
 cmd+=['--ro-bind',str(ROOT/'source/third_party/DROID-SLAM'),'/home/mingzheliu/VIStream/third_party/DROID-SLAM']
 cmd+=['--dir','/sensors/sequences','--dir','/imu']
 for seq in data:
  d=data[seq]
  for name in ['image_2','times.txt','calib.txt']:
   cmd+=['--ro-bind',str(d/name),'/sensors/sequences/'+seq+'/'+name]
  f=imu/('kitti_odo_'+seq)/'imu.csv'
  if f.exists():cmd+=['--ro-bind',str(f),'/imu/kitti_odo_'+seq+'/imu.csv']
 env={'PATH':'/bundle/tools:/bundle/runtime/envs/lingbot-map/bin:/usr/bin:/bin','HOME':'/tmp/home','USER':'reproduction','LOGNAME':'reproduction',
 'LANG':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
 'CUDA_VISIBLE_DEVICES':gpu,'TORCHINDUCTOR_COMPILE_THREADS':'1','TORCHINDUCTOR_CACHE_DIR':'/tmp/torchinductor',
 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','XDG_CACHE_HOME':'/tmp/cache','TORCH_HOME':'/bundle/deps/torch',
 'LD_LIBRARY_PATH':runtime_ld(),'PYTHONPATH':PROJECT+':/bundle/source/third_party/DROID-SLAM:/bundle/source/third_party/DROID-SLAM/droid_slam',
 'GIT_OPTIONAL_LOCKS':'0'}
 for k,v in env.items():cmd+=['--setenv',k,v]
 return cmd

def inventory(root):
 return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--kitti-root',type=Path,required=True,help='KITTI odometry dataset containing sequences/00/image_2 etc. poses/ is never mounted')
 ap.add_argument('--imu-root',type=Path,required=True,help='Directory with kitti_odo_XX/imu.csv (03 has no matching IMU in the target protocol)')
 ap.add_argument('--run-root',type=Path,required=True,help='MUST NOT exist; failed attempts are preserved')
 ap.add_argument('--gpus',default='0',help='Comma-separated NVIDIA GPU UUIDs preferred, or CUDA indices; one sequence worker per GPU')
 ap.add_argument('--native-workers',type=int,default=1,help='CPU native off/on processes; choose based on available CPU/RAM')
 ap.add_argument('--sequences',nargs='+',choices=SEQS,default=SEQS,help='Default all11. Subsets are diagnostics and cannot receive a full reproduction seal.')
 ap.add_argument('--preflight-only',action='store_true',help='Validate all data/resources and imports without inference')
 a=ap.parse_args()
 if platform.system()!='Linux' or platform.machine()!='x86_64':ap.error('Run on Linux x86_64 with an NVIDIA driver. ZIP may be stored on macOS.')
 if a.native_workers<1:ap.error('--native-workers must be positive')
 gpus=a.gpus.split(',')
 if any(not x.strip() for x in gpus) or len(set(gpus))!=len(gpus):ap.error('GPU list must contain distinct nonempty identifiers')
 seqs=sorted(set(a.sequences));run=a.run_root.resolve();imu=a.imu_root.resolve();data={s:(a.kitti_root/'sequences'/s).resolve() for s in seqs}
 if run.is_relative_to(ROOT):raise ValueError('Run outputs must be outside the immutable package')
 if run.exists():raise FileExistsError('Use a NEW --run-root; automatic retry/result selection is prohibited: '+str(run))
 run.mkdir(parents=True);(run/'logs').mkdir();(run/'namespace').mkdir()
 runid=str(uuid.uuid4());save(run/'RUN_PLAN.json',{'schema':'vistream-run-plan-v1','run_id':runid,'mode':'fresh_from_sensors','sequences':seqs,'stages':STAGES,'gpus':gpus,'native_workers':a.native_workers,'target':'2026-10-12 repaired KITTI configuration','gt_mounted':False,'numerical_reproduction_not_assumed':True})
 print('Validating packaged assets and input sensors...',flush=True)
 asset_path=ROOT/'ASSET_MANIFEST.json';asset=json.loads(asset_path.read_text());code_files=asset['files']
 for rel,expected in code_files.items():
  p=ROOT/rel
  if not p.is_file() or sha(p)!=expected:raise ValueError('Packaged asset missing/changed: '+rel)
 # The runtime is restored into this extraction and recorded, including native
 # extensions. These hashes distinguish it from any unrelated installed env.
 env=ROOT/'runtime/envs/lingbot-map'
 if not (env/'bin/python').exists():raise FileNotFoundError('First run python3 scripts/bootstrap.py')
 envfiles=inventory(env);save(run/'provenance/runtime.json',{'role':'runtime','root':str(env),'files':envfiles})
 save(run/'provenance/assets.json',{'role':'code','root':str(ROOT),'asset_manifest_sha256':sha(asset_path),'files':code_files})
 sensors=[]
 for seq,d in data.items():
  if not all((d/x).exists() for x in ['image_2','times.txt','calib.txt']):raise FileNotFoundError('Incomplete KITTI sequence '+seq)
  frames=sorted((d/'image_2').glob('*.png'));n=len((d/'times.txt').read_text().strip().splitlines())
  if len(frames)!=n or [p.name for p in frames]!=[f'{i:06d}.png' for i in range(n)]:raise ValueError('KITTI images/timestamps do not have complete contiguous support: '+seq)
  f=imu/('kitti_odo_'+seq)/'imu.csv'
  if seq!='03' and not f.is_file():raise FileNotFoundError('Missing required IMU for sequence '+seq)
  if seq=='03' and f.exists():raise ValueError('Target protocol uses missing-matching-IMU fallback on 03; supplied 03 IMU changes the configuration')
  files=frames+[d/'times.txt',d/'calib.txt']+([f] if f.exists() else [])
  sensors.append({'sequence':seq,'frames':n,'imu_present':f.exists(),'role':'sensor','files':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]})
 save(run/'provenance/sensors.json',{'role':'sensor','sequences':sensors,'scope':'raw RGB,timestamps,calibration and raw IMU only; no poses mounted'})
 inputs=[{'kind':kind,'path':path,'sha256':sha(run/path)} for kind,path in [('sensor_manifest','provenance/sensors.json'),('code_manifest','provenance/assets.json'),('runtime_manifest','provenance/runtime.json')]]
 save(run/'SOURCE_MANIFEST.json',{'schema':'vistream-source-manifest-v1','mode':'fresh_from_sensors','historical_prediction_dependencies':[],
 'resources':[{'role':role,'path':path,'sha256':sha(run/path)} for role,path in [('sensor','provenance/sensors.json'),('code','provenance/assets.json'),('runtime','provenance/runtime.json')]]})
 # Initial imports and compiled CUDA modules must succeed inside the same
 # namespace, before a native fallback or any long-running experiment exists.

 for idx,gpu in enumerate(gpus):
  cmd=sandbox(run,data,imu,gpu)+[PY,'/bundle/scripts/namespace_exec.py','--receipt',f'/work/namespace/preflight_{idx}.json','--',PY,'/bundle/scripts/preflight_runtime.py','--output',f'/work/provenance/preflight_runtime_{idx}.json']
  with (run/'logs'/f'preflight_{idx}.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'LD_LIBRARY_PATH':str(ROOT/'tools/lib')})
  if r.returncode:raise RuntimeError('Preflight failed; inspect logs/preflight_'+str(idx)+'.log')
 save(run/'PREFLIGHT.json',{'status':'PASS','gpus':gpus,'full_sequence_inputs':len(seqs),'asset_files':len(code_files),'runtime_files':len(envfiles),'new_inference_started':False})
 if a.preflight_only:print('PREFLIGHT PASS (no inference)',flush=True);return
 receipts=[];lock=threading.Lock();failed=threading.Event();gpu_queue=queue.Queue()
 for g in gpus:gpu_queue.put(g)
 futures=[];futures_lock=threading.Lock()
 def stage(seq,name,command,gpu):
  if failed.is_set():raise RuntimeError('Aborted after another stage failure')
  logrel=f'logs/{seq}_{name}.log';proofrel=f'namespace/{seq}_{name}.json'
  out=run/'visual'/seq if name=='visual' else run/'upstream'/seq/name
  if out.exists():raise FileExistsError(out)
  with lock:
   dependencies=[{'kind':'generated_artifact','path':path,'sha256':digest} for r in receipts if r['sequence']==seq for path,digest in r['artifacts'].items() if path.endswith('.npz') or path.endswith('manifest_before_gt.json') or path.endswith('native_stage_receipt.json')]
  started=time.time()
  cmd=sandbox(run,data,imu,gpu)+[PY,'/bundle/scripts/namespace_exec.py','--receipt','/work/'+proofrel,'--',*command]
  print('START',seq,name,flush=True)
  with (run/logrel).open('x') as f:
   result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'LD_LIBRARY_PATH':str(ROOT/'tools/lib')})
  receipt={'sequence':seq,'stage':name,'run_id':runid,'started_unix':started,'finished_unix':time.time(),'argv':cmd,'returncode':result.returncode,'status':'SUCCEEDED' if result.returncode==0 else 'FAILED',
   'inputs':inputs+dependencies,'log':{'path':logrel,'sha256':sha(run/logrel)},'artifacts':{str(p.relative_to(run)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
   'isolation':{'gt_visible':False,'historical_predictions_visible':False,'network_disabled':True},'namespace_receipt':{'path':proofrel,'sha256':sha(run/proofrel)} if (run/proofrel).exists() else None}
  if name.startswith('orb_') and (out/seq/'native_stage_receipt.json').exists():
   nr=out/seq/'native_stage_receipt.json';n=json.loads(nr.read_text())
   if n['status']=='UNAVAILABLE':
    receipt['status']='UNAVAILABLE'
    receipt['unavailable_evidence']={'native_returncode':n['native_returncode'],'native_completed':n['native_completed'],'export_completed':n['export_completed'],'validation_status':n['validation_status'],'reason':n['validation_reason'],'receipt':{'path':str(nr.relative_to(run)),'sha256':sha(nr)}}
    if n['validation_reason']!='timestamps_not_strictly_increasing':receipt['status']='FAILED'
  with lock:receipts.append(receipt);save(run/'stage_receipts.json',receipts)
  print('END',seq,name,receipt['status'],f'{time.time()-started:.1f}s',flush=True)
  if result.returncode or receipt['status']=='FAILED':failed.set();raise RuntimeError('Stage failed: '+seq+' '+name+'; inspect '+logrel)
  return receipt
 def native_pair(seq,mode,ready,gpu):
  settings='KITTI00-02.yaml' if int(seq)<3 else 'KITTI03.yaml' if seq=='03' else 'KITTI04-12.yaml'
  r='/work/upstream/'+seq
  stage(seq,'orb_'+mode,[PY,'/bundle/native_stage/run_native.py','--project-root',PROJECT,'--runtime','/bundle/native_stage/runtime.json','--sequence-dir','/sensors/sequences/'+seq,'--trajectory','/work/visual/'+seq+'/trajectory.npz','--settings','/bundle/deps/orb_settings/'+mode+'/'+settings,'--output-dir',r+'/orb_'+mode+'/'+seq],gpu)
  ready.wait()
  if failed.is_set():return
  stage(seq,'fusion_'+mode,[PY,'/bundle/native_stage/run_fusion.py','--project-root',PROJECT,'--sequence',seq,'--base-root',r+'/base','--orb-root',r+'/orb_'+mode,'--output-dir',r+'/fusion_'+mode],gpu)
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.native_workers) as cpupool:
  def sequence(seq):
   gpu=gpu_queue.get();ready=threading.Event()
   try:
    stage(seq,'visual',[PY,'/bundle/scripts/fresh_visual.py','--sequence',seq],gpu)
    with futures_lock:
     for mode in ['off','on']:futures.append(cpupool.submit(native_pair,seq,mode,ready,gpu))
    for name,cmd in stages(seq):stage(seq,name,cmd,gpu)
   finally:ready.set();gpu_queue.put(gpu)
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as gpupool:
   work=[gpupool.submit(sequence,s) for s in seqs]
   errors=[]
   for f in concurrent.futures.as_completed(work):
    try:f.result()
    except BaseException as e:failed.set();errors.append(repr(e))
  for f in concurrent.futures.as_completed(futures):
   try:f.result()
   except BaseException as e:failed.set();errors.append(repr(e))
 if errors:save(run/'RUN_FAILED.json',{'status':'FAILED','errors':errors});raise RuntimeError(str(errors))
 for rel,expected in code_files.items():
  if sha(ROOT/rel)!=expected:raise ValueError('Packaged asset changed during inference: '+rel)
 # A mount-only audit is explicitly identified as such. It does not make a
 # claim about every syscall, and cannot accidentally hide dropped trace lines.
 evidence={};issues=[]
 for r in receipts:
  p=run/r['namespace_receipt']['path'];d=json.loads(p.read_text());evidence[str(p.relative_to(run))]=sha(p)
  if d['status']!='PASS' or d['visible_forbidden_paths']:issues.append(str(p))
 audit={'schema':'vistream-generation-access-audit-v1','run_id':runid,'status':'PASS' if not issues else 'FAIL','evidence_mode':'namespace_mounts','stage_coverage':[{'sequence':r['sequence'],'stage':r['stage']} for r in receipts],
 'namespace_checks_complete':True,'forbidden_inputs':issues,'issues':issues,'unresolved_records':[],'evidence_files':evidence}
 save(run/'GENERATION_ACCESS_AUDIT.json',audit)
 preds=[]
 for seq in seqs:
  for method,st,path,key in [('visual','visual',f'visual/{seq}/trajectory.npz','predicted_c2w'),('loop_off','fusion_off',f'upstream/{seq}/fusion_off/{seq}/candidate_before_gt.npz','c2w'),('loop_on','fusion_on',f'upstream/{seq}/fusion_on/{seq}/candidate_before_gt.npz','c2w')]:
   preds.append({'sequence':seq,'method':method,'stage':st,'path':path,'pose_key':key,'frame_indices_key':'frame_indices','sha256':sha(run/path)})
 contract={'schema':'vistream-complete-generation-v1','mode':'fresh_from_sensors','run_id':runid,'sequences':seqs,'gt_evaluation_performed':False,'historical_prediction_dependencies':[],
 'stage_receipts':{'path':'stage_receipts.json','sha256':sha(run/'stage_receipts.json')},'source_manifest':{'path':'SOURCE_MANIFEST.json','sha256':sha(run/'SOURCE_MANIFEST.json')},'access_audit':{'path':'GENERATION_ACCESS_AUDIT.json','sha256':sha(run/'GENERATION_ACCESS_AUDIT.json')},'predictions':preds}
 save(run/'GENERATION_CONTRACT.json',contract)
 print('GENERATION COMPLETE. Numerical reproduction has not been evaluated.',flush=True)
 if seqs!=SEQS:
  save(run/'DIAGNOSTIC_SUBSET.json',{'status':'COMPLETED_SUBSET_ONLY','sequences':seqs,'full_reproduction':False});return
 subprocess.run([str(env/'bin/python'),str(ROOT/'evaluation/verify_results.py'),'seal','--run-root',str(run)],check=True)
if __name__=='__main__':main()
