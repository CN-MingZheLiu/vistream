#!/usr/bin/env python3
"""Rebuild the target native stack from bundled complete working-tree sources.

Linux only. --plan-only performs extraction, per-file source verification and
exact patch replay without compilation. A new work root is always required.
A rebuild is source-equivalent; different compilers/CPUs need not reproduce the
archived .so bytes or asynchronous ORB trajectories exactly.
"""
import argparse, json, platform, subprocess, tarfile, time
from pathlib import Path
from portable_common import ROOT, artifact, save, sha

def safe_extract(archive,dest):
    with tarfile.open(archive,'r:gz') as t:
        for m in t.getmembers():
            rel=Path(m.name)
            if rel.is_absolute() or '..' in rel.parts or m.islnk() or m.issym():
                raise ValueError('Unsafe or linked source archive member: '+m.name)
        t.extractall(dest)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--vocabulary',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();work=a.work_root.resolve()
    if a.jobs<1:p.error('jobs must be >=1')
    if work.exists():raise FileExistsError('A new build root is required')
    work.mkdir(parents=True);src=work/'source';src.mkdir()
    archive=ROOT/'vendor/native_sources.tar.gz'
    lock=json.loads((ROOT/'vendor/SOURCE_ARCHIVE.json').read_text())
    if sha(archive)!=lock['sha256']:raise ValueError('Bundled native source archive changed')
    safe_extract(archive,src)
    source_manifest=src/'NATIVE_SOURCE_MANIFEST.json';manifest=json.loads(source_manifest.read_text())
    for name,state in manifest.items():
        for rel,r in state['files'].items():
            if sha(src/name/rel)!=r['sha256']:raise ValueError('Source extraction mismatch '+name+'/'+rel)
    orb=src/'ORB_SLAM3';original_system=orb/'src/System.cc'
    target_record=json.loads((ROOT/'target_source/NATIVE_BUILD.json').read_text())
    if sha(original_system)!=target_record['original_source_sha256']:raise ValueError('Target patch baseline mismatch')
    subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(ROOT/'target_source/orb_shutdown_barrier.patch')],cwd=orb,check=True)
    if sha(original_system)!=target_record['patched_source_sha256']:raise ValueError('Patch replay differs from target System.cc')
    prefix=work/'install';opencv=src/'OpenCV-4.5.5';pangolin=src/'Pangolin'
    commands=[]
    def cmake(source,build,flags,targets=None,install=False):
        commands.append(['cmake','-S',str(source),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',*flags])
        cmd=['cmake','--build',str(build),'--parallel',str(a.jobs)]
        if targets:cmd.extend(['--target',*targets])
        commands.append(cmd)
        if install:commands.append(['cmake','--install',str(build)])
    cvflags=['-DCMAKE_INSTALL_PREFIX='+str(prefix/'opencv'),'-DBUILD_LIST=core,imgproc,imgcodecs,features2d,calib3d,video,highgui',
             '-DBUILD_TESTS=OFF','-DBUILD_PERF_TESTS=OFF','-DBUILD_EXAMPLES=OFF','-DBUILD_opencv_apps=OFF',
             '-DBUILD_opencv_python2=OFF','-DBUILD_opencv_python3=OFF','-DWITH_FFMPEG=OFF','-DWITH_GSTREAMER=OFF','-DWITH_GTK=OFF',
             '-DWITH_IPP=OFF','-DWITH_OPENCL=OFF','-DWITH_OPENEXR=OFF']
    cmake(opencv,work/'build/opencv',cvflags,install=True)
    cmake(pangolin,work/'build/pangolin',['-DCMAKE_INSTALL_PREFIX='+str(prefix/'pangolin'),'-DBUILD_EXAMPLES=OFF','-DBUILD_TOOLS=OFF','-DBUILD_PANGOLIN_PYTHON=OFF'],install=True)
    depflags=['-DCMAKE_PREFIX_PATH='+str(prefix/'opencv')+';'+str(prefix/'pangolin'),'-DOpenCV_DIR='+str(prefix/'opencv/lib/cmake/opencv4')]
    cmake(orb/'Thirdparty/DBoW2',work/'build/DBoW2',depflags)
    cmake(orb,work/'build/orb',depflags,targets=['mono_kitti'])
    plan={'schema':'VIStream_native_source_build/v1','started_unix':time.time(),'source_archive':artifact(archive),'source_manifest':artifact(source_manifest),
          'patch':artifact(ROOT/'target_source/orb_shutdown_barrier.patch'),'patched_system':artifact(original_system),'source_verification':'PASS',
          'patch_replay':'BYTE_EXACT_TARGET','commands':commands,'original_target_build':artifact(ROOT/'target_source/NATIVE_BUILD.json'),'plan_only':a.plan_only,
          'limitation':'Compiler/CPU/system dependency differences can change bytes and asynchronous native results; this rebuild is not asserted bit-reproducible.'}
    save(work/'BUILD_PLAN.json',plan)
    if a.plan_only:print(json.dumps({'status':'SOURCE_AND_PATCH_VERIFIED','plan':str(work/'BUILD_PLAN.json')}));return
    if platform.system()!='Linux':raise RuntimeError('Native target builds require Linux; use --plan-only for portable source verification')
    with (work/'build.log').open('x') as log:
        for command in commands:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    library=orb/'lib/libORB_SLAM3.so';binary=orb/'Examples/Monocular/mono_kitti'
    record={**plan,'finished_unix':time.time(),'build_status':'SUCCEEDED','patched_library_sha256':sha(library),
            'original_source_sha256':target_record['original_source_sha256'],'patched_source_sha256':target_record['patched_source_sha256'],
            'binary':artifact(binary),'library':artifact(library)}
    build_record=work/'NATIVE_BUILD.json';save(build_record,record)
    artifacts={'binary':binary,'vocabulary':a.vocabulary,'orb_library':library,'dbow2_library':orb/'Thirdparty/DBoW2/lib/libDBoW2.so',
               'g2o_library':orb/'Thirdparty/g2o/lib/libg2o.so','build_record':build_record,'source_manifest':source_manifest,
               'source_diff':orb/'WORKTREE_DIFF.patch','shutdown_patch':ROOT/'target_source/orb_shutdown_barrier.patch'}
    runtime={'schema':'VIStream_native_runtime/v1','artifacts':{n:artifact(v) for n,v in artifacts.items()},
             'library_dirs':[str(x) for x in [orb/'lib',orb/'Thirdparty/DBoW2/lib',orb/'Thirdparty/g2o/lib',prefix/'opencv/lib',prefix/'pangolin/lib']]}
    save(work/'runtime.json',runtime);print(json.dumps({'status':'BUILT','runtime':str(work/'runtime.json')}))
if __name__=='__main__':main()
