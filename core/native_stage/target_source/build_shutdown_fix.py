#!/usr/bin/env python3
"""Recompile only System.cc, keeping other original native objects immutable."""
from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
ROOT=Path('/home/mingzheliu/vistream_kitti_fix_20261012_side')
ORB=Path('/home/mingzheliu/VIStream/third_party/ORB_SLAM3')
OUT=ROOT/'native_fix'
OUT.mkdir(exist_ok=False)
src=(ORB/'src/System.cc').read_text()
start=src.index('    // Wait until all thread have effectively stopped',src.index('void System::Shutdown()'))
end=src.index('    if(!mStrSaveAtlasToFile.empty())',start)
replacement='''    // No pose may be exported while mapping / loop correction can still write it.
    // Preserve the existing finish requests; wait for workers and pending global BA.
    const auto shutdown_wait_start = std::chrono::steady_clock::now();
    cout << "VIStream shutdown barrier: waiting for mapping, loop closing and GBA" << endl;
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        const double wait_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - shutdown_wait_start).count();
        if(wait_s > 600.0)
            throw std::runtime_error("SLAM shutdown timed out: refusing an unfinished trajectory");
        usleep(5000);
    }
    if(mptLocalMapping && mptLocalMapping->joinable())
        mptLocalMapping->join();
    if(mptLoopClosing && mptLoopClosing->joinable())
        mptLoopClosing->join();
    cout << "VIStream shutdown barrier: complete after "
         << std::chrono::duration<double>(std::chrono::steady_clock::now() - shutdown_wait_start).count()
         << " seconds" << endl;

'''
fixed=src[:start]+replacement+src[end:]
fixed='#include <chrono>\n#include <stdexcept>\n'+fixed
(OUT/'System.cc').write_text(fixed)
(ROOT/'orb_shutdown_barrier.patch').write_text(''.join(difflib.unified_diff(src.splitlines(True),fixed.splitlines(True),fromfile='a/src/System.cc',tofile='b/src/System.cc')))
flags={}
for line in (ORB/'build/CMakeFiles/ORB_SLAM3.dir/flags.make').read_text().splitlines():
    if ' = ' in line:
        k,v=line.split(' = ',1);flags[k]=shlex.split(v)
compile_cmd=['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-c',str(OUT/'System.cc'),'-o',str(OUT/'System.cc.o')]
link_cmd=shlex.split((ORB/'build/CMakeFiles/ORB_SLAM3.dir/link.txt').read_text())
original_objects={}
for i,value in enumerate(link_cmd):
    if value.endswith('.o'):
        p=(ORB/'build'/value).resolve(); original_objects[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        link_cmd[i]=str(OUT/'System.cc.o')if value.endswith('/System.cc.o')else str(p)
link_cmd[link_cmd.index('-o')+1]=str(OUT/'libORB_SLAM3.so')
with (ROOT/'native_build.log').open('x')as log:
    subprocess.run(compile_cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    subprocess.run(link_cmd,cwd=ORB/'build',check=True,stdout=log,stderr=subprocess.STDOUT)
record={'change':'Shutdown synchronization only; unchanged input, tracking, optimization and mixture policy','compile_argv':compile_cmd,'link_argv':link_cmd,'original_objects':original_objects,'original_source_sha256':hashlib.sha256(src.encode()).hexdigest(),'patched_source_sha256':hashlib.sha256(fixed.encode()).hexdigest(),'patched_library_sha256':hashlib.sha256((OUT/'libORB_SLAM3.so').read_bytes()).hexdigest()}
(ROOT/'NATIVE_BUILD.json').write_text(json.dumps(record,indent=2))
print(json.dumps({'build':'PASS','patched_library_sha256':record['patched_library_sha256']}),flush=True)
