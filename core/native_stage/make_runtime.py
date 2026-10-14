#!/usr/bin/env python3
"""Record local native assets under a chosen execution namespace prefix."""
import argparse,json
from pathlib import Path
from portable_common import ROOT,sha,save
PATHS={
 'binary':'prebuilt/ORB_SLAM3/Examples/Monocular/mono_kitti',
 'vocabulary':'prebuilt/ORB_SLAM3/Vocabulary/ORBvoc.txt',
 'orb_library':'prebuilt/ORB_SLAM3/lib/libORB_SLAM3.so',
 'dbow2_library':'prebuilt/ORB_SLAM3/Thirdparty/DBoW2/lib/libDBoW2.so',
 'g2o_library':'prebuilt/ORB_SLAM3/Thirdparty/g2o/lib/libg2o.so',
 'build_record':'target_source/NATIVE_BUILD.json',
 'source_manifest':'vendor/NATIVE_SOURCE_STATE.json',
 'source_diff':'vendor/ORB_WORKTREE_DIFF.patch',
 'shutdown_patch':'target_source/orb_shutdown_barrier.patch',
}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-root',type=Path,default=ROOT);p.add_argument('--execution-prefix',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('New runtime record required')
 prefix=a.execution_prefix.rstrip('/')+'/'
 record={'schema':'VIStream_native_runtime/v1','artifacts':{k:{'path':prefix+rel,'sha256':sha(a.native_root/rel)}for k,rel in PATHS.items()},
  'library_dirs':[prefix+n for n in ['prebuilt/ORB_SLAM3/lib','prebuilt/ORB_SLAM3/Thirdparty/DBoW2/lib','prebuilt/ORB_SLAM3/Thirdparty/g2o/lib','prebuilt/OpenCV-4.5.5/install/lib','prebuilt/Pangolin/install/lib']]}
 save(a.output,record)
if __name__=='__main__':main()
