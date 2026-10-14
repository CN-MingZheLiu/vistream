#!/usr/bin/env python3
"""Fresh historical KITTI chain: consume only newly generated visual outputs."""
import json
from pathlib import Path
import subprocess
import time
BUNDLE = Path('/bundle')
PY = BUNDLE/'runtime/envs/lingbot-map/bin/python'

PROJECT = BUNDLE/'source/original_project/VIStream'
CONFIG = PROJECT/'configs'
SCRIPTS = PROJECT/'scripts'
THIRD = BUNDLE/'source/third_party'


def stages(seq):
    visual=Path('/work/visual')/seq/'trajectory.npz'
    root=Path('/work/upstream')/seq
    image=Path('/sensors/sequences')/seq
    def config(name):return str(CONFIG/name)
    def entry(script,arguments):
        return [str(PY),'/bundle/scripts/run_single_stage.py','--sequence',seq,'--script',script,'--','generate',*map(str,arguments)]
    tasks=[]
    tasks.append(('raw_imu',entry('evaluate_training_free_kitti_imu',[
        '--cache-root','/work/visual','--image-root','/sensors/sequences','--imu-root','/imu',
        '--calibration-root','/bundle/deps/calibration','--config',config('training_free_local_window32.json'),
        '--output-dir',root/'raw_imu','--visual-convention','c2w'])))
    tasks.append(('loop_geometry',entry('evaluate_kitti_training_free_loop',[
        '--cache-root','/work/visual','--image-root','/sensors/sequences','--pose-key','predicted_c2w',
        '--policy',config('training_free_loop_environment_gate.json'),
        '--loop-script',SCRIPTS/'training_free_loop_closure.py','--device','cuda:0',
        '--batch-size','64','--output-dir',root/'loop_geometry'])))
    tasks.append(('translation',entry('training_free_translation_direction',[
        '--cache-root','/work/visual','--image-root','/sensors/sequences',
        '--policy',config('training_free_translation_direction_fixed.json'),'--output-dir',root/'translation'])))
    tasks.append(('router',entry('training_free_expert_router',[
        '--visual-root','/work/visual','--imu-root',root/'raw_imu','--loop-root',root/'loop_geometry',
        '--descriptor-root',root/'loop_geometry','--translation-root',root/'translation',
        '--policy',config('training_free_expert_router_v6_fixed.json'),'--output-dir',root/'router'])))
    tasks.append(('dpvo',entry('training_free_dpvo_expert',[
        '--trajectory',visual,'--image-dir',image/'image_2','--calibration',image/'calib.txt',
        '--tracking-times',image/'times.txt','--tracking-stride','1',
        '--dpvo-root','/bundle/deps/DPVO','--dpvo-commit','859bbbfdac6c6185f345003b3c473901fcd13ace',
        '--dpvo-compat-patch','/bundle/deps/dpvo_compat.patch','--network','/bundle/deps/weights/dpvo.pth',
        '--dpvo-config','/bundle/deps/DPVO/config/fast.yaml','--seed','1234',
        '--output-dir',root/'dpvo'/seq])))
    tasks.append(('droid',entry('training_free_droid_expert',[
        '--trajectory',visual,'--image-dir',image/'image_2','--calibration',image/'calib.txt',
        '--tracking-times',image/'times.txt','--tracking-stride','1',
        '--droid-root',THIRD/'DROID-SLAM','--droid-commit','2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99',
        '--lietorch-commit','7f687644fcea81ab337831749445224503c1290d',
        '--pytorch-scatter-commit','6cf77c420f837a427b0d57965e414d68c1bd89ec',
        '--droid-diff-sha256','75a11da44c802486bc6f65640aa48a730f0f684c5c07a42ba3cd1735eb3fb070',
        '--weights','/bundle/deps/weights/droid.pth','--weights-mirror','/bundle/deps/weights/droid.mirror.pth',
        '--policy',config('training_free_droid_fixed.json'),'--seed','1234',
        '--output-dir',root/'droid'/seq])))
    tasks.append(('base',entry('training_free_consensus_moe',[
        '--base-router-root',root/'router','--dense-dpvo-root',root/'dpvo','--droid-root',root/'droid',
        '--policy',config('training_free_consensus_moe_fixed.json'),'--output-dir',root/'base'])))
    return tasks
