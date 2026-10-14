"""GT-free shape agreement check before accepting a global monocular expert.

A single positive-scale Sim(3) must explain at least 75% of the reference
trajectory's spatial variance. This is an explicit conservative engineering
threshold, not an accuracy estimate or a promise that the base is correct.
No sequence identity, loop mode, GT, or archived prediction is used.
"""
import numpy as np
from vistream.metrics import umeyama_alignment

POLICY = {'minimum_geometry_r2': 0.75, 'minimum_spatial_variance': 1e-12}

def geometry_agreement(base, candidate, policy=POLICY):
    base=np.asarray(base,dtype=np.float64); candidate=np.asarray(candidate,dtype=np.float64)
    if base.shape!=candidate.shape or base.ndim!=3 or base.shape[1:]!=(4,4) or len(base)<3:
        return {'valid':False,'reason':'incompatible_pose_support','r2':None}
    if not np.isfinite(base).all() or not np.isfinite(candidate).all():
        return {'valid':False,'reason':'nonfinite_pose','r2':None}
    target=base[:,:3,3]; source=candidate[:,:3,3]
    target_variance=float(np.mean(np.sum((target-target.mean(0))**2,axis=1)))
    source_variance=float(np.mean(np.sum((source-source.mean(0))**2,axis=1)))
    if min(target_variance,source_variance)<=policy['minimum_spatial_variance']:
        return {'valid':False,'reason':'insufficient_spatial_support','r2':None}
    alignment=umeyama_alignment(source,target,with_scale=True)
    residual=alignment.apply(source)-target
    normalized_mse=float(np.mean(np.sum(residual**2,axis=1))/target_variance)
    r2=float(1.0-normalized_mse)
    valid=bool(np.isfinite(r2)and alignment.scale>0 and r2>=policy['minimum_geometry_r2'])
    return {'valid':valid,'reason':'shared_global_geometry'if valid else'global_geometry_conflict','r2':r2,'minimum_r2':policy['minimum_geometry_r2'],'normalized_rmse':float(np.sqrt(max(normalized_mse,0.0))),'candidate_to_base_scale':float(alignment.scale)}

def apply_geometry_admission(gate, diagnostic):
    gate=dict(gate)
    gate['geometry_admission']=diagnostic
    if gate['orb_weight']>0 and not diagnostic['valid']:
        gate.update(valid_orb_head=False,environment='orb_geometry_conflict_base_fallback',orb_weight=0.0,base_weight=1.0,selected_expert='base')
    return gate
