"""Shared-geometry admission including drift in the local monocular scale.

The global test alone admits nonlinear progress on an almost straight path.
With shared frame indices, one global gauge requires the relative traveled-
distance ratio to stay reasonably constant. Compare overlapping 16-increment
windows, excluding degenerate windows. The central 80% of the ratios may span
at most a factor of two. This is a development safeguard, not a GT error model.
"""
import numpy as np
from geometry_admission import geometry_agreement as global_agreement
from geometry_admission import apply_geometry_admission
POLICY={'minimum_geometry_r2':0.75,'minimum_spatial_variance':1e-12,'distance_window_increments':16,'minimum_windows':5,'maximum_local_scale_spread':2.0}

def geometry_agreement(base,candidate,policy=POLICY):
    out=global_agreement(base,candidate,policy)
    if out.get('r2')is None:return out
    base=np.asarray(base);candidate=np.asarray(candidate)
    db=np.linalg.norm(np.diff(base[:,:3,3],axis=0),axis=1)
    dc=np.linalg.norm(np.diff(candidate[:,:3,3],axis=0),axis=1)
    window=min(policy['distance_window_increments'],len(db))
    sb=np.convolve(db,np.ones(window),'valid');sc=np.convolve(dc,np.ones(window),'valid')
    valid=(sb>max(float(np.median(sb))*1e-3,1e-12))&(sc>max(float(np.median(sc))*1e-3,1e-12))
    ratios=sc[valid]/sb[valid]
    if len(ratios)<policy['minimum_windows']:
        out.update(valid=False,reason='insufficient_motion_scale_support',scale_windows=int(len(ratios)));return out
    q10,q50,q90=np.quantile(ratios,[.1,.5,.9]);spread=float(q90/q10)
    out.update(local_scale_spread=spread,local_scale_q10=float(q10),local_scale_median=float(q50),local_scale_q90=float(q90),scale_windows=int(len(ratios)),maximum_local_scale_spread=policy['maximum_local_scale_spread'])
    if not np.isfinite(spread)or spread>policy['maximum_local_scale_spread']:
        out.update(valid=False,reason='nonuniform_relative_motion_scale')
    return out
