"""Apply scale consistency only where the base retains native motion magnitudes.

Direction-transfer heads retain the visual model's per-step lengths, so their
local distance ratios are not a valid native-scale reference. The existing
consensus router explicitly identifies its native DPVO head; do not infer this
property from sequence identity or GT accuracy.
"""
from geometry_admission_local import geometry_agreement,POLICY
from geometry_admission import apply_geometry_admission

def reference_is_native(metadata):
    gate=metadata.get('gate',{})
    return bool(gate.get('route_enabled')and gate.get('selected_head')=='dense_native_open_road'and gate.get('selected_expert')=='dpvo')

def admission_diagnostic(base,candidate,metadata):
    native=reference_is_native(metadata)
    if candidate is None:
        return {'valid':False,'applicable':native,'reason':'orb_unavailable','r2':None}
    if not native:
        return {'valid':True,'applicable':False,'reason':'not_applied_without_native_motion_reference','r2':None,'reference_head':metadata.get('gate',{}).get('selected_head')}
    result=geometry_agreement(base,candidate)
    result.update(applicable=True,reference_head=metadata['gate']['selected_head'])
    return result
