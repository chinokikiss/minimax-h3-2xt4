"""CPU-only encoded-video motion diagnostic; not causal attribution."""
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

root=Path(__file__).resolve().parent
path=root/'kaggle/gpu_final/turbo_085mp_e2e/generation/video.mp4'
out=root/'quality_review';out.mkdir(exist_ok=True)
cv2.setNumThreads(2)
capture=cv2.VideoCapture(str(path));fps=capture.get(cv2.CAP_PROP_FPS)
frames=[]
while True:
    ok,frame=capture.read()
    if not ok:break
    gray=cv2.cvtColor(cv2.resize(frame,(384,222)),cv2.COLOR_BGR2GRAY)
    frames.append(gray)
capture.release();assert len(frames)==240 and fps==24
rows=[];scale=704/222
for i in range(1,len(frames)):
    a,b=frames[i-1],frames[i]
    points=cv2.goodFeaturesToTrack(a,maxCorners=600,qualityLevel=.015,minDistance=6)
    row=dict(frame=i,time_seconds=i/fps,decode_stride_phase=i%17)
    if points is not None:
        following,ok,_=cv2.calcOpticalFlowPyrLK(a,b,points,None,winSize=(21,21),maxLevel=3)
        source=points[ok.ravel()==1].reshape(-1,2);target=following[ok.ravel()==1].reshape(-1,2)
        if len(source)>=12:
            matrix,inliers=cv2.estimateAffinePartial2D(source,target,method=cv2.RANSAC,ransacReprojThreshold=2.)
            if matrix is not None:
                row.update(dy_original_pixels=float(matrix[1,2]*scale),
                    dx_original_pixels=float(matrix[0,2]*1216/384),inliers=int(inliers.sum()),tracked=len(source))
    rows.append(row)
valid=[r for r in rows if r.get('inliers',0)>=20]
for r in valid:r['abs_dy']=abs(r['dy_original_pixels'])
phases=[]
for phase in range(17):
    values=[r['abs_dy'] for r in valid if r['decode_stride_phase']==phase]
    phases.append(dict(phase=phase,count=len(values),median_abs_dy=float(np.median(values)) if values else None))
top=sorted(valid,key=lambda r:r['abs_dy'],reverse=True)[:15]
summary=dict(video_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frames=240,fps=24,
    method='Downscaled grayscale Shi-Tomasi/Lucas-Kanade tracks and RANSAC affine partial fit; dy is an estimated affine translation component, not pure camera displacement.',
    warnings=['Foreground motion, scale/rotation, occlusion and compression confound global motion estimates.',
        'A single video cannot isolate TE, DiT or VAE causes.',
        '17-frame Native chunk cadence is shared by native and Temporal planning, not unique to parallel execution.'],
    valid_transitions=len(valid),phase_medians=phases,largest_vertical_components=top)
(out/'motion_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
with (out/'motion.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=['frame','time_seconds','decode_stride_phase','dy_original_pixels','dx_original_pixels','inliers','tracked','abs_dy'])
    writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(valid_transitions=len(valid),phase_medians=phases,top=top[:8]),indent=2))
