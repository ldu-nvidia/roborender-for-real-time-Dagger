import numpy as np, glob, os, cv2
A="/svl/u/ravenh/fast_roborender/videogen_dataset"; B="/svl/u/ravenh/fast_roborender/ab_test_dataset"
rows=[]
for bf in sorted(glob.glob(f"{B}/chunk_*.npz"))[:60]:
    name=os.path.basename(bf); af=os.path.join(A,name)
    if not os.path.exists(af): continue
    zb=np.load(bf)
    if "rgb_gen" not in zb.files or "depth_raw__head_color" not in zb.files: continue
    raw=zb["depth_raw__head_color"].astype(np.float32)
    dmin=np.where(raw>0.01,raw,np.inf).min(axis=(1,2))
    t=int(dmin.argmin())
    if dmin[t]>0.20: continue
    za=np.load(af)
    if "rgb_gen" not in za.files: continue
    H,W=216,384
    a=cv2.resize(za["rgb_gen"][t],(W,H),interpolation=cv2.INTER_NEAREST)
    b=cv2.resize(zb["rgb_gen"][t],(W,H),interpolation=cv2.INTER_NEAREST)
    db=cv2.resize(zb["depth_ctrl__head_color"][t],(W,H))
    row=np.concatenate([np.repeat(db[...,None],3,-1),a,b],axis=1)
    cv2.putText(row,f"{name[-11:-4]} t{t} d={dmin[t]:.2f}m  [B-depth | A-gen | B-gen]",(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
    rows.append(row)
    if len(rows)>=4: break
import imageio.v3 as iio
iio.imwrite("/tmp/near_ab.png", np.concatenate(rows,axis=0))
print("saved", len(rows), "near-object comparisons")
