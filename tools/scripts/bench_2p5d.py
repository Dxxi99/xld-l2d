#!/usr/bin/env python3
# 5부(현재) 셰이더와 동일한 변형으로 yaw/pitch 격자 수치 측정. 판단 없음.
import sys, json
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from PIL import Image

def load_f32(p, n=None):
    a = np.fromfile(p, dtype='<f4')
    return a.reshape(-1, n) if n else a

def load_u32(p):
    return np.fromfile(p, dtype='<u4')

def parse_meta(p):
    d = {}
    if Path(p).exists():
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                d[parts[0]] = parts[1]
    return d

def main(proj, yaws, pitches, scale):
    proj = Path(proj)
    meta = parse_meta(proj / 'meta.txt')
    W = int(float(meta.get('width',  meta.get('W', 1856))))
    H = int(float(meta.get('height', meta.get('H', 2640))))
    print('[image] %dx%d' % (W, H))

    vxy = load_f32(proj / 'vertices_xy.bin', 2)
    vz  = load_f32(proj / 'vertices_z.bin')
    bx  = load_f32(proj / 'basis_x.bin')
    by  = load_f32(proj / 'basis_y.bin')
    tris = load_u32(proj / 'triangles.bin').reshape(-1, 3)
    N = vxy.shape[0]
    print('[mesh] v=%d t=%d z_min=%.3f z_max=%.3f' % (N, len(tris), vz.min(), vz.max()))

    rX = float(meta.get('R_X', meta.get('rX', 100)))
    rY = float(meta.get('R_Y', meta.get('rY',  60)))
    print('[rig] rX=%.2f rY=%.2f' % (rX, rY))

    e = np.concatenate([
        np.stack([tris[:,0], tris[:,1]], 1),
        np.stack([tris[:,1], tris[:,2]], 1),
        np.stack([tris[:,2], tris[:,0]], 1)], 0)
    e = np.unique(np.sort(e, axis=1), axis=0)
    zd = np.abs(vz[e[:,0]] - vz[e[:,1]])
    print('[depth_grad] p50=%.4f p90=%.4f p99=%.4f max=%.4f' % (
        np.percentile(zd,50), np.percentile(zd,90),
        np.percentile(zd,99), zd.max()))

    spacing = float(np.median(np.linalg.norm(vxy[e[:,0]] - vxy[e[:,1]], axis=1)))
    cover_r = spacing * 1.5
    print('[spacing] edge=%.2f cover_r=%.2f' % (spacing, cover_r))

    oW, oH = W // scale, H // scale
    yy, xx = np.mgrid[0:oH, 0:oW]
    px = np.stack([xx.ravel()*scale + scale//2,
                   yy.ravel()*scale + scale//2], 1).astype(np.float32)
    print('[grid] %dx%d = %d px (scale=%d)' % (oW, oH, px.shape[0], scale))

    def deform(yaw_deg, pitch_deg):
        sx = np.sin(np.deg2rad(yaw_deg)) * rX
        sy = np.sin(np.deg2rad(pitch_deg)) * rY
        p = vxy.copy()
        p[:,0] += bx * sx
        p[:,1] += by * sy
        return p

    def covered(p):
        t = cKDTree(p)
        d, _ = t.query(px, k=1, workers=-1)
        return (d < cover_r).reshape(oH, oW)

    print('[ref] building base tree...')
    cov0 = covered(vxy)
    ref = int(cov0.sum())
    print('[ref] covered@(0,0)=%d px' % ref)

    viz_dir = proj / 'bench_viz'
    viz_dir.mkdir(exist_ok=True)

    out = {'image':[W,H], 'scale':scale, 'vertices':int(N),
           'median_edge_px':spacing, 'cover_radius_px':cover_r,
           'rig':{'rX':rX,'rY':rY}, 'reference_covered_pixels':ref,
           'angles':[]}

    print('')
    print('  yaw pitch   covered     hole   expand   ratio')
    print('  ' + '-'*52)
    for yaw in yaws:
        for pitch in pitches:
            cov = covered(deform(yaw, pitch))
            cpix = int(cov.sum())
            hole = int((cov0 & ~cov).sum())
            exp  = int((~cov0 & cov).sum())
            ratio = cpix / max(ref, 1)
            out['angles'].append({'yaw_deg':yaw,'pitch_deg':pitch,
                'covered_pixels':cpix,'disocclusion_pixels':hole,
                'expansion_pixels':exp,'coverage_ratio':ratio})
            print('  %+4d %+5d %10d %8d %8d   %.4f' % (yaw, pitch, cpix, hole, exp, ratio))
            if yaw in (yaws[0], 0, yaws[-1]) and pitch in (pitches[0], 0, pitches[-1]):
                viz = np.zeros((oH, oW, 3), np.uint8)
                viz[cov0 &  cov] = [255,255,255]
                viz[cov0 & ~cov] = [255,0,0]
                viz[~cov0 &  cov] = [0,128,255]
                Image.fromarray(viz).save(viz_dir / ('y%+03d_p%+03d.png' % (yaw, pitch)))

    (proj / 'bench.json').write_text(json.dumps(out, indent=2))
    print('')
    print('[done] %s/bench.json + bench_viz/*.png' % proj)

if __name__ == '__main__':
    proj = sys.argv[1] if len(sys.argv) > 1 else 'projects/Image_v2'
    main(proj, [-30,-15,0,15,30], [-15,0,15], 8)
