#!/usr/bin/env python3
# 2.5D 루프의 측정기: 실루엣 + 내부 변형 + face/body 비율.
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
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 2: d[parts[0]] = parts[1]
    return d

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)

def main(proj, yaws, pitches, scale):
    proj = Path(proj)
    meta = parse_meta(proj / 'meta.txt')
    W = int(float(meta.get('W', meta.get('width', 7424))))
    H = int(float(meta.get('H', meta.get('height', 10560))))
    print('[image] %dx%d' % (W, H))

    vxy  = load_f32(proj / 'vertices_xy.bin', 2)
    vz   = load_f32(proj / 'vertices_z.bin')
    bx   = load_f32(proj / 'basis_x.bin')
    by   = load_f32(proj / 'basis_y.bin')
    tris = load_u32(proj / 'triangles.bin').reshape(-1, 3)
    N = vxy.shape[0]
    print('[mesh] v=%d z_hist(10bin)=%s' % (N, np.histogram(vz, 10, (0,1))[0].tolist()))

    rX = float(meta.get('R_X', 100))
    rY = float(meta.get('R_Y',  60))
    print('[rig] rX=%.2f rY=%.2f' % (rX, rY))

    # face/body 마스크 (z 기준)
    face_mask = vz > 0.65
    body_mask = vz < 0.35
    print('[mask] face=%d body=%d (%.1f%% / %.1f%%)' % (
        face_mask.sum(), body_mask.sum(),
        100*face_mask.mean(), 100*body_mask.mean()))

    # 커버리지용 grid
    spacing = 4.12
    cover_r = spacing * 1.5
    oW, oH = W // scale, H // scale
    yy, xx = np.mgrid[0:oH, 0:oW]
    px = np.stack([xx.ravel()*scale + scale//2,
                   yy.ravel()*scale + scale//2], 1).astype(np.float32)

    def deform(mode, yaw_deg, pitch_deg):
        yaw = np.deg2rad(yaw_deg); pitch = np.deg2rad(pitch_deg)
        sx = np.sin(yaw) * rX
        sy = np.sin(pitch) * rY
        dx = bx * sx
        dy = by * sy
        if mode == 'A':           # 현재: 모든 vertex 동일
            w = np.ones_like(vz)
        elif mode == 'B':         # z 비례 (0.3 ~ 1.0)
            w = 0.3 + 0.7 * vz
        elif mode == 'C':         # smoothstep
            w = 0.15 + 0.85 * smoothstep(0.30, 0.85, vz)
        p = vxy.copy()
        p[:,0] += dx * w
        p[:,1] += dy * w
        return p, dx * w, dy * w

    def coverage(p):
        t = cKDTree(p)
        d, _ = t.query(px, k=1, workers=-1)
        return (d < cover_r).reshape(oH, oW)

    cov0 = coverage(vxy)
    ref = int(cov0.sum())
    print('[ref] covered@(0,0)=%d px' % ref)

    results = {}
    for mode in ('A', 'B', 'C'):
        print('')
        print('=== MODE %s ===' % mode)
        print('  yaw pitch | ratio | hole  expand | face_px body_px f/b')
        print('  ' + '-'*62)
        rows = []
        for yaw in yaws:
            for pitch in pitches:
                p, dx, dy = deform(mode, yaw, pitch)
                disp = np.sqrt(dx*dx + dy*dy)
                fm = float(disp[face_mask].mean()) if face_mask.any() else 0
                bm = float(disp[body_mask].mean()) if body_mask.any() else 0
                fb = fm / max(bm, 1e-6)
                cov = coverage(p)
                hole = int((cov0 & ~cov).sum())
                exp  = int((~cov0 & cov).sum())
                ratio = float(cov.sum()) / max(ref, 1)
                rows.append({'yaw':yaw,'pitch':pitch,'ratio':ratio,
                             'hole':hole,'expand':exp,
                             'face_px':fm,'body_px':bm,'fb_ratio':fb})
                print('  %+4d %+5d | %.4f | %5d %6d | %6.2f %6.2f %5.2f' % (
                    yaw, pitch, ratio, hole, exp, fm, bm, fb))
        results[mode] = rows

    # face/body 비율 요약 (yaw=+30, pitch=0 기준)
    print('')
    print('=== 요약: (yaw=+30, pitch=0) face/body 비율 ===')
    for mode in ('A', 'B', 'C'):
        for r in results[mode]:
            if r['yaw'] == 30 and r['pitch'] == 0:
                print('  mode %s: %.2f  (face=%.2f px, body=%.2f px, hole=%d)' % (
                    mode, r['fb_ratio'], r['face_px'], r['body_px'], r['hole']))

    out = {'image':[W,H], 'scale':scale, 'face_pct':float(face_mask.mean()),
           'body_pct':float(body_mask.mean()), 'ref':ref, 'results':results}
    (proj / 'bench_v2.json').write_text(json.dumps(out, indent=2, default=float))
    print('')
    print('[done] %s/bench_v2.json' % proj)

if __name__ == '__main__':
    proj = sys.argv[1] if len(sys.argv) > 1 else 'projects/Image_v2'
    main(proj, [-30,-15,0,15,30], [-15,0,15], 8)
