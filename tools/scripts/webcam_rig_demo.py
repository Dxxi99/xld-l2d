#!/usr/bin/env python3
"""webcam_rig_demo.py — 실시간 webcam -> yaw/pitch -> mesh 렌더"""
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from PIL import Image
import time

# ---------- load rig ----------
P = Path.home() / "XLD-L2D/assets/real/mesh/rig_v12_flipfree.npz"
d = np.load(P)
xs = d['xs'].astype(np.float32)
ys = d['ys'].astype(np.float32)
tris = d['triangles'].astype(np.int32)
bx = d['basis_x'].astype(np.float32)
by = d['basis_y'].astype(np.float32)
R_X = float(d['R_X']); R_Y = float(d['R_Y'])
H, W = int(d['H']), int(d['W'])
n_v, n_t = len(xs), len(tris)
print(f"[load] n_v={n_v} n_t={n_t} H={H} W={W}")

# ---------- precompute source colors per triangle ----------
SRC = Path.home() / "XLD-L2D/assets/real/my_vtuber.png"
src_img = Image.open(SRC).convert("RGB")
if src_img.size != (W, H):
    src_img = src_img.resize((W, H), Image.LANCZOS)
src_np = np.array(src_img)

tri_cx = ((xs[tris[:,0]] + xs[tris[:,1]] + xs[tris[:,2]]) / 3).astype(np.int32)
tri_cy = ((ys[tris[:,0]] + ys[tris[:,1]] + ys[tris[:,2]]) / 3).astype(np.int32)
tri_cx = np.clip(tri_cx, 0, W-1); tri_cy = np.clip(tri_cy, 0, H-1)
tri_col = src_np[tri_cy, tri_cx]  # (n_t, 3) uint8

# ---------- face mesh ----------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_head_pose(landmarks, W_img, H_img):
    """yaw, pitch (deg) from face landmarks — 간단 3점 기반"""
    # 코끝(1), 왼눈(33), 오른눈(263), 턱(152), 이마(10)
    def pt(i): return np.array([landmarks[i].x * W_img, landmarks[i].y * H_img])
    nose = pt(1); le = pt(33); re = pt(263)
    chin = pt(152); top = pt(10)
    # yaw: 좌우 눈 대비 코 위치
    eye_mid = (le + re) / 2
    eye_w = np.linalg.norm(re - le) + 1e-6
    yaw = np.degrees(np.arctan2(nose[0] - eye_mid[0], eye_w))
    # pitch: 눈-코-턱 세로 비율
    eye_y = eye_mid[1]
    total = chin[1] - top[1] + 1e-6
    ratio = (nose[1] - eye_y) / total
    pitch = (ratio - 0.35) * 100  # calibration
    return yaw * 2.0, pitch * 0.5  # gain

# ---------- render ----------
def render(yaw, pitch, base_img):
    sx = np.sin(np.deg2rad(yaw)) * R_X
    sy = np.sin(np.deg2rad(pitch)) * R_Y
    x2 = xs + bx * sx
    y2 = ys + by * sy
    # rasterize triangles (cv2.fillConvexPoly, float->int)
    canvas = np.zeros_like(base_img)
    # 작은 삼각형 수가 많아 Python loop은 느림 → numpy로 bbox loop 최적화
    tri_x = x2[tris]  # (n_t, 3)
    tri_y = y2[tris]
    # bbox
    x_min = np.floor(tri_x.min(1)).astype(np.int32)
    x_max = np.ceil(tri_x.max(1)).astype(np.int32)
    y_min = np.floor(tri_y.min(1)).astype(np.int32)
    y_max = np.ceil(tri_y.max(1)).astype(np.int32)
    # 간단히: z-order 없이 뒤→앞 (tri index 순서 그대로)
    for t in range(n_t):
        pts = np.stack([tri_x[t], tri_y[t]], 1).astype(np.int32)
        cv2.fillConvexPoly(canvas, pts, tri_col[t].tolist(), lineType=cv2.LINE_AA)
    return canvas

# ---------- main loop ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("[webcam] press 'q' to quit")
last_t = time.time(); fps = 0
alpha = 0.3
smooth_yaw, smooth_pitch = 0.0, 0.0
while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)  # mirror
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    yaw_t, pitch_t = 0.0, 0.0
    if res.multi_face_landmarks:
        yaw_t, pitch_t = get_head_pose(res.multi_face_landmarks[0].landmark, w, h)
        yaw_t = float(np.clip(yaw_t, -30, 30))
        pitch_t = float(np.clip(pitch_t, -15, 15))
    # smoothing
    smooth_yaw = (1-alpha)*smooth_yaw + alpha*yaw_t
    smooth_pitch = (1-alpha)*smooth_pitch + alpha*pitch_t
    # render
    out = render(smooth_yaw, smooth_pitch, src_np)
    # resize to fit window
    disp = cv2.resize(out, (W//2, H//2))
    cv2.putText(disp, f"yaw={smooth_yaw:+.1f}  pitch={smooth_pitch:+.1f}  fps={fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    cv2.putText(disp, f"yaw={smooth_yaw:+.1f}  pitch={smooth_pitch:+.1f}  fps={fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
    cv2.imshow("XLD-L2D webcam rig demo", disp)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    now = time.time()
    fps = 0.9*fps + 0.1/(now - last_t + 1e-6)
    last_t = now
cap.release()
cv2.destroyAllWindows()
