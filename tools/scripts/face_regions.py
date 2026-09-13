"""Anime face landmark → 캐릭터 눈/입 좌표 추출 → meta.txt 저장"""
import sys, cv2, numpy as np
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("usage: face_regions.py <source.png> <out_meta_dir>")
        return 1
    img_path = sys.argv[1]
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        print(f"[face] ERROR: cannot read {img_path}")
        return 1
    H, W = img.shape[:2]
    print(f"[face] image: {W}x{H}")

    from anime_face_detector import create_detector
    det = create_detector('yolov3', device='mps')
    preds = det(img)
    print(f"[face] faces detected: {len(preds)}")
    if not preds:
        print("[face] WARN: no face, using defaults")
        return 0

    p = preds[0]
    kps = p['keypoints']
    bbox = p['bbox']
    print(f"[face] bbox score: {bbox[4]:.2f}")

    # --- 그룹 (hysts 28pt 공식) ---
    # 11-16: 한쪽 눈 (6점, 2행 3열)
    # 17-22: 반대쪽 눈 (6점)
    # 23: 코, 24: 입 중심, 25-26: 입꼬리, 27: 턱
    # 눈썹은 이 모델에 없음
    eyeL_pts = kps[11:17]   # 6점
    eyeR_pts = kps[17:23]   # 6점
    nose = kps[23]
    mouth_c = kps[24]
    mouthL = kps[25]
    mouthR = kps[26]
    chin = kps[27]

    eyeL_c = eyeL_pts[:, :2].mean(axis=0)
    eyeR_c = eyeR_pts[:, :2].mean(axis=0)
    mouth_c_xy = mouth_c[:2]
    # 입꼬리 두 점 평균 = 입 중심 보정
    corner_mid = (mouthL[:2] + mouthR[:2]) * 0.5
    # 입 중심은 mouth_c, 입꼬리는 좌우
    mouthL_xy = mouthL[:2]
    mouthR_xy = mouthR[:2]

    # --- 눈 크기 기반 radius (6점 bbox 기준) ---
    eyeL_xs = eyeL_pts[:, 0]; eyeL_ys = eyeL_pts[:, 1]
    eyeR_xs = eyeR_pts[:, 0]; eyeR_ys = eyeR_pts[:, 1]
    eyeL_w = eyeL_xs.max() - eyeL_xs.min()
    eyeL_h = eyeL_ys.max() - eyeL_ys.min()
    eyeR_w = eyeR_xs.max() - eyeR_xs.min()
    eyeR_h = eyeR_ys.max() - eyeR_ys.min()
    # 눈 반폭 기준 (shader falloff 크기)
    eye_radius = ((eyeL_w + eyeR_w) * 0.5) * 0.6  # 반폭 → 절반이 아니라 눈살 반경 정도

    # --- 입 크기 기반 radius ---
    mouth_w = np.linalg.norm(mouthL_xy - mouthR_xy)
    mouth_radius = mouth_w * 0.7

    # --- 정규화 ---
    def norm(pt):
        return float(pt[0]) / W, float(pt[1]) / H

    eyeL_n = norm(eyeL_c)
    eyeR_n = norm(eyeR_c)
    mouth_n = norm(mouth_c_xy)
    mouthL_n = norm(mouthL_xy)
    mouthR_n = norm(mouthR_xy)
    nose_n = norm(nose[:2])
    chin_n = norm(chin[:2])

    eye_r_n = float(eye_radius) / W
    mouth_r_n = float(mouth_radius) / W

    # --- 저장 ---
    lines = [
        f"region_eye_l_x {eyeL_n[0]:.5f}",
        f"region_eye_l_y {eyeL_n[1]:.5f}",
        f"region_eye_r_x {eyeR_n[0]:.5f}",
        f"region_eye_r_y {eyeR_n[1]:.5f}",
        f"region_mouth_x {mouth_n[0]:.5f}",
        f"region_mouth_y {mouth_n[1]:.5f}",
        f"region_mouth_l_x {mouthL_n[0]:.5f}",
        f"region_mouth_l_y {mouthL_n[1]:.5f}",
        f"region_mouth_r_x {mouthR_n[0]:.5f}",
        f"region_mouth_r_y {mouthR_n[1]:.5f}",
        f"region_nose_x {nose_n[0]:.5f}",
        f"region_nose_y {nose_n[1]:.5f}",
        f"region_chin_x {chin_n[0]:.5f}",
        f"region_chin_y {chin_n[1]:.5f}",
        f"region_eye_radius {eye_r_n:.5f}",
        f"region_mouth_radius {mouth_r_n:.5f}",
    ]

    meta_path = out_dir / "meta.txt"
    # 기존 meta.txt 있으면 append, 없으면 새로
    existing = ""
    if meta_path.exists():
        existing = meta_path.read_text()
    # 기존에 region_* 있으면 제거
    kept = [l for l in existing.splitlines() if not l.startswith("region_")]
    kept.extend(lines)
    meta_path.write_text("\n".join(kept) + "\n")

    print(f"[face] wrote {len(lines)} region lines to {meta_path}")
    for l in lines:
        print(f"  {l}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
