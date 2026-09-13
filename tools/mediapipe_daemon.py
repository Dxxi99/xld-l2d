"""MediaPipe Face Landmarker → UDP로 blendshape + transform 전송"""
import sys, time, socket, json, argparse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5005)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--model', default='models/mediapipe/face_landmarker.task')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--include-landmarks', action='store_true', default=False)
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = ('127.0.0.1', args.port)
    last_sent = [0.0]

    def on_result(result, image, timestamp_ms):
        now = time.time()
        if now - last_sent[0] < 1.0 / 60.0:
            return
        last_sent[0] = now
        payload = {'t': now, 'faces': 0}
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            bs = result.face_blendshapes[0]
            payload['blendshapes'] = {b.category_name: round(float(b.score), 4) for b in bs}
            payload['faces'] = 1
        if result.facial_transformation_matrixes:
            m = result.facial_transformation_matrixes[0]
            payload['matrix'] = [round(float(x), 5) for x in np.array(m).flatten()]
        # landmarks 는 옵션 (UDP 크기 때문에 기본 off)
        if args.include_landmarks and result.face_landmarks and len(result.face_landmarks) > 0:
            lms = result.face_landmarks[0]
            payload['landmarks'] = [[round(float(lm.x), 4), round(float(lm.y), 4), round(float(lm.z), 4)] for lm in lms]
        try:
            sock.sendto(json.dumps(payload).encode('utf-8'), addr)
        except Exception as e:
            print(f'[daemon] send error: {e}', file=sys.stderr)

    base_opts = mp_python.BaseOptions(model_asset_path=args.model)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        result_callback=on_result
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        print('[daemon] ERROR: cannot open camera', file=sys.stderr)
        return 1

    print(f'[daemon] started, sending to UDP {addr}', file=sys.stderr)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_img, ts_ms)
            time.sleep(1.0 / args.fps)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        landmarker.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
