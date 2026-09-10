"""Live drone detection from a UVC camera.

The venv's cv2 is the headless build (albumentations pulls opencv-python-headless),
so there is no imshow. View the stream at http://localhost:8000/ instead, or use
--save to write an annotated mp4.
"""
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--weights", default="runs/train/v8n_p3_relu62/weights/best.pt")
p.add_argument("--device", type=int, default=0)
p.add_argument("--size", default="640x480", help="capture resolution WxH")
p.add_argument("--fps", type=int, default=120, help="frame rate to request from the camera")
p.add_argument("--preview-fps", type=float, default=30.0, help="annotate+encode at most this often")
p.add_argument("--imgsz", type=int, default=320)
p.add_argument("--conf", type=float, default=0.25)
p.add_argument("--port", type=int, default=8000)
p.add_argument("--save", default=None, help="also write an annotated mp4 here")
p.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = forever)")
a = p.parse_args()

latest = {"jpeg": None}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
        self.end_headers()
        while True:
            jpeg = latest["jpeg"]
            if jpeg is not None:
                try:
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
            time.sleep(0.02)

    def log_message(self, *args):
        pass


w, h = (int(x) for x in a.size.split("x"))
cap = cv2.VideoCapture(a.device, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS, a.fps)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # 1 starves the V4L2 queue and halves the frame rate
if not cap.isOpened():
    raise SystemExit(f"cannot open /dev/video{a.device}")

model = YOLO(a.weights)
writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), float(a.fps), (w, h)) if a.save else None

server = HTTPServer(("127.0.0.1", a.port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"streaming at http://localhost:{a.port}/  (ctrl-c to stop)", flush=True)

n, t_prev, t_preview, fps = 0, time.time(), 0.0, 0.0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]

        now = time.time()
        inst = 1.0 / max(now - t_prev, 1e-6)
        fps = inst if n == 0 else 0.9 * fps + 0.1 * inst
        t_prev = now
        if now - t_preview >= 1.0 / a.preview_fps or writer:
            t_preview = now
            out = r.plot()
            cv2.putText(out, f"{fps:5.1f} FPS  {len(r.boxes)} det", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            latest["jpeg"] = cv2.imencode(".jpg", out)[1].tobytes()
            if writer:
                writer.write(out)

        n += 1
        if a.frames and n >= a.frames:
            break
except KeyboardInterrupt:
    pass

cap.release()
if writer:
    writer.release()
print(f"{n} frames, {fps:.1f} FPS")
