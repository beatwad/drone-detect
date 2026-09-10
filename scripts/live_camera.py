"""Live drone detection from a UVC camera.

Capture runs in its own thread so the MJPEG decode overlaps inference; at full
resolution the two are the whole frame budget (6.1 ms and 4.9 ms) and serialising
them caps the loop at ~81 FPS. The thread keeps only the newest frame: if
inference falls behind, frames are dropped rather than queued, so latency stays
bounded at the cost of throughput.

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
ok, warm = cap.read()          # CUDA init costs ~0.5 s; pay it before the thread starts
if ok:
    model.predict(warm, imgsz=a.imgsz, verbose=False)
writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), float(a.fps), (w, h)) if a.save else None

server = HTTPServer(("127.0.0.1", a.port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"streaming at http://localhost:{a.port}/  (ctrl-c to stop)", flush=True)

newest = {"frame": None, "seq": 0}
new_frame = threading.Condition()
stop = threading.Event()


def capture():
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        with new_frame:
            newest["frame"], newest["seq"] = frame, newest["seq"] + 1
            new_frame.notify()
    with new_frame:            # unblock the consumer on camera failure
        stop.set()
        new_frame.notify()


grabber = threading.Thread(target=capture, daemon=True)
grabber.start()

n, t_prev, t_preview, fps, seq, dropped = 0, time.time(), 0.0, 0.0, 0, 0
try:
    while True:
        with new_frame:
            new_frame.wait_for(lambda: newest["seq"] != seq or stop.is_set())
            if stop.is_set():
                break
            frame, last = newest["frame"], newest["seq"]
        dropped += last - seq - 1
        seq = last
        r = model.predict(frame, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]

        now = time.time()
        inst = 1.0 / max(now - t_prev, 1e-6)
        fps = inst if n == 0 else 0.9 * fps + 0.1 * inst
        t_prev = now
        if now - t_preview >= 1.0 / a.preview_fps or writer:
            t_preview = now
            out = r.plot()
            cv2.putText(out, f"{fps:5.1f} FPS  {len(r.boxes)} det  {dropped} dropped", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            latest["jpeg"] = cv2.imencode(".jpg", out)[1].tobytes()
            if writer:
                writer.write(out)

        n += 1
        if a.frames and n >= a.frames:
            break
except KeyboardInterrupt:
    pass

stop.set()
grabber.join(timeout=2.0)   # never release the device under an in-flight read()
cap.release()
if writer:
    writer.release()
print(f"{n} frames, {fps:.1f} FPS, {dropped} dropped")
