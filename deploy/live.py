"""The whole chain on the board: camera -> accelerator -> decode -> tracker -> aim.

    python3 live.py                 # run until Ctrl-C
    python3 live.py -n 600          # 600 frames, then print the latency table
    python3 live.py --depth 1       # one frame at a time, as before 2026-09-26

Prints one status line a second (the console is a 115200-baud UART, so no
per-frame output) and, at the end, per-stage latency. `age` is measured from the
driver's buffer timestamp, so it starts after exposure and part of readout —
a lower bound on glass-to-aim latency until there is a hardware trigger.

Aim offsets are in network-input pixels from the window centre, which is 4-5 px
off the sensor centre in the default 720p mode (capture.py). Every threshold in
track.py is still a placeholder: this proves the chain, not the aim.

Up to `--depth` frames are in the accelerator at once (pipeline.py), each
result handed to the tracker as soon as its frame leaves. 3 is the sweet spot
for ~12 ms of CPU per frame against ~22.5 ms of PL latency: in a host model of
the DMAs it cut the loop from 34.5 to 12.0 ms for +1.6 ms of age; 4 adds only
age (+12 ms), because the extra frame waits for the CPU.
"""
import argparse
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dev", default="/dev/video0")
    p.add_argument("--mode", default="1280x720")
    p.add_argument("--fps", type=int, default=200)
    p.add_argument("-n", type=int, default=0, help="frames to run (0 = until Ctrl-C)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--depth", type=int, default=3, help="frames in the accelerator at once")
    args = p.parse_args()

    os.environ.setdefault("XILINX_XRT", "/usr")          # see run_on_board.py
    os.makedirs("/lib/firmware", exist_ok=True)
    from pynq.pl_server.device import Device
    from driver_base import FINNExampleOverlay
    from capture import Camera, OUT_H, OUT_W
    from pipeline import AccelPipeline
    from postprocess import decode_confident
    from run_on_board import io_shape_dict
    from track import AimTracker

    dq = np.load(os.path.join(HERE, "v8n_p3_w4a4_192x320_dequant.npz"))
    scale, bias = dq["scale"], dq["bias"]
    acc = FINNExampleOverlay(bitfile_name=os.path.join(HERE, "resizer.bit"),
                             platform="zynq-iodma", io_shape_dict=io_shape_dict,
                             batch_size=1, runtime_weight_dir=os.path.join(HERE, "runtime_weights/"),
                             device=Device.devices[0])
    pipe = AccelPipeline(acc, depth=args.depth)
    w, h = map(int, args.mode.split("x"))
    cam = Camera(args.dev, w, h, args.fps)
    trk = AimTracker((OUT_W, OUT_H))
    now = lambda: time.clock_gettime(time.CLOCK_MONOTONIC)

    names = ["wait frame", "submit", "in PL", "unpack", "decode", "track", "age at aim"]
    T = {k: [] for k in names}
    t_prev, t_print, k = None, now(), 0
    print(f"camera {args.mode} @{cam.fps:.0f}, window {OUT_W}x{OUT_H}, depth {args.depth}; "
          "Ctrl-C to stop", flush=True)
    try:
        t_start = now()
        while not args.n or k < args.n:
            # keep the accelerator fed: the newest frame into every slot it can take
            while pipe.can_submit():
                t0 = now()
                x, t_frame = cam.read()
                t1 = now()
                pipe.submit(x, (t_frame, t1))
                T["wait frame"].append(t1 - t0)
                T["submit"].append(now() - t1)
            t2 = now()
            r = pipe.poll()
            if r is None:
                time.sleep(0.0002)                        # let the capture thread run
                continue
            t3 = now()
            (t_frame, t_sub), y = r
            boxes, scores = decode_confident(y, scale, bias, args.conf)
            t4 = now()
            aim = trk.update(boxes, scores, 0.0 if t_prev is None else t_frame - t_prev)
            t5 = now()
            t_prev = t_frame
            for n, v in zip(names[2:], (t2 - t_sub, t3 - t2, t4 - t3, t5 - t4, t5 - t_frame)):
                T[n].append(v)
            k += 1
            if t5 - t_print >= 1.0:
                a = np.array(T["age at aim"][-200:]) * 1e3
                state = (f"aim dx {aim.offset[0]:+6.1f} dy {aim.offset[1]:+6.1f} px"
                         f"{'  CENTRED' if aim.close_enough else ''}" if aim.offset is not None
                         else "tracking, no aim yet" if aim.tracking else "no track")
                print(f"{k:6d}  {len(scores):2d} boxes  {state:34s}  age {np.median(a):5.1f} ms"
                      f"  skipped {cam.skipped}", flush=True)
                t_print = t5
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()

    fps = k / (now() - t_start) if k else 0.0
    print(f"\n{k} frames at {fps:.1f} FPS, {cam.skipped} skipped by the camera thread")
    print(f"{'stage':12s} {'median':>8s} {'p95':>8s} {'max':>8s}   ms")
    for n in names:
        a = 1e3 * np.array(T[n][30:] or T[n])            # drop start-up
        print(f"{n:12s} {np.median(a):8.2f} {np.percentile(a, 95):8.2f} {a.max():8.2f}")


if __name__ == "__main__":
    main()
