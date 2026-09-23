"""The whole chain on the board: camera -> accelerator -> decode -> tracker -> aim.

    python3 live.py                 # run until Ctrl-C
    python3 live.py -n 600          # 600 frames, then print the latency table

Prints one status line a second (the console is a 115200-baud UART, so no
per-frame output) and, at the end, per-stage latency. `age` is measured from the
driver's buffer timestamp, so it starts after exposure and part of readout —
a lower bound on glass-to-aim latency until there is a hardware trigger.

Aim offsets are in network-input pixels from the window centre, which is 4-5 px
off the sensor centre in the default 720p mode (capture.py). Every threshold in
track.py is still a placeholder: this proves the chain, not the aim.
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
    args = p.parse_args()

    os.environ.setdefault("XILINX_XRT", "/usr")          # see run_on_board.py
    os.makedirs("/lib/firmware", exist_ok=True)
    from pynq.pl_server.device import Device
    from driver_base import FINNExampleOverlay
    from capture import Camera, OUT_H, OUT_W
    from postprocess import decode_confident
    from run_on_board import io_shape_dict
    from track import AimTracker

    dq = np.load(os.path.join(HERE, "v8n_p3_w4a4_192x320_dequant.npz"))
    scale, bias = dq["scale"], dq["bias"]
    acc = FINNExampleOverlay(bitfile_name=os.path.join(HERE, "resizer.bit"),
                             platform="zynq-iodma", io_shape_dict=io_shape_dict,
                             batch_size=1, runtime_weight_dir=os.path.join(HERE, "runtime_weights/"),
                             device=Device.devices[0])
    w, h = map(int, args.mode.split("x"))
    cam = Camera(args.dev, w, h, args.fps)
    trk = AimTracker((OUT_W, OUT_H))
    now = lambda: time.clock_gettime(time.CLOCK_MONOTONIC)

    names = ["wait frame", "accelerator", "decode", "track", "age at aim"]
    T = {k: [] for k in names}
    t_prev, t_print, k = None, now(), 0
    print(f"camera {args.mode} @{cam.fps:.0f}, window {OUT_W}x{OUT_H}; Ctrl-C to stop", flush=True)
    try:
        while not args.n or k < args.n:
            t0 = now()
            x, t_frame = cam.read()
            t1 = now()
            y = acc.execute(x)
            t2 = now()
            boxes, scores = decode_confident(y, scale, bias, args.conf)
            t3 = now()
            aim = trk.update(boxes, scores, 0.0 if t_prev is None else t_frame - t_prev)
            t4 = now()
            t_prev = t_frame
            for n, v in zip(names, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t_frame)):
                T[n].append(v)
            k += 1
            if t4 - t_print >= 1.0:
                a = np.array(T["age at aim"][-200:]) * 1e3
                state = (f"aim dx {aim.offset[0]:+6.1f} dy {aim.offset[1]:+6.1f} px"
                         f"{'  CENTRED' if aim.close_enough else ''}" if aim.offset is not None
                         else "tracking, no aim yet" if aim.tracking else "no track")
                print(f"{k:6d}  {len(scores):2d} boxes  {state:34s}  age {np.median(a):5.1f} ms"
                      f"  skipped {cam.skipped}", flush=True)
                t_print = t4
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()

    print(f"\n{k} frames, {cam.skipped} skipped by the camera thread")
    print(f"{'stage':12s} {'median':>8s} {'p95':>8s} {'max':>8s}   ms")
    for n in names:
        a = 1e3 * np.array(T[n][30:] or T[n])            # drop start-up
        print(f"{n:12s} {np.median(a):8.2f} {np.percentile(a, 95):8.2f} {a.max():8.2f}")


if __name__ == "__main__":
    main()
