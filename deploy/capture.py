"""Camera -> the accelerator's input: a centre 320x192 RGB window. NumPy only.

The board has no OpenCV, so this talks V4L2 directly: ioctl + mmap, YUYV. YUYV
is uncompressed, so the window is cut out of the raw buffer *before* any
per-pixel work and only 320x192 pixels are ever converted — constant cost, no
JPEG decode whose time depends on the scene (build_notes §12.6).

WHICH MODE, AND WHY 1280x720
  Measured 2026-09-23 on SuperSpeed (build_notes §12.7): 1280x720 and 512x512 are
  both native crops of the 1920x1200 sensor at their full rates, so the centre
  window is the same pixels either way. 720p "@200" really delivers ~249 fps
  (4.01 ms, no duplicated frames), and its frames are fresher when read: 6.7 ms
  vs 11.6 ms for 512x512 @120. The camera evidently sends a frame at the pace
  of sensor readout, so the frame period, not the 3.5x larger transfer, sets
  the age. 720p also leaves room for a window that moves. Its one cost: the
  window sits 4-5 px off the sensor centre, (316, 236) not (320, 240).

ONLY THE NEWEST FRAME
  A thread dequeues continuously, copies the raw window out and requeues the
  buffer at once. `read()` returns the newest frame not yet seen and counts the
  ones it skipped. A queue would feed the detector ever older images while
  reporting a healthy rate; for aiming a stale frame is worse than none.

TIMESTAMPS
  `read()` returns the driver's buffer timestamp, CLOCK_MONOTONIC. uvcvideo
  stamps it when the frame's first USB packet arrives, i.e. after exposure and
  part of readout. So `time.clock_gettime(CLOCK_MONOTONIC) - t` is a lower bound
  on the frame's age, not the true glass-to-aim latency — that needs the
  hardware trigger (issues §4).

Keep callers on `read()`: deployment may replace USB with MIPI into the PL
(CLAUDE.md), and nothing downstream should know the frames came over USB.
"""
import ctypes
import fcntl
import mmap
import os
import select
import struct
import threading
import time

import numpy as np

OUT_H, OUT_W = 192, 320      # the bitstream's input


def _iowr(nr, size, d=3):
    return (d << 30) | (size << 16) | (ord("V") << 8) | nr


# 64-bit layouts (aarch64 and x86-64 agree)
VIDIOC_S_FMT = _iowr(5, 208)
VIDIOC_REQBUFS = _iowr(8, 20)
VIDIOC_QUERYBUF = _iowr(9, 88)
VIDIOC_QBUF = _iowr(15, 88)
VIDIOC_DQBUF = _iowr(17, 88)
VIDIOC_STREAMON = _iowr(18, 4, d=1)
VIDIOC_STREAMOFF = _iowr(19, 4, d=1)
VIDIOC_S_PARM = _iowr(22, 204)
BUF_TYPE_VIDEO_CAPTURE, MEMORY_MMAP = 1, 1
YUYV = struct.unpack("<I", b"YUYV")[0]

# ITU-R BT.601 limited range, fixed point, as OpenCV's COLOR_YUV2RGB_YUYV does it
_SHIFT, _CY, _CUB, _CUG, _CVG, _CVR = 20, 1220542, 2116026, -409993, -852492, 1673527


# Every term depends on one byte, so each is a 256-entry table; the final
# shift + clip is one more table over the whole reachable range.
_b = np.arange(256, dtype=np.int64)
_TY = (np.maximum(_b - 16, 0) * _CY + (1 << (_SHIFT - 1))).astype(np.int32)
_TRV, _TGV = (_CVR * (_b - 128)).astype(np.int32), (_CVG * (_b - 128)).astype(np.int32)
_TGU, _TBU = (_CUG * (_b - 128)).astype(np.int32), (_CUB * (_b - 128)).astype(np.int32)
_LO = -(1 << 10)                                  # (value >> SHIFT) lies in [-260, 535]
_TCLIP = np.clip(np.arange(_LO, -_LO), 0, 255).astype(np.uint8)


def yuyv_to_rgb(raw):
    """(H, 2W) YUYV bytes -> (H, W, 3) RGB uint8, bit-exact with OpenCV.

    The NumPy reference, and the fallback where libyuyv.so does not load (any
    host). Table lookups, chroma applied per pixel pair: still 13.1 ms for a
    320x192 window on the A53, against 14.7 ms for the plain version.
    """
    h, w2 = raw.shape
    q = raw.reshape(h, w2 // 4, 4)                # Y0 U Y1 V
    y = _TY[q[..., 0::2]]                         # (h, w/2, 2)
    u, v = q[..., 1], q[..., 3]
    out = np.empty((h, w2 // 4, 2, 3), np.uint8)
    for c, uv in enumerate((_TRV[v], _TGU[u] + _TGV[v], _TBU[u])):
        s = y + uv[..., None]
        s >>= _SHIFT
        s -= _LO
        out[..., c] = _TCLIP[s]
    return out.reshape(h, w2 // 2, 3)


def _load(path):
    lib = ctypes.CDLL(path)
    p8 = ctypes.POINTER(ctypes.c_uint8)
    lib.yuyv_window_to_rgb.argtypes = [p8, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, p8]
    lib.yuyv_window_to_rgb.restype = None
    return lib


try:                                   # yuyv.c, cross-compiled for the A53
    _lib = _load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libyuyv.so"))
except OSError:
    _lib = None


def window_to_rgb(raw):
    """(H, 2W) YUYV window -> (H, W, 3) RGB uint8: libyuyv.so if loaded, else NumPy."""
    if _lib is None:
        return yuyv_to_rgb(raw)
    raw = np.ascontiguousarray(raw)
    h, w = raw.shape[0], raw.shape[1] // 2
    out = np.empty((h, w, 3), np.uint8)
    p8 = ctypes.POINTER(ctypes.c_uint8)
    _lib.yuyv_window_to_rgb(raw.ctypes.data_as(p8), raw.strides[0], 0, 0, w, h,
                            out.ctypes.data_as(p8))
    return out


class Camera:
    def __init__(self, dev="/dev/video0", width=1280, height=720, fps=200, nbuf=4):
        assert width >= OUT_W and height >= OUT_H, "mode smaller than the window"
        self.w, self.h = width, height
        self.x0 = ((width - OUT_W) // 2) & ~1          # even: 4:2:2 pairs chroma
        self.y0 = (height - OUT_H) // 2
        self.fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)

        fmt = bytearray(208)
        struct.pack_into("<I", fmt, 0, BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into("<4I", fmt, 8, width, height, YUYV, 1)   # field NONE
        fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt)
        w, h, pf = struct.unpack_from("<3I", fmt, 8)
        self.stride = struct.unpack_from("<I", fmt, 24)[0]
        if (w, h, pf) != (width, height, YUYV):
            raise RuntimeError(f"camera gave {w}x{h} {pf:#x}, asked {width}x{height} YUYV")

        parm = bytearray(204)
        struct.pack_into("<I", parm, 0, BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into("<2I", parm, 12, 1, fps)                 # timeperframe 1/fps
        fcntl.ioctl(self.fd, VIDIOC_S_PARM, parm)
        num, den = struct.unpack_from("<2I", parm, 12)
        self.fps = den / num

        req = bytearray(struct.pack("<3I8x", nbuf, BUF_TYPE_VIDEO_CAPTURE, MEMORY_MMAP))
        fcntl.ioctl(self.fd, VIDIOC_REQBUFS, req)
        self.bufs = []
        for i in range(struct.unpack_from("<I", req, 0)[0]):
            b = self._buf(i)
            fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, b)
            offset, length = struct.unpack_from("<I", b, 64)[0], struct.unpack_from("<I", b, 72)[0]
            self.bufs.append(mmap.mmap(self.fd, length, mmap.MAP_SHARED,
                                       mmap.PROT_READ | mmap.PROT_WRITE, offset=offset))
            fcntl.ioctl(self.fd, VIDIOC_QBUF, b)

        self._lock = threading.Condition()
        self._latest = None           # (raw window, timestamp, driver sequence)
        self._seen = -1
        self.skipped = 0
        self._run = True
        fcntl.ioctl(self.fd, VIDIOC_STREAMON, struct.pack("<I", BUF_TYPE_VIDEO_CAPTURE))
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _buf(i):
        b = bytearray(88)
        struct.pack_into("<2I", b, 0, i, BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into("<I", b, 60, MEMORY_MMAP)
        return b

    def _loop(self):
        rows = slice(self.y0, self.y0 + OUT_H)
        cols = slice(2 * self.x0, 2 * (self.x0 + OUT_W))
        while self._run:
            if not select.select([self.fd], [], [], 1.0)[0]:
                continue
            b = self._buf(0)
            fcntl.ioctl(self.fd, VIDIOC_DQBUF, b)
            i = struct.unpack_from("<I", b, 0)[0]
            sec, usec = struct.unpack_from("<2q", b, 24)
            seq = struct.unpack_from("<I", b, 56)[0]
            frame = np.frombuffer(self.bufs[i], np.uint8, self.stride * self.h)
            window = frame.reshape(self.h, self.stride)[rows, cols].copy()
            fcntl.ioctl(self.fd, VIDIOC_QBUF, b)
            with self._lock:
                self._latest = (window, sec + usec * 1e-6, seq)
                self._lock.notify_all()

    def read(self, timeout=5.0):              # the first frame can take ~2 s
        """Newest unseen frame -> ((1, 192, 320, 3) RGB uint8, CLOCK_MONOTONIC stamp)."""
        with self._lock:
            if not self._lock.wait_for(lambda: self._latest and self._latest[2] != self._seen,
                                       timeout):
                raise TimeoutError("no frame from the camera")
            window, t, seq = self._latest
        if self._seen >= 0:
            self.skipped += seq - self._seen - 1
        self._seen = seq
        return window_to_rgb(window)[None], t

    def close(self):
        self._run = False
        self._thread.join()
        fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, struct.pack("<I", BUF_TYPE_VIDEO_CAPTURE))
        for m in self.bufs:
            m.close()
        os.close(self.fd)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="capture self-check: rate, skips, frame age")
    p.add_argument("--dev", default="/dev/video0")
    p.add_argument("--mode", default="1280x720")
    p.add_argument("--fps", type=int, default=200)
    p.add_argument("-n", type=int, default=300)
    p.add_argument("--save", default="", help="write the last window here (.npy)")
    a = p.parse_args()
    w, h = map(int, a.mode.split("x"))
    cam = Camera(a.dev, w, h, a.fps)
    for _ in range(30):                        # start-up: the first frames take ~2 s
        cam.read()
    cam.skipped = 0
    ts, age = [], []
    for _ in range(a.n):
        x, t = cam.read()
        ts.append(t)
        age.append(time.clock_gettime(time.CLOCK_MONOTONIC) - t)
    cam.close()
    d = np.diff(ts) * 1e3
    print(f"{a.mode} @{cam.fps:.0f}: {a.n} frames, interval median {np.median(d):.2f} ms "
          f"(min {d.min():.2f}, max {d.max():.2f}), skipped {cam.skipped}, "
          f"age at read median {1e3 * np.median(age):.2f} ms, window {x.shape} {x.dtype}")
    if a.save:
        np.save(a.save, x)
