"""Latency vs throughput of the PL alone, on the board.

    FCLK=187.5 python3 tput.py

Times `execute_on_buffers()` at batch 1..32 (DMA launch + wait: no packing, no
Python pre/post), median of 5 runs each. A pipeline takes T(b) = L + (b-1)*I,
so latency L is the batch-1 time and the interval I is the slope between the
two largest batches. Expected at 187.5 MHz: L 22.6 ms, I 5.94 ms = 168 FPS
(build_notes §11.14). Was a snippet in accelerator_diagnosis.md §2.

Stop drone-detect.service first: it holds the accelerator.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("XILINX_XRT", "/usr")          # see run_on_board.py
os.makedirs("/lib/firmware", exist_ok=True)
from pynq.pl_server.device import Device
from pynq.ps import Clocks
from driver_base import FINNExampleOverlay
from run_on_board import io_shape_dict

T = {}
for b in (1, 2, 4, 8, 16, 32):
    acc = FINNExampleOverlay(bitfile_name=os.path.join(HERE, "resizer.bit"), platform="zynq-iodma",
                             io_shape_dict=io_shape_dict, batch_size=b,
                             runtime_weight_dir=os.path.join(HERE, "runtime_weights/"),
                             device=Device.devices[0], fclk_mhz=float(os.environ.get("FCLK", "187.5")))
    t = []
    for _ in range(5):
        t0 = time.perf_counter(); acc.execute_on_buffers(); t.append(time.perf_counter() - t0)
    T[b] = t = sorted(t)[2]
    print(f"batch {b:3d}: {t*1e3:8.2f} ms total, {t*1e3/b:6.2f} ms/frame, {b/t:6.1f} FPS"
          f"  @ {Clocks.fclk0_mhz:.2f} MHz", flush=True)

I = (T[32] - T[16]) / 16
print(f"\nlatency L = {T[1]*1e3:.2f} ms, interval I = {I*1e3:.2f} ms = {1/I:.1f} FPS")
