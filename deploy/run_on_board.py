"""Board-side bring-up check: does the real accelerator match the simulation?

Runs ON THE ZCU102, not on the host. Copy the whole deployment package
(`resizer.bit`, `resizer.hwh`, `driver_base.py`, `postprocess.py`, this file)
plus `inputs.npz` and `out_hw.npz` into the board's rootfs and run it there.

This is the one junction in the chain that host-side verification cannot reach.
Everything from the QAT model down to `step_yolov8_convert_to_hw_layers` was
checked on the host (build_notes §10.18) and the hardware-layer graph came out
bit-exact. What is untested is whether the *bitstream* computes what that graph
says it does, and whether the driver packs and unpacks the streams correctly.

THREE SHAPE/DTYPE CONVERSIONS, ALL EASY TO GET WRONG
  `inputs.npz`  is (N, 3, 192, 320) float32 with integral values 0..255 — NCHW,
                because it was made for `execute_onnx`. The accelerator wants
                (1, 192, 320, 3) UINT8 NHWC. Verified integral, so the cast is
                lossless; do not rescale to 0..1.
  `out_hw.npz`  is (M, 65, 24, 40) float32 NCHW and is ALREADY DEQUANTIZED: it
                came from the checkpoint *before* `step_create_dataflow_partition`
                split the per-channel Mul/Add out into the parent graph. The
                accelerator emits raw INT21 NHWC instead, so the board side must
                apply `postprocess.dequantize` before comparing.
  M vs N        all 60 frames have a golden output (2026-08-19). The comparison
                still runs over min(M, N), so a shorter golden set is fine — any
                extra frames are executed anyway, for timing. The simulation is
                deterministic: re-running it reproduced the first 8 frames of the
                earlier pass bit for bit.

VERDICT
  The integer path should be exact and the dequantization is the same float32
  arithmetic on both sides, so the honest expectation is a difference of zero.
  Deltas are reported in LSB — units of the per-channel quantization step — so
  the number means the same thing as in the host-side verification scripts.

  The tolerance is 0.05 LSB, and neither end of that is arbitrary. A genuine
  disagreement is an accelerator output off by one, i.e. 1.0 LSB. The floor is
  float32 itself: the largest logits are ~1.5e5 LSB, where a float32 ULP is
  2**-6, so re-expressing an exact integer already costs up to 0.008 LSB. The
  threshold sits an order of magnitude above that noise and 20x below a real
  error, so it cannot be reached by accident from either side.
"""
import argparse
import os
import time

import numpy as np
from qonnx.core.datatype import DataType
from pynq.pl_server.device import Device

from driver_base import FINNExampleOverlay
from postprocess import dequantize

HERE = os.path.dirname(os.path.abspath(__file__))
TOL_LSB = 0.05      # see the docstring: 1.0 = off by one, ~0.008 = float32 ULP

io_shape_dict = {
    "idt": [DataType["UINT8"]],
    "odt": [DataType["INT21"]],
    "ishape_normal": [(1, 192, 320, 3)],
    "oshape_normal": [(1, 24, 40, 65)],
    "ishape_folded": [(1, 192, 320, 3, 1)],
    "oshape_folded": [(1, 24, 40, 65, 1)],
    "ishape_packed": [(1, 192, 320, 3, 1)],
    "oshape_packed": [(1, 24, 40, 65, 3)],
    "input_dma_name": ["idma0"],
    "output_dma_name": ["odma0"],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 1,
}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bitfile", default=os.path.join(HERE, "resizer.bit"))
    p.add_argument("--inputs", default=os.path.join(HERE, "inputs.npz"))
    p.add_argument("--golden", default=os.path.join(HERE, "out_hw.npz"))
    p.add_argument("--dequant", default=os.path.join(HERE, "v8n_p3_w4a4_192x320_dequant.npz"))
    p.add_argument("-n", type=int, default=0, help="frames to run (0 = all in inputs.npz)")
    p.add_argument("--save", default="", help="write raw INT21 output to this .npz")
    args = p.parse_args()

    x = np.load(args.inputs)["x"]
    if args.n:
        x = x[: args.n]
    assert np.array_equal(x, np.round(x)), "inputs are not integral — wrong file?"
    x = x.transpose(0, 2, 3, 1).astype(np.uint8)          # NCHW float -> NHWC uint8

    dq = np.load(args.dequant)
    scale, bias = dq["scale"], dq["bias"]

    accel = FINNExampleOverlay(
        bitfile_name=args.bitfile,
        platform="zynq-iodma",
        io_shape_dict=io_shape_dict,
        batch_size=1,
        runtime_weight_dir=os.path.join(HERE, "runtime_weights/"),
        device=Device.devices[0],
    )
    print(f"bitstream loaded, fclk = {accel.fclk_mhz:.1f} MHz")

    raw, dt = [], []
    for k in range(x.shape[0]):
        t0 = time.perf_counter()
        y = accel.execute(x[k : k + 1])
        dt.append(time.perf_counter() - t0)
        raw.append(y[0])
    raw = np.stack(raw)                                    # (N, 24, 40, 65) INT21
    if args.save:
        np.savez(args.save, y=raw)

    dt = np.array(dt)
    print(f"\n{len(dt)} frames, per-frame execute(): "
          f"median {1e3 * np.median(dt):.2f} ms, min {1e3 * dt.min():.2f} ms "
          f"-> {1.0 / np.median(dt):.1f} FPS (driver included)")
    print("accelerator alone: " + ", ".join(
        f"{k} {v}" for k, v in accel.throughput_test().items()
        if k in ("runtime[ms]", "throughput[images/s]")))

    gold = np.load(args.golden)["y"]                       # (M, 65, 24, 40) float NCHW
    m = min(len(gold), len(raw))
    got = dequantize(raw[:m], scale, bias)                 # -> (m, 65, 24, 40) float
    d = np.abs(got - gold[:m])
    lsb = d / scale.reshape(1, -1, 1, 1)

    print(f"\ncompared {m} frames against the simulation "
          f"({len(raw) - m} more ran without a golden output)")
    print(f"  max |delta|      {d.max():.6e}")
    print(f"  max |delta| LSB  {lsb.max():.6e}")
    print(f"  elements off > {TOL_LSB} LSB   {int((lsb > TOL_LSB).sum())} / {lsb.size}")
    for k in range(m):
        print(f"    frame {k}: {lsb[k].max():.3e} LSB")

    ok = lsb.max() <= TOL_LSB
    print("\n" + ("PASS — hardware matches the simulated graph"
                  if ok else "FAIL — hardware disagrees with the simulation"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
