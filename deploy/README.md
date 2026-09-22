# `deploy/` — everything the board runs

This directory **is** the board's `/home/root/deploy`, minus two host-side
subdirectories. `petalinux/mksd.sh` copies it verbatim, so anything added here
lands on the card.

| | |
|---|---|
| `resizer.bit` | the shipping bitstream, 2026-08-15, timing closed at 100 MHz. Byte-identical to `zynq_drone/.../top_wrapper.bit` and to the `system.bit` in the PetaLinux image |
| `resizer.hwh` | hardware handoff; `pynq.Overlay` will not load without it |
| `driver_base.py` | `FINNExampleOverlay`. **FINN never generated this** — it was recovered by hand, build_notes §10.17, and does not come back from a rebuild |
| `driver.py`, `validate.py` | FINN's own stubs, kept as they came |
| `finn/`, `qonnx/` | six trimmed modules `driver_base` imports; the board has neither package |
| `runtime_weights/` | empty by design — the network has no external weights — but the driver expects the directory |
| `run_on_board.py` | the bring-up check: 60 frames through real hardware against the simulation |
| `inputs.npz`, `out_hw.npz` | that check's inputs and golden outputs |
| `postprocess.py` | dequantize → DFL → boxes → sigmoid → NMS, NumPy only |
| `v8n_p3_w4a4_192x320_dequant.npz` | per-channel scale and bias; **changes on every rebuild** |
| `track.py` | seed → IoU cluster → WBF → Kalman → centring gate. Never run on a real detection |
| `pynq_offline/` | a pure-Python PYNQ 3.0.1 and its aarch64 wheels, so the board needs neither network nor compiler. See its README |
| `boot/` | `BOOT.BIN`, `image.ub`, `boot.scr` for the FAT32 partition, and the `.xsa` the image was built from |
| `petalinux/` | how that image is built, on the host |

## The one file that is not here

**`rootfs.tar.gz` (73 MB) is too large to commit.** `mksd.sh` reads it from
`/home/alex/petalinux/projects/drone/images/linux/` by default; override with
`ROOTFS=/path/to/rootfs.tar.gz`. If it is ever lost it can be rebuilt from
`petalinux/` with `boot/drone_v8.xsa` as the input — which is why the `.xsa` is
committed even though nothing at runtime needs it.

`pynq_offline/` similarly omits the 60 MB upstream sdist; the wheel built from
it is what gets installed, and `pure-python.patch` reproduces the build.

## Write the card

```bash
./deploy/petalinux/mksd.sh /dev/sdX          # dry run, changes nothing
./deploy/petalinux/mksd.sh /dev/sdX --yes    # erases the card and writes it
```

## Check the harness without a board

```bash
uv run deploy/run_on_board.py --self-test
```

Quantizing the golden outputs back to INT21 is exactly what a correct
accelerator would emit, so that round trip is a positive control and the same
data with one element moved by one is a negative control. Both pass on the
host, which is what makes a failure on hardware mean hardware.
