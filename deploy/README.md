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
| `boot/` | `BOOT.BIN`, `image.ub`, `boot.scr` for the FAT32 partition, `rootfs.tar.gz` for the ext4 one, and the `.xsa` the image was built from |
| `petalinux/` | how that image is built, on the host |

## The one file here that is not in git

**`boot/rootfs.tar.gz`, 73 MB** — over the 50 MB commit cap, so it is
gitignored yet present in the working tree. Nothing this project needs lives
outside the project, but a fresh clone will not have this one. Rebuild it from
`petalinux/` with `boot/drone_v8.xsa` as the input — which is why the `.xsa` is
committed although nothing at runtime wants it — or point `mksd.sh` at a copy
with `ROOTFS=`.

PYNQ's 60 MB upstream sdist is deliberately **not** kept: only the wheel built
from it is ever installed, and PyPI has the sdist whenever that wheel needs
rebuilding. See `pynq_offline/README.md`.

## Bring-up, start to finish

Nothing here needs Vivado, FINN or PetaLinux. The card is written with ordinary
utilities and the bitstream is loaded on the board by `pynq.Overlay`.

### 1. On the host

```bash
sudo apt install parted gdisk dosfstools e2fsprogs screen
```

`mksd.sh` uses `sgdisk` (gdisk), `parted`, `mkfs.vfat` (dosfstools) and
`mkfs.ext4` (e2fsprogs); `screen` is for the serial console. `uv sync` is *not*
required — it is only needed for `--self-test` below, and it pulls torch.

### 2. Write the card

```bash
lsblk -d -o NAME,SIZE,TYPE,RM,MODEL        # find the card, check RM=1
./deploy/petalinux/mksd.sh /dev/sdX        # dry run, changes nothing
./deploy/petalinux/mksd.sh /dev/sdX --yes  # erases the card and writes it
```

The dry run prints every file it would write. `--yes` repartitions and asks for
`ERASE` before touching anything.

### 3. Jumpers, boot mode, console

The table is in the top-level README §9 and the reasoning in build_notes §11.9.
The short version: **SW6 [4:1] = off, off, off, on** (only SW6-3 and SW6-4 move
from the factory QSPI32 default), **J7 OPEN → ON** and **J110 1-2 → 2-3** for
USB host, everything else unchanged. Console is the CP2108 on J83:

```bash
screen /dev/ttyUSB0 115200
```

Log in as `petalinux`, password `root` — root itself has no login (`*` in
`/etc/shadow`). A fresh card has no password and forces one to be set at first
login; this is the one we set. Everything below runs as root, so start with
`sudo -i`. Leave the camera unplugged for the first boot.

### 4. Check the image came up right

Three things, each of which breaks everything downstream silently:

```bash
grep -o 'cma=512M' /proc/cmdline   # CMA reserved for XRT's buffers
ls /dev/dri/                       # renderD128 -- this is zocl
lsmod | grep zocl                  # if empty: modprobe zocl
xbutil examine                     # XRT should see a device
```

`renderD128` is the one that matters: without it PYNQ finds no device, and the
symptom reads like a Python problem. XRT is in `/usr`, not `/opt/xilinx/xrt`, and
there is no `setup.sh` to source — but PYNQ still needs `XILINX_XRT=/usr`, see
step 5.

### 5. Install PYNQ

```bash
cd /home/root/deploy/pynq_offline
pip3 install --no-index --find-links wheels pynq-3.0.1-py3-none-any.whl ipython bitstring
export XILINX_XRT=/usr
python3 -c "import pynq; print(pynq.__version__, pynq.Device.devices)"
```

The device list must be **non-empty**. PYNQ imports its device classes *only if*
`XILINX_XRT` is set, and then loads `$XILINX_XRT/lib/libxrt_core.so`; unset, it
prints `No devices found, is the XRT environment sourced?` and returns `[]` even
with zocl up. `run_on_board.py` sets it itself. Empty *with* it set means step 4.

`bitstring` is for `finn/util/data_packing.py`, which imports it at module scope.

### 6. Run it

```bash
cd /home/root/deploy && python3 run_on_board.py
```

60 frames through the real accelerator, compared against the simulation in LSB.
It also creates `/lib/firmware`, where `pynq.Overlay` hands the bitstream to
fpga_manager and which this image lacks. Measured 2026-09-23:

```
bitstream loaded, fclk = 100.0 MHz
60 frames, per-frame execute(): median 86.61 ms ... -> 11.5 FPS (driver included)
accelerator alone: runtime[ms] 42.29..., throughput[images/s] 23.64...
  max |delta| LSB  0.000000e+00
  elements off > 0.05 LSB   0 / 3744000
PASS  hardware matches the simulated graph
```

`accelerator alone` is batch 1, i.e. **latency**, not throughput — build_notes
§11.11.

### If it goes wrong

| Symptom | Cause |
|---|---|
| No console output at all | SW6 — the factory default is QSPI32, not SD |
| `IndexError` on `Device.devices[0]` | zocl did not come up (step 4), or `XILINX_XRT` unset (step 5) |
| `No module named 'bitstring'` | `bitstring` left off the install line |
| `No such file or directory: '/lib/firmware/resizer.bin'` | an old `run_on_board.py`; `mkdir -p /lib/firmware` |
| `(ro)` after `mmcblk0` in the boot log, then `error -30` panic | the SD adapter's LOCK slider — the slot reads it as write-protected |
| `ModuleNotFoundError: No module named 'IPython'` | `ipython` left off the install line |
| Camera enumerates at 480M, not 5000M | J7/J110, or a micro-AB adapter with no SuperSpeed lanes |
| `FAIL` on the comparison | that is the result: hardware disagrees with the simulated graph |

## Check the harness without a board

```bash
uv run deploy/run_on_board.py --self-test
```

Quantizing the golden outputs back to INT21 is exactly what a correct
accelerator would emit, so that round trip is a positive control and the same
data with one element moved by one is a negative control. Both pass on the
host, which is what makes a failure on hardware mean hardware.
