# Open Issues

Problems known to be unsolved, in one place. **Nothing here is measured fact —
that is [build_notes.md](build_notes.md); nothing here is a decision — those are
in [CLAUDE.md](../../CLAUDE.md).** This file is the list of things that will bite
us if we forget them, each with what "solved" would look like.

An issue leaves this file when it is measured, decided, or refuted — and the
outcome goes to build_notes (measured), CLAUDE.md (decided) or
[research/](research/) (sourced), not here.

Last reviewed 2026-09-11.

---

## 1. No real footage, through the real lens — *the root issue*

Every threshold in the pipeline is a placeholder, and every accuracy number we
have comes from diverse web photos rather than from one fixed camera pointed at
one real drone.

- The val "mid" regime is web imagery at assorted focal lengths. The deployment
  image is a centre crop through one lens with one sensor, one noise profile,
  one motion-blur signature. **These are not the same distribution**, and
  mid ≈ close — the finding the whole optics argument rests on — is measured on
  the former.
- `deploy/track.py` (seed, IoU cluster, WBF, Kalman, centring gate) has **never
  run on a real detection**. Its own docstring says the thresholds are
  placeholders to be measured. They cannot be tuned without footage.
- The training set has no images from this camera at all.

**Solved when:** a few hundred frames of a real drone, captured through the
chosen lens at the intended range, exist as a held-out val set; per-regime mAP
and centre error are re-measured on it; and the tracker thresholds are fitted to
real tracks rather than synthetic ones. Collecting enough to *train* on (not
just validate) is the larger follow-on.

**Blocked on:** the lens (§2). The camera is here.

## 2. Lens FOV × crop is still unpicked

CLAUDE.md open question 1 has a working answer — a 30–45° full-frame lens plus a
416×416 centre crop puts a 30 cm drone at 10 m in the *mid* regime — but the
lens has not been bought, and three parts are genuinely unresolved:

- **Acquisition.** A 10–14° effective FOV is a narrow search cone. Downscaling
  the full frame to find the target first does not rescue it (~12 px, at or
  below the measured detection floor). Needs platform slew, an external cue, or
  a second wide sensor. **No answer yet.**
- **Fixed ROI vs steerable crop.** Camera-side ROI is ~2.6× faster readout and
  12× less USB traffic, but the window is fixed. A crop steered by the Kalman
  prediction needs full-frame transfer. Minimum latency vs wider capture area.
  **One leg of this argument is gone (2026-09-11):** under YUYV capture a
  host-side crop costs **0.05 ms** — the crop precedes the colour conversion, so
  only the window is touched (build_notes §12.6). "Zero CPU work" is no longer a
  reason to prefer the camera. Readout time and USB traffic still are.
- **Does this camera even have a camera-side ROI?** Its UVC mode list is fixed
  — 1920×1200, 1920×1080, 1600×1200, 1280×960, 1280×720, 640×480, 512×512 —
  with no arbitrary window. But it *advertises* digital PTZ: `Zoom, Absolute`
  100–200 (1–2×), `Pan/Tilt, Absolute` ±648000 arcsec in 1° steps, and a
  `Region of Interest Auto Ctrls` flag. If those work, that is a steerable
  camera-side ROI and the fork above resolves itself. **Be skeptical:** ±180°
  pan on a fixed camera reads like a generic bridge descriptor, and bridge-side
  zoom typically crops and then *upscales* back to the output size — which buys
  no readout time and costs interpolation blur.
  **Cheap to settle:** capture at zoom 100 and 200 and test whether the second is
  a 2× resample of the first's centre, and whether the frame interval moves. Do
  the same across 1920×1200 / 1280×720 / 640×480 to learn whether the smaller
  modes **crop** the sensor (narrower FOV, native angular resolution — what we
  want) or **scale** it (same FOV, resolution discarded). The effective FOV,
  and therefore the lens, depends on that answer.

**Solved when:** a lens is chosen and mounted, and the effective FOV is computed
from the actual sensor size and focal length rather than estimated — which
requires knowing whether the capture mode crops or scales the sensor.

## 3. Frame rate costs light, and we have not measured what that costs accuracy

Exposure is capped by the frame interval: 8.3 ms at 120 fps. Measured indoors,
mean grey level falls **98 → 43** going 30 → 120 fps, and to 27 at 200
(build_notes §12.4). Against a bright sky this may be free; at dusk it is not.

We know the brightness cost exactly and the **accuracy cost not at all**.
Faster capture is not obviously better: it buys timestamp precision and shorter
motion blur, and it spends signal-to-noise.

**Solved when:** mAP and centre error are measured as a function of capture rate
on real footage (§1), in the lighting the system is meant to work in, and a
capture rate is chosen on that basis rather than on "the camera can do 120".

## 4. No hardware trigger

Deferred 2026-08-19 — the camera was bought without one, and at 120 fps the
8.3 ms inter-frame interval bounds timestamp error well enough for bring-up.
**The requirement returns the moment aim error is the thing being measured.**

Without a trigger, exposure time is unknown to the software, so variable USB and
Linux delay lands directly in aim error instead of being a measured quantity the
Kalman can predict forward from. With PL driving the trigger, t₀ is exact.

**Solved when:** the trigger is driven from PL and t₀ is timestamped at exposure,
or it is shown by measurement that the residual jitter is small against the
error budget.

## 5. USB is the wrong interface for deployment

USB is a **PS** peripheral. Frames land in PS DDR and are DMA'd to PL, so the
whole acquisition path is scheduled by Linux, and PS + DDR must stay awake.

- **Latency/jitter:** every millisecond of scheduler jitter is aim error, same
  as §4. Accepted for the baseline (~10–25 ms, inside the 50–100 ms budget).
- **Power:** on the Z7020 reference, **1.9 W of 2.55 W was PS + DDR idle** while
  the fabric drew 0.22–0.65 W. USB keeps exactly that awake, and it fights the
  2–5 W deployment target.

The endpoint is MIPI straight into the fabric, with preprocessing, decode and
tracker in PL and the A53s parked — which is both the low-latency and the
low-power architecture. Note the ZCU102 has **no native MIPI**; this needs a
camera FMC module on the devkit, or the deployment part's own pins.

**Solved when:** a MIPI sensor streams into PL and the frame path no longer
traverses Linux. Until then: keep preprocessing behind an interface and do not
let host code assume a USB-shaped frame source.

## 6. Post-processing runs on the A53s, and that is known to be wrong

Decided 2026-08-20 as a **baseline** — prove the chain end to end; the arithmetic
is ~10k ops/frame, microseconds anywhere. The endpoint is the whole chain in PL.
Same jitter argument as §4 and §5. Tracked here so the temporary does not become
permanent by default.

**Solved when:** decode and tracker are in PL and nothing between exposure and
aim command is scheduled by Linux.

## 7. Capture aspect ratio silently decides the network input

At `imgsz=320` Ultralytics letterboxes to the long side, so the capture
resolution picks the input shape: 640×480 → 256×320, 1280×720 → **192×320**,
1920×1200 → 224×320 (build_notes §12.5).

**Only 16:9 capture matches the built bitstream's 192×320.** The host does not
care — the net is fully convolutional. The FPGA cannot run the other two at all,
because the shape is baked into the bitstream. This makes host-side benchmarks
quietly incomparable to the board unless capture is 1280×720.

**Solved when:** the host capture path pins the shape the bitstream expects, and
the preprocessing (crop, not resize — CLAUDE.md open question 1) produces it
directly rather than by letterboxing whatever the camera happened to send.

## 8. ~~`pip install pynq` over the built XRT is unverified~~ — SOLVED 2026-09-23

`run_on_board.py` passed on hardware: 60 frames, 0 of 3,744,000 outputs off.
It took two missing wheels, `XILINX_XRT=/usr` and `/lib/firmware`; build_notes
§11.11. Kept below as it was.


The one genuinely unknown step in board bring-up: no official PYNQ image exists
for ZCU102, so PYNQ goes on top of our own PetaLinux 2022.2 image. Everything
either side of it is done — the bitstream, the driver, the golden set, and
`deploy/run_on_board.py`, which passes positive and negative controls on the
host. See [research/pynq-on-zcu102.md](research/pynq-on-zcu102.md).

**Blocked on:** the board arriving.
**Solved when:** `run_on_board.py` compares real INT21 against the 60-frame
golden set on hardware and passes at 0.05 LSB.

## 9. Real throughput is 2.4× below FINN's estimate, and power is unmeasured

**Measured 2026-09-23** (build_notes §11.11): single-frame latency ≈ 42 ms,
steady-state ≈ 26.2 ms/frame ≈ **38 FPS** at 100 MHz — against the **90.4 FPS**
FINN's cycle estimate predicted. DMA is not it (~7 MB/s). Either a layer is
slower than `estimate_layer_cycles` says, or FIFO back-pressure throttles the
pipeline. Power under load is still only the Vivado report (5.10 W, PS8 2.74 W).

**Solved when:** the slow stage is named (`RTLSIM_PERFORMANCE` on the stitched
IP, or per-layer counters) and either fixed or accepted with a number, and board
power is measured under a continuous frame stream.

## 10. The deployment part is undecided, and BRAM is the constraint

Candidates: ZU3EG (Ultra96-V2, 7.6 Mbit), ZU5EV / Kria K26 (~23 Mbit incl.
URAM), K24. The ZCU102 is deliberately oversized and its headroom must not drive
architecture. **FINN under-estimates BRAM ×2.86** because
`estimate_layer_resources` excludes FIFOs; the shipping build sits at 75.8% of
the ZCU102's 912 tiles. Budget BRAM at ×2.9 the estimate — it, not LUT, is the
go/no-go for a smaller part.

**Solved when:** a part is chosen against a measured footprint, with the tracker
and preprocessing in PL (§5, §6) included in the budget rather than assumed free.
