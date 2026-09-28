# Latency: what is left, and what each remaining lever would cost

Written 2026-09-28, after the software side was worked down (build_notes
§12.8–12.12). For deciding the next step on the old PC, where the model and
build sources live. Nothing here is started; the gains are estimates unless a
measurement is cited.

## Where the time goes now

`deploy/live.py`, default settings (187.5 MHz bitstream, pipeline depth 3,
decode in the DMA buffer, RGB in the capture thread, `gc.freeze()`), 70,000
frames, ms (build_notes §12.12):

| # | stage | median | p95 | p99 | max |
|---|---|---|---|---|---|
| 1 | driver stamp → DQBUF (USB transfer + thread wake-up) | 5.51 | 6.44 | 9.88 | 11.76 |
| 2 | DQBUF → ready (crop + YUYV → RGB, C) | 1.75 | 2.46 | 2.77 | 4.10 |
| 3 | ready → read (waits for an accelerator slot) | 2.83 | 5.86 | 7.82 | 10.87 |
| 4 | submit | 0.49 | 0.78 | 0.82 | 1.61 |
| 5 | **in PL** | **23.14** | 25.10 | 25.98 | 27.44 |
| 6 | decode (`decode_packed`) | 2.08 | 2.47 | 2.59 | 9.13 |
| 7 | track (boxes in view; ~0.1 on an empty scene) | 1.81 | 2.73 | 2.86 | 4.45 |
| | **age at aim** | **37.26** | 40.80 | 43.02 | 46.76 |
| | aim gap (time between aims) | 8.64 | 14.83 | 16.78 | 19.97 |

35.8 ms on an empty scene; 0 of 70,000 frames over 60 ms; ~111–116 FPS.
**Exposure is not in these numbers** — the driver stamps the first USB packet,
after exposure (§12.9) — so glass-to-aim = exposure + age at aim.

The PL is ~62% of it, frame delivery (1–3) ~28%, the rest of the CPU ~12%.

## Software: close to exhausted (~1–2 ms left in total)

- Stage 3 is ~2.4 ms inherent (half a camera interval, newest-frame-only);
  ~0.5 ms is not.
- The capture thread's tail (stage 1 p99 9.9 vs median 5.5): try `chrt -f`
  real-time priority, one 10-minute run with `live.py` to compare. Keep it below
  the kernel's IRQ threads (FIFO 50), or USB itself starves.
- Moving the whole main loop to C: ~1 ms, and it means leaving PYNQ.

## Levers that matter

### 1. Short fixed exposure — cheap, can be done on this PC

**Gain:** not visible in age at aim, but real glass-to-aim shrinks by whatever
auto exposure was using — easily 5–10 ms indoors — and a fast drone blurs less.
**Cost:** a camera setting (`v4l2-ctl -c exposure_auto=1 -c
exposure_absolute=N`, N in 100 µs; `gain` 0–190). Darker, noisier frames.
**To decide:** the same drone image in front of the camera under auto and at
e.g. 1 ms / 2 ms fixed with gain raised: compare detection confidence and box
stability (`live.py` status lines), and whether the frame rate holds (manual
exposure ran 720p at ~210 fps instead of ~249, §12.9). Needs the real lens and
real light before it means much (issues §1–2).

### 2. More parallel folding — the big one; needs the old PC and a rebuild

**Why it helps latency, not just throughput:** single-frame latency is ~3.8×
the steady-state interval (22.6 vs 5.94 ms, §11.14). Latency is the time to
fill the pipeline, which scales with how fast each layer on the path runs —
so raising PE/SIMD on the layers that dominate it shortens both.
**Gain (estimate):** doubling the parallelism of the dominant layers →
roughly −8 to −11 ms of PL latency and ~2× throughput. Unmeasured.
**Cost:** a rebuild — codegen + ipgen ~4 min, stitch ~13 h, synthesis +
implementation ~42 min (§11.13) — with the §10.14 HLS-race check and the
`BD 5-336` harness (§10.15).
**The constraint is BRAM:** 720 of 912 tiles (79%) used, and FINN
under-estimates BRAM ×2.9 because it leaves FIFOs out (§10.7). LUT (30%) and
DSP (13%) have plenty of room. More PE splits weight memories into more, smaller
BRAMs, so BRAM can grow even when the weights do not.
**How to evaluate before paying 13 h:**
1. From the build's `estimate_layer_cycles.json`, find which layers bound the
   latency (not only the interval) — the long chain through the backbone into
   SPPF and back up the FPN to P3.
2. `export/balance_folding.py` for candidate foldings, budgeting BRAM at ×2.9.
3. Per-node Verilator timing of the changed layers with `export/rtlsim/tb.sh` /
   `layer.sh` (minutes each). Whole-design latency in `export/rtlsim/` needs a
   stitched netlist, i.e. the rebuild — the model there is tied to the current
   build tree (see its README).
4. Only then rebuild, and re-check the FIFO depths at the P3/P4 skip joins —
   their sizing cap is what cost 2.4× last time (§11.12).

### 3. Camera over MIPI straight into the PL — the target architecture

**Gain (estimate):** no USB transfer (5.5 ms), no RGB conversion or slot wait
on the CPU, and the network's first layers can start on the first rows while
the rest of the frame is still being read out, instead of waiting for the whole
frame. Plausibly −10 to −15 ms, and almost no jitter, since Linux stops being on
the path. Also the low-power shape (CLAUDE.md, "Power shape").
**Cost:** a different camera or an FMC camera module, a PL capture +
preprocessing front end, and eventually decode + tracker in PL. A project of its
own. The hardware trigger (issues §4) belongs to the same step.

### 4. Higher PL clock — little left

Timing met at 187.48 MHz with +0.55 ns (§11.13). 200 MHz would give ~−1.5 ms of
PL latency; the margin says it is not free. Low priority.

### 5. Model changes — not before real footage

Latency is set by the network's depth: SPPF needs the whole frame, and P3 waits
for it through the FPN. Dropping SPPF, a shallower backbone, or a smaller input
would shorten it, but each means retraining, a rebuild, and a risk to accuracy
that cannot be judged until there is footage through the real lens (issues §1).

### 6. Leaving PYNQ — small for latency, bigger for robustness

What PYNQ still does for us, and what would replace it:

| job | now | without PYNQ | effort |
|---|---|---|---|
| load the bitstream | `Overlay`: `.bit` → `.bin` in `/lib/firmware`, name to `fpga_manager` sysfs | the same by hand (strip the `.bit` header, byteswap, write the name to `/sys/class/fpga_manager/fpga0/firmware`), or `fpgautil` if the image has it | low |
| set the PL clock | `Clocks.fclk0_mhz` writes PS `PL0_REF_CTRL` | the same register via `mmap` of `/dev/mem` | low |
| DMA registers | `MMIO.read/write` | `mmap` of `/dev/mem` (or UIO) — a pointer in C | low |
| **DMA buffers** (physically contiguous, physical address, cache sync) | `pynq.allocate` over XRT (zocl, CMA) | XRT's own C++ API (`xrt::bo`, `sync`) — XRT stays in the image; or `u-dma-buf`, which needs a PetaLinux rebuild | **medium** — the one non-trivial part |

The rest of FINN's driver is already mostly bypassed: `pipeline.py` writes the
IODMA registers itself, UINT8 input packing is a view, and the output is decoded
by our C (`decode_packed`). Addresses are fixed (`idma0` 0xA0000000, `odma0`
0xA0001000), so losing PYNQ's `.hwh` parsing costs little.

**Lose:** editing Python on the board with no cross-compile; running FINN's
driver unchanged on another PYNQ board (e.g. Kria).
**Gain:** ~1 ms median and a tighter tail (a C loop, no GIL, no garbage
collector); no hand-built PYNQ wheel and its 33 dependencies, no
`XILINX_XRT=/usr` / `/lib/firmware` quirks — and PYNQ + pydantic's heap is what
made a full GC cost ~200 ms (§12.12); fast start-up; and code that is bare
register + DMA access, which is what the R5F / bare-metal endpoint in CLAUDE.md
needs anyway.
**Verdict:** not worth it for latency now. Worth it when moving to the
deployment board or putting control on the R5F. A middle step if wanted: keep
PYNQ only to load the bitstream and allocate buffers at start-up, and run the
hot loop (capture → submit → poll → decode → track) as one C thread — most of
the gain for a fraction of the rework.

## Suggested order

1 now (cheap, real glass-to-aim) → 2 when on the old PC (the largest step that
keeps the current hardware) → 3 when the deployment camera is chosen. 4 and 5
only for a specific reason.
