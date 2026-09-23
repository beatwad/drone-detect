# Accelerator diagnosis: why the PL runs ~2.4× slower than FINN estimated

Opened 2026-09-23. Tracks issues §9. Meant to be worked **on the old PC**, where
the build tree of the shipping bitstream lives, by someone (or a Claude session)
with no memory of the conversation that produced it. Everything needed is here
or linked.

## 1. The symptom, measured on the ZCU102

Bitstream: `deploy/resizer.bit` = the 2026-08-15 build (build_notes §10.16),
yolov8n-P3 ReLU6 W4A4 @ 192×320, 100 MHz. Output is bit-exact against the
simulation (build_notes §11.11), so this is a **speed** problem only.

`execute_on_buffers()` at batch *b* (DMA launch + wait, no packing, no Python
pre/post), median of 5:

| batch | total ms | ms/frame | FPS |
|---|---|---|---|
| 1 | 42.24 | 42.24 | 23.7 |
| 2 | 68.47 | 34.23 | 29.2 |
| 4 | 120.94 | 30.23 | 33.1 |
| 8 | 225.88 | 28.24 | 35.4 |
| 16 | 435.77 | 27.24 | 36.7 |
| 32 | 855.53 | 26.74 | 37.4 |

Fits `total = L + (b−1)·I` cleanly:

| | measured | FINN's estimate | ratio |
|---|---|---|---|
| steady-state interval `I` | **26.2 ms = ~2.62 M cycles** | 11.06 ms = 1.106 M cycles (90.4 FPS) | **2.4×** |
| single-frame latency `L` | **42.2 ms = ~4.22 M cycles** | (not estimated reliably, §10.4) | — |

The PL latency is steady to ±0.03 ms across 60 frames. In the live chain
(`deploy/live.py`) the accelerator stage is ~80% of the loop, so this gap is
where the remaining latency lives (build_notes §12.8).

## 2. The feedback loop

On the board (`ssh`/serial, as root), this reproduces the table above in a few
seconds. It is deterministic. Run it before and after any fix:

```python
# /tmp/tput.py -- latency vs throughput of the PL alone
import os, sys, time
sys.path.insert(0, "/home/root/deploy")
os.environ.setdefault("XILINX_XRT", "/usr")
os.makedirs("/lib/firmware", exist_ok=True)
from pynq.pl_server.device import Device
from driver_base import FINNExampleOverlay
from run_on_board import io_shape_dict

for b in (1, 2, 4, 8, 16, 32):
    acc = FINNExampleOverlay(bitfile_name="/home/root/deploy/resizer.bit", platform="zynq-iodma",
                             io_shape_dict=io_shape_dict, batch_size=b,
                             runtime_weight_dir="/home/root/deploy/runtime_weights/",
                             device=Device.devices[0])
    t = []
    for _ in range(5):
        t0 = time.perf_counter(); acc.execute_on_buffers(); t.append(time.perf_counter() - t0)
    t = sorted(t)[2]
    print(f"batch {b:3d}: {t*1e3:8.2f} ms total, {t*1e3/b:6.2f} ms/frame, {b/t:6.1f} FPS", flush=True)
```

A fixed bitstream is green when batch 32 approaches ~11 ms/frame (or whatever
the corrected estimate says, see H1). Keep `run_on_board.py` passing too — a
faster wrong accelerator is not a fix.

On the host the equivalent loop is FINN's rtlsim of the stitched IP (H3/H4
below): slower (minutes to hours), but it can see inside.

## 3. Already excluded, with evidence

- **Clock.** `resizer.hwh`: `PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ 99.990005`;
  `idma0`/`odma0` `ap_clk` = `zynq_ps.pl_clk0` at 99,990,005 Hz; PYNQ reports
  `fclk = 100.0 MHz`. The partition's own clock port is not listed in the hwh
  (hierarchical cell), but FINN's ZynqBuild uses one `pl_clk0` for everything.
  *Residual risk: low; confirm in the Vivado project if convenient.*
- **DMA bandwidth.** Input stream is 8 bits/beat: 192×320×3 = 184,320 beats
  = 1.84 ms/frame. Output is 24 bits/beat: 24×40×65 = 62,400 beats. Both on
  S_AXI_HP (GP2, 128-bit). Neither comes near 26 ms.
- **Unsized FIFOs.** The build ran `largefifo_rtlsim` sizing — it crashed
  *inside* `step_set_fifo_depths` once (build_notes §10.14), so the step
  existed and was used. (That does not rule out H4.)
- **Driver/Python overhead.** The table times `execute_on_buffers()` only:
  a handful of register writes and a status poll. Constant per call, and the
  per-frame slope `I` excludes it anyway.

## 4. Hypotheses, ranked — each with the prediction that falsifies it

**H1 — The 90.4 FPS figure does not describe *our* build.** build_notes
first quotes 90.42 FPS in §10.3, in the table for the **reference** network
(mdanilow's COCO detector with their hand-tuned `final_hw_config_90fps.json`).
§10.16 then repeats "90.4 FPS at the built 100 MHz" for ours without citing our
own report. If our folding differs — node names shifted when their config was
applied to our graph, `balance_folding.py` picked a slower point, a layer fell
back to PE=SIMD=1 — our design may simply be a ~38 FPS design.
*Prediction:* our `report/estimate_network_performance.json` says
`estimated_throughput_fps` ≈ 38 and `max_cycles` ≈ 2.6 M. Then the accelerator
is fine and the bug is the citation; the fix is a folding change (and a
rebuild). *Falsified if* the report says ~90 FPS / ~1.1 M cycles.
*Cost:* one file read. **Do this first.**

**H2 — An HLS layer missed II=1.** FINN's cycle estimate assumes each layer
achieves its initiation interval. If Vitis HLS scheduled one at II=2–3 (typical
suspects in this net: `StreamingMaxPool_hls` in SPPF, `UpsampleNearestNeighbour_hls`,
`StreamingConcat_hls`, `DuplicateStreams_hls`, `AddStreams_hls`, any
`*_hls` Thresholding), that layer becomes the bottleneck.
*Prediction:* one layer's csynth latency ≈ 2.6 M cycles (or its achieved II ×
its estimated cycles ≈ 2.6 M), everything else ≤ 1.1 M.
*Falsified if* every HLS layer's csynth latency is within a few % of its
`estimate_layer_cycles` entry.

**H3 — An RTL layer is slower than its estimate.** RTL layers (`MVAU_rtl`,
`ConvolutionInputGenerator_rtl`, `Thresholding_rtl`, `FMPadding_rtl`,
`StreamingDataWidthConverter_rtl`) have no csynth report. Prime suspect:
`ConvolutionInputGenerator_rtl_0` — build_notes §C4 says balanced folding moved
the bottleneck onto it. SWG timing for stride-2 / non-`parallel_window`
configurations is a known place for estimate and RTL to disagree.
*Prediction:* node-level rtlsim of the suspect gives ≈ 2.6 M cycles against an
estimate ≤ 1.1 M. *Falsified if* per-node rtlsim matches the estimates.

**H4 — FIFO back-pressure at the joins.** 20 joins (C2f concat, residual adds,
SPPF, FPN). Sizing ran, but if a reconvergent branch's FIFO is still too
shallow — sizing input rate, a depth cap, `split_large_fifos`, or depths frozen
from an earlier folding — the short branch stalls the long one every frame.
*Prediction:* stitched-IP rtlsim (`RTLSIM_PERFORMANCE`) reproduces ~2.62 M
cycles/frame, yet no single layer's own cycles exceed ~1.1 M (H2/H3 clean);
deepening the join FIFOs moves it toward 1.1 M.
*Falsified if* stitched rtlsim gives ~1.1 M (then the problem is not in the
logic at all — see H5), or if H2/H3 already explain 2.6 M.

**H5 — Something between rtlsim and silicon.** Only if stitched rtlsim says
~1.1 M while the board says 2.6 M: clock actually lower, AXI/HP-port stalls on
the DMA side, the IODMA burst setup, or the deployed bitstream ≠ the simulated
IP. *Prediction:* rtlsim ≈ 1.1 M cycles. Low prior — it would contradict three
checks in §3.

## 5. What to do on the old PC, in order

Paths per build_notes: harness `/home/alex/finn_build_mdanilow/zynq_drone/`,
FINN forks `~/Repos/finn-mdanilow` (authors' fork, used for this build) and
`~/Repos/finn-dev`. `$OUT` below = the FINN `output_dir` of the drone build;
`$FINN_HOST_BUILD_DIR` = its build dir with `code_gen_ipgen_*`.

**Step 1 — H1 (minutes).**
```bash
cat $OUT/report/estimate_network_performance.json
python3 - <<'E'
import json, os
c = json.load(open(os.environ['OUT'] + '/report/estimate_layer_cycles.json'))
for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:15]:
    print(f"{v:>10,d}  {k}")
E
```
Also diff our `final_hw_config.json` PE/SIMD against the reference
`final_hw_config_90fps.json` for the layers at the top of that list. **If
max_cycles ≈ 2.6 M, stop: H1 is the answer.** Record which layer and why its
folding differs.

**Step 2 — H2 (minutes).** Achieved latency/II of every HLS layer:
```bash
cd $FINN_HOST_BUILD_DIR
for f in $(find . -path '*sol1/syn/report/csynth.xml'); do
  layer=$(echo $f | grep -o 'code_gen_ipgen_[^/]*')
  lat=$(grep -m1 -o '<Worst-caseLatency>[0-9]*' $f | grep -o '[0-9]*$')
  ii=$(grep -m1 -o '<Interval-max>[0-9]*' $f | grep -o '[0-9]*$')
  echo "$lat $ii $layer"
done | sort -rn | head -20
```
(Tag names differ slightly between Vitis HLS versions — if empty, open one
`csynth.xml` and adjust.) Compare each against `estimate_layer_cycles.json`.
**A layer near 2.6 M = H2 confirmed.**

**Step 3 — H3/H4 (hours, mechanical).** Stitched-IP rtlsim, from the saved
checkpoint, inside the FINN container of the fork that built it:
```python
from qonnx.core.modelwrapper import ModelWrapper
from finn.core.throughput_test import throughput_test_rtlsim
m = ModelWrapper("<OUT>/intermediate_models/step_create_stitched_ip.onnx")
print(throughput_test_rtlsim(m, batchsize=4))   # cycles, throughput, latency
```
or equivalently resume `build_dataflow` at the stitched-IP checkpoint with
`generate_outputs=[DataflowOutputType.RTLSIM_PERFORMANCE]` →
`report/rtlsim_performance.json`. *API names are from memory of FINN ~0.10 —
check them in the fork at hand.* Expect ~2.6 M cycles/frame if the logic is
the cause; ~1.1 M points at H5.
- If ~2.6 M and steps 1–2 were clean → per-node rtlsim of the top-10 layers
  from step 1 (execute each node alone with `exec_mode=rtlsim` on one frame and
  read the node's `cycles_rtlsim` attribute) to separate H3 from H4.
- If per-node cycles all match the estimates → H4: inspect FIFO depths at the
  20 joins in `final_hw_config.json`; the join whose short branch has the
  shallowest FIFO relative to the long branch's latency is the first suspect.

## 6. What to bring back

- Which hypothesis held, with the one number that proves it (the report line,
  the csynth latency, the rtlsim cycle count).
- The fix: a folding change, an HLS pragma/II change, deeper FIFOs, or a
  corrected estimate — and whether it needs a rebuild (a rebuild is 20–30 h and
  needs the xczu9eg synthesis licence, which is on the old PC).
- If rebuilt: the new `resizer.bit` + `resizer.hwh` + `v8n_p3_w4a4_192x320_dequant.npz`
  (changes on every rebuild) + regenerated `out_hw.npz`, into `deploy/`. Then
  on the board: `run_on_board.py` must still PASS, and `tput.py` shows the new
  interval.
- Whatever you find, write it into build_notes §11.11 / issues §9 and correct
  the "90.4 FPS" line in §10.16 if H1 holds.

Also worth copying back regardless, so this PC can re-derive things: the build's
`report/*.json`, `final_hw_config.json`, `auto_folding_config.json`,
`build_dataflow.log`, `time_per_step.json`, and the HLS `csynth.xml` files
(`find ... | tar czf`), a few MB in total.
