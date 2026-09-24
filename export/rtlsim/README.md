# rtlsim — timing the built accelerator without the board

Tooling that found why the PL ran at 26.2 ms/frame instead of FINN's 11.06
(build_notes §11.12). It simulates the **stitched** design — the exact Verilog
Vivado turned into `resizer.bit` — in Verilator, fast enough to run many
experiments an hour, and it matches the board to 64 cycles on a 2.6M-cycle
interval.

Sources live here; objects and logs go to `$WORK`
(default `~/finn_build_mdanilow/diag`). Everything runs inside the FINN image the
build used (it has Verilator); no Vivado, no FINN Python.

It reads the build tree directly (`~/finn_build_mdanilow/vivado_stitch_proj_a5f6s728`,
`code_gen_ipgen_*`, `pyverilator_*`), so it is tied to that build. Paths are at the
top of each script.

## Whole design

```bash
./design.sh build                      # verilate ~2,170 sources, ~15 min, once
./design.sh run 4                      # 4 frames at full input rate
./design.sh run 6 1150000 +fifo_report # paced: a frame every 1.15M cycles
./set_depths.sh fix 3 -- 165=131072 246=32768   # per-FIFO depths, by stitch index
```

`design_tb` prints the cycle each output frame completes: frame 1 is the
latency, the gaps after it are the steady-state interval. The built design gives
**4,207,300 / 2,623,215**; the board measured 4,223,578 / 2,623,279.

The 173 Xilinx `axis_data_fifo` instances are replaced by
`axis_data_fifo_sim.sv`, which Verilator can compile. It behaves as the built
depth unless told otherwise, and takes, at run time:

| plusarg | |
|---|---|
| `+fifo_scale=K` | every deep FIFO at K× its built depth (≤ 4, the allocation) |
| `+fifo:<%m>=D` | one instance to D words — use `set_depths.sh`, see below |
| `+fifo_list` | print every instance with depth and width at start |
| `+fifo_report` | print every instance's max occupancy at the end |
| `+stat_from=C` | with `+fifo_report`: % of cycles full / empty from cycle C on |

The 287 shallow FIFOs are FINN's own `Q_srl` and are not adjustable here.

**Use `set_depths.sh` for per-instance depths, never hand-written plusargs.** The
key the module matches is `%m` inside its `initial` block, which is
`<instance>.unnamedblk1`, not `<instance>`. A key without the suffix matches
nothing and the run silently simulates the built design — that cost five wrong
conclusions once. `set_depths.sh` builds the key correctly and reports how many
limits actually changed; if that number is 0, nothing was tested.

Stitch indices (`StreamingFIFO_rtl_<N>` in the netlist) are not the FINN model's
names — the stitch renumbered them. `fifo_map.py` maps each deep FIFO to its
producer and consumer from the netlist; `resize_plan.py` turns a `+fifo_list`
run and a `+fifo_report` run into a proposed set of depths and its BRAM cost.

## One node

```bash
./tb.sh MVAU_rtl_3 69120 61440         # one node, ideal weight stream: 1,105,928 cycles
./layer.sh MVAU_rtl_3 69120 61440      # the node + its memstream, as built: 1,105,928
```

Both need the node's `pyverilator_<node>_*` build, which FINN's own
`PrepareRTLSim` leaves behind. Arguments are the node's input and output words per
frame. `tb.sh` links a C++ testbench against those objects — about a million
times faster than FINN's `cycles_rtlsim`, which steps the clock from Python and
did not finish `MVAU_rtl_3` in 40 minutes.

## Timing without the reset net

```bash
cd $WORK && vivado -mode batch -source .../data_paths.tcl
```

Opens the routed checkpoint read-only, excludes the `proc_sys_reset` cells from
analysis and reports the worst data paths. On the shipping build: +4.552 ns at
10 ns, ~183 MHz, versus the reset-limited 8.235 ns in Vivado's own summary.
