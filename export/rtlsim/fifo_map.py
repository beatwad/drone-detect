"""fifo_map.py REPORT.log
Joins the per-FIFO stats printed by design_tb (+fifo_report) with the stitched
netlist, so each deep FIFO reads as  producer -> [FIFO] -> consumer, in network
order. 'full' = % of steady-state cycles the FIFO was full (its producer was
blocked); 'empty' = % it was empty (its consumer was starved).
"""
import re, sys

SRCS = "/home/alex/finn_build_mdanilow/vivado_stitch_proj_a5f6s728/all_verilog_srcs.txt"
P = "StreamingDataflowPartition_1_"
net = [l.strip() for l in open(SRCS) if l.strip().endswith(
    "bd/StreamingDataflowPartition_1/synth/StreamingDataflowPartition_1.v")][0]
txt = open(net).read()
inst = {}
for m in re.finditer(r"\n\s+(\w+)\s+(\w+)\s*\n?\s*\((.*?)\);", txt, re.S):
    ports = {k.lower(): v for k, v in re.findall(r"\.(\w+)\((\w+)\)", m.group(3))}
    if ports:
        inst[m.group(2).replace(P, "")] = ports
cons, prod = {}, {}
for i, ps in inst.items():
    for p, n in ps.items():
        if p.endswith("_tvalid"):
            (cons if p.startswith(("in", "s_axis")) else prod)[n] = i


def up(i):      # nearest non-FIFO upstream of instance i
    while True:
        n = [v for p, v in inst[i].items() if p.startswith(("in", "s_axis")) and p.endswith("_tvalid")]
        i = prod.get(n[0]) if n else None
        if i is None or "StreamingFIFO" not in i:
            return i or "INPUT"


def down(i):    # nearest non-FIFO downstream of instance i
    while True:
        n = [v for p, v in inst[i].items() if p.startswith(("out", "m_axis")) and p.endswith("_tvalid")]
        i = cons.get(n[0]) if n else None
        if i is None or "StreamingFIFO" not in i:
            return i or "OUTPUT"


rows = []
for line in open(sys.argv[1]):
    m = re.search(r"StreamingFIFO_rtl_(\d+)\.fifo\.inst depth (\d+) limit (\d+) max (\d+).*full ([\d.]+) empty ([\d.]+)", line)
    if m:
        k = int(m.group(1)); name = f"StreamingFIFO_rtl_{k}"
        rows.append((k, int(m.group(2)), float(m.group(5)), float(m.group(6)), up(name), down(name)))
print(f"{'fifo':>8s} {'depth':>6s} {'full%':>6s} {'empty%':>6s}   producer -> consumer")
for k, d, f, e, u, w in sorted(rows):
    flag = "  <<< FULL" if f > 50 else ("  ... empty" if e > 50 else "")
    print(f"rtl_{k:<4d} {d:>6d} {f:6.1f} {e:6.1f}   {u} -> {w}{flag}")
