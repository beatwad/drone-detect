"""resize_plan.py LIST.log PACED.log [MARGIN]
From a +fifo_list run (widths) and a paced +fifo_report run (max occupancy with
large FIFOs), propose new depths: max(built, need * MARGIN), rounded up to a
power of two like FINN's axis_data_fifo segments. Prints the per-FIFO changes,
the total extra bits / BRAM36, and writes the plusargs to apply the plan.
"""
import math, re, sys

margin = float(sys.argv[3]) if len(sys.argv) > 3 else 1.10
width, built, need, capped = {}, {}, {}, set()
for line in open(sys.argv[1]):
    m = re.search(r"^FIFO (\S+) depth (\d+) width (\d+)", line)
    if m:
        k = m.group(1).replace(".unnamedblk1", "")   # %m inside initial carries the block
        width[k] = int(m.group(3)); built[k] = int(m.group(2))
for line in open(sys.argv[2]):
    m = re.search(r"FIFOMAX (\S+) depth (\d+) limit (\d+) max (\d+)(  FULL)?", line)
    if m:
        need[m.group(1)] = int(m.group(4))
        if m.group(5): capped.add(m.group(1))

extra_bits, rows, args = 0, [], []
for name, b in built.items():
    n = need.get(name, 0)
    new = max(b, 2 ** math.ceil(math.log2(max(1, n * margin))))
    if new > b:
        d = (new - b) * width[name]
        extra_bits += d
        short = re.search(r"StreamingFIFO_rtl_\d+", name).group(0)
        rows.append((d, short, b, n, new, width[name], name in capped))
        args.append(f"+fifo:{name}.unnamedblk1={new}")   # the key the module builds
for d, short, b, n, new, w, cap in sorted(rows, reverse=True):
    print(f"{short:24s} built {b:6d} need {n:6d} -> {new:6d}  x{w:3d} b  +{d/1e3:7.1f} Kbit{'  CAPPED (need may be larger)' if cap else ''}")
print(f"\n{len(rows)} of {len(built)} deep FIFOs grow; extra {extra_bits/1e6:.2f} Mbit = {extra_bits/36864:.0f} BRAM36 at full packing")
print("ZCU102 free after the current build: ~221 BRAM36 = ~8.0 Mbit")
open(sys.argv[2].replace(".log", ".plan"), "w").write(" ".join(args))
