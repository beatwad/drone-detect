"""Gate: do the branches entering every join carry the SAME quantisation scale?

This is the precondition for FINN streamlining a branched graph, and it is
decided in QAT — not here, and not fixable at export. FINN's
`MoveLinearPastEltwiseAdd` rewrites

    (x * C) + (y * C)  ->  (x + y) * C

and guards on `np.array_equal(init0, init1)`. `a*x + b*y` does not factor unless
`a == b`, so a checkpoint with independent per-branch scales exports Muls that
never streamline, and the FINN build fails for a reason that has nothing to do
with whether branched topologies are supported.

This script applies exactly that guard, offline, in seconds — so a failing
checkpoint is caught before a multi-hour build rather than during one.

    uv run python export/check_join_scales.py --onnx export/<model>.onnx

Exit 0 = every join is tied. Exit 1 = at least one join would stall streamlining.
"""
import argparse
import sys
from collections import defaultdict

import numpy as np
import onnx
from onnx import numpy_helper

# Nodes that pass a value through unchanged, so a Quant on the far side still
# governs the scale arriving at the join. MaxPool/Resize are monotonic and
# elementwise-in-value; Relu likewise. Anything else ends the walk.
TRANSPARENT = {'MaxPool', 'Resize', 'Upsample', 'Relu', 'Identity'}


def scale_feeding(tensor, producer, inits, depth=0):
    """Walk back from `tensor` to the Quant node that sets its scale."""
    if depth > 8 or tensor not in producer:
        return None, None
    node = producer[tensor]
    if node.op_type == 'Quant':
        s = inits.get(node.input[1])
        return (None if s is None else np.asarray(s).reshape(-1)), node.name
    if node.op_type in TRANSPARENT:
        return scale_feeding(node.input[0], producer, inits, depth + 1)
    return None, f'<{node.op_type}>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    g = onnx.load(args.onnx).graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    producer = {o: n for n in g.node for o in n.output}

    joins = [n for n in g.node if n.op_type in ('Add', 'Concat')]
    if not joins:
        print(f'{args.onnx}: no joins — nothing to check (join-free model)')
        return 0

    bad, untraced = [], []
    by_type = defaultdict(lambda: [0, 0])
    for n in joins:
        scales, names = [], []
        for inp in n.input:
            s, who = scale_feeding(inp, producer, inits)
            scales.append(s)
            names.append(who)
        if any(s is None for s in scales):
            untraced.append((n.name, n.op_type, names))
            by_type[n.op_type][1] += 1
            continue
        tied = all(np.array_equal(scales[0], s) for s in scales[1:])
        by_type[n.op_type][0 if tied else 1] += 1
        if not tied:
            bad.append((n.name, n.op_type, [float(s.reshape(-1)[0]) for s in scales]))
        elif args.verbose:
            print(f'  OK   {n.op_type:<6} {n.name:<44} scale {float(scales[0].reshape(-1)[0]):.6g}')

    print(f'\n{args.onnx}')
    print(f'  joins checked : {len(joins)}')
    for t, (ok, no) in sorted(by_type.items()):
        print(f'    {t:<7} tied {ok:>3} / untied-or-untraced {no:>3}')

    for name, t, s in bad:
        print(f'  UNTIED  {t:<6} {name}\n            scales {s}')
    for name, t, who in untraced:
        print(f'  NO QUANT ON A BRANCH  {t:<6} {name}\n            producers {who}')

    ok = not bad and not untraced
    print(f'\n  verdict : {"all joins tied — streamlining precondition met" if ok else "NOT READY — fix in qat/quantize.py, not at export"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
