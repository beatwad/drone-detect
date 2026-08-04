"""Streamlining transform FINN is missing: move a Mul past a join Concat.

    Concat(x*C, y*C, ...)  ->  Concat(x, y, ...) * C

WHY THIS HAS TO EXIST
  Our exports carry 13 Concat joins. After `step_streamline` converts Quant nodes
  into Mul/Add pairs, each branch entering a Concat is preceded by a Mul. Nothing
  upstream moves it past: `streamline/reorder.py` (v0.10.1) has
  `MoveLinearPastEltwiseAdd` for Add joins and `MoveTransposePastJoinAdd` for
  layout ops, but no Mul-past-Concat. Left in place, those Muls block the
  Mul/Add absorption that streamlining depends on, and the build stalls.

WHY NOT SUBCLASS `MoveIdenticalOpPastJoinOp`
  It looked like a ~10-line subclass with `ops_to_move = ["Mul"]`. It is not.
  Three of its assumptions are false here:
    1. It is hardcoded BINARY -- reads `n.input[0]`/`n.input[1]` and writes both.
       SPPF's Concat has FOUR inputs.
    2. It has no value check, only `prod0.op_type == prod1.op_type`. That is safe
       for Transpose (layout only, its stated assumption) and WRONG for Mul:
       `a*x ⧺ b*y` does not factor unless `a == b`. Guard mirrored from
       `MoveLinearPastEltwiseAdd`, which does exactly this `np.array_equal` check.
    3. Its `move_node` sets the join output shape from an INPUT shape. True for
       Add, false for Concat, which sums along the concat axis.

SCOPE: SCALAR SCALES ONLY, deliberately
  A scalar multiplies every element uniformly, so the rewrite is valid on any
  concat axis. Per-channel scales are rejected.
  Note the generalisation we are NOT implementing: for an axis-1 (channel)
  concat with per-channel scales, `Concat(x*a, y*b) == Concat(x,y) * concat(a,b)`
  -- the scale vectors concatenate, so unequal scales would be fine. That would
  make tying Concat scales in QAT unnecessary (it would still be required for
  Add, which is pure algebra). We keep the narrow version because our exports are
  per-tensor quantized, so the scales are scalars and already tied; implementing
  an untestable generalisation to avoid a constraint we do not have would be
  the wrong trade.

Self-test (no FINN needed, qonnx only):
    uv run python export/finn_transforms.py
"""
import numpy as np
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_shapes import InferShapes


class MoveMulPastJoinConcat(Transformation):
    """Concat(x*C, y*C, ...) -> Concat(x, y, ...) * C, for scalar C, N-ary."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        for node in list(graph.node):
            if node.op_type != "Concat" or not model.is_join_node(node):
                continue

            prods = [model.find_producer(i) for i in node.input]
            if any(p is None or p.op_type != "Mul" for p in prods):
                continue
            # distinct producer nodes: a shared Mul feeding two concat inputs is a
            # fork, and removing it would delete the other branch's operation
            if len({id(p) for p in prods}) != len(prods):
                continue

            # each Mul must have exactly one initializer input (the scale)
            parsed = []
            for p in prods:
                if len(p.input) < 2:
                    break
                init0 = model.get_initializer(p.input[0])
                init1 = model.get_initializer(p.input[1])
                if (init0 is None) == (init1 is None):
                    break                       # both const or neither -> not our pattern
                if init1 is not None:
                    parsed.append((p.input[0], p.input[1], init1, p))
                else:
                    parsed.append((p.input[1], p.input[0], init0, p))
            if len(parsed) != len(prods):
                continue

            scales = [s for _, _, s, _ in parsed]
            if any(s.size != 1 for s in scales):
                continue                        # scalar only -- see module docstring
            if not all(np.array_equal(scales[0], s) for s in scales[1:]):
                continue                        # the MoveLinearPastEltwiseAdd guard

            # every Mul must feed ONLY this Concat, or removing it drops a consumer
            if any(len(model.find_consumers(p.output[0])) != 1 for *_, p in parsed):
                continue

            out = node.output[0]
            mid = model.make_new_valueinfo_name()
            model.set_tensor_shape(mid, model.get_tensor_shape(out))

            for k, (data, _, _, _) in enumerate(parsed):
                node.input[k] = data            # Concat consumes the pre-Mul tensors
            node.output[0] = mid

            keep = parsed[0][3]                 # reuse the first Mul as the moved-out op
            keep.input[0] = mid                 # Mul is commutative, so force data-first
            keep.input[1] = parsed[0][1]
            keep.output[0] = out

            for *_, p in parsed[1:]:
                graph.node.remove(p)

            graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph(), make_deepcopy=False, cleanup=False)
            model = model.transform(InferShapes())
        return (model, graph_modified)


# --------------------------------------------------------------------------- #
# self-test


def _graph(n_branches, scales, axis=1, chans=None):
    """Concat of n Mul'd branches -> a ModelWrapper."""
    from onnx import TensorProto, helper
    from qonnx.core.modelwrapper import ModelWrapper

    chans = chans or [2] * n_branches
    shapes = [[1, chans[i], 4, 4] for i in range(n_branches)]
    # the concat axis sums; every other dim is carried through unchanged
    out_shape = list(shapes[0])
    out_shape[axis] = sum(s[axis] for s in shapes)

    ins, muls, mul_outs, inits = [], [], [], []
    for i in range(n_branches):
        ins.append(helper.make_tensor_value_info(f"x{i}", TensorProto.FLOAT, shapes[i]))
        mul_outs.append(f"m{i}")
        muls.append(helper.make_node("Mul", [f"x{i}", f"c{i}"], [f"m{i}"], name=f"Mul_{i}"))
        inits.append(helper.make_tensor(f"c{i}", TensorProto.FLOAT, [1], [scales[i]]))
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)
    cat = helper.make_node("Concat", mul_outs, ["out"], name="Concat_0", axis=axis)
    g = helper.make_graph(muls + [cat], "t", ins, [out], initializer=inits)
    # pin the opset: make_model defaults to the newest, which onnxruntime refuses
    proto = helper.make_model(g, producer_name="t",
                              opset_imports=[helper.make_opsetid("", 13)])
    return ModelWrapper(proto).transform(InferShapes())


def _run(model, feeds):
    from qonnx.core.onnx_exec import execute_onnx
    return execute_onnx(model, feeds)["out"]


def _selftest():
    np.random.seed(0)
    fails = 0

    def check(name, model, should_fire, n_expect_mul):
        nonlocal fails
        feeds = {i.name: np.random.rand(*[d.dim_value for d in i.type.tensor_type.shape.dim])
                 .astype(np.float32) for i in model.graph.input}
        before = _run(model, feeds)
        new, fired = MoveMulPastJoinConcat().apply(model)
        after = _run(new, feeds)
        n_mul = sum(1 for n in new.graph.node if n.op_type == "Mul")
        same = np.allclose(before, after, atol=1e-6)
        ok = (fired == should_fire) and same and (n_mul == n_expect_mul)
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<42} "
              f"fired={fired} muls={n_mul} numerically_equal={same}")

    print("MoveMulPastJoinConcat self-test")
    check("2 branches, equal scales -> fires", _graph(2, [0.25, 0.25]), True, 1)
    check("4 branches (SPPF shape), equal -> fires", _graph(4, [0.5] * 4), True, 1)
    check("2 branches, DIFFERENT scales -> no-op", _graph(2, [0.25, 0.5]), False, 2)
    check("3 branches, one differs -> no-op", _graph(3, [0.25, 0.25, 0.5]), False, 3)
    check("uneven channels, equal -> fires", _graph(3, [0.125] * 3, chans=[2, 5, 3]), True, 1)
    check("axis=2 concat, equal scalars -> fires", _graph(2, [0.75, 0.75], axis=2), True, 1)
    print("all passed" if not fails else f"{fails} FAILURE(S)")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
