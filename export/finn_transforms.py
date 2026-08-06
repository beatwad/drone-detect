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
from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_shapes import InferShapes


class InputQuantToUintDtype(Transformation):
    """Unsigned Quant on the graph input -> UINT<n> input tensor + a Mul by its scale.

    WHY
      `qat/quantize.py` puts `input_quant=Uint8ActPerTensorFloat` on the first
      conv, so the export starts with a Quant node fed by nothing. FINN routes
      any Quant whose predecessor is None to `QuantIdentityHandler`
      (`valid_predecessor_op_types` includes None), and that handler hard-fails:

          ValueError: FINN only supports signed Quant nodes for identity
                      activations.

      Signing it would be wrong -- the tensor really is unsigned pixel data. The
      FINN idiom is to carry that information as the input tensor's DATATYPE and
      leave only the dequantising scale in the graph, which streamlining then
      absorbs into the first layer's thresholds. That is what this does:

          x_float -> Quant(s, unsigned, zp=0)  =>  x_uint8 -> Mul(s)

    EQUIVALENCE, and the one thing it changes
      Quant(x) = clip(round(x/s), 0, 2^n - 1) * s. Feeding the integer code
      directly and multiplying by s is the same value, PROVIDED the host now
      feeds `round(x/s)` rather than x. So the accelerator's input becomes raw
      integer codes -- better for deployment (no float preprocessing on the ARM
      side), but it IS a change to the input contract. Our s is 0.0039158 vs
      1/255 = 0.0039216, so the code differs from the raw 0..255 pixel by at
      most 1 LSB.

    Guards: only fires on the graph input, only if unsigned, zero-point 0, and a
    scalar scale. Anything else is left alone for FINN to complain about.
    """

    def apply(self, model):
        graph = model.graph
        inp = graph.input[0].name
        cons = model.find_consumers(inp)
        if len(cons) != 1 or cons[0].op_type != "Quant":
            return (model, False)
        q = cons[0]

        signed = next((a.i for a in q.attribute if a.name == "signed"), None)
        if signed != 0:
            return (model, False)          # signed identity Quant is FINN's own supported case

        zp = model.get_initializer(q.input[2])
        scale = model.get_initializer(q.input[1])
        bitwidth = model.get_initializer(q.input[3])
        if zp is None or scale is None or bitwidth is None:
            return (model, False)
        if np.any(zp != 0) or scale.size != 1:
            return (model, False)          # only the plain per-tensor, zero-point-0 case

        dt = DataType["UINT%d" % int(np.asarray(bitwidth).reshape(-1)[0])]

        # the Quant becomes a plain dequantising Mul, reusing its own scale initializer
        mul = helper.make_node("Mul", [inp, q.input[1]], [q.output[0]],
                               name="InputDequant_Mul")
        graph.node.insert(list(graph.node).index(q), mul)
        graph.node.remove(q)
        model.set_tensor_datatype(inp, dt)

        model = model.transform(InferShapes())
        return (model, True)


class ConcatToNHWC(Transformation):
    """NCHW channel-Concat -> Transpose-in, Concat on last axis, Transpose-out.

    WHY
      `InferConcatLayer` only converts Concat "operating on last/-1 axis" --
      FINN's StreamingConcat is channel-last. Ours are `axis=1` on 4D NCHW, so
      all 13 are skipped and the dataflow block comes out non-contiguous.

    WHY NOT MoveTransposePastJoinConcat
      It fixed only 4 of 13 (measured). It subclasses `MoveIdenticalOpPastJoinOp`,
      which requires EVERY producer to be the same op -- but our joins are
      asymmetric: `fed_by=['MultiThreshold', 'Transpose[0,3,1,2]']`. A branch
      ending in a conv carries conv-lowering's trailing transpose; a branch ending
      in a pool or another join does not.

    This rewrite does not care what feeds the branches. It is a pure identity:
    NCHW -> NHWC on each input, concat on the channel axis (now last), NHWC ->
    NCHW on the output. The inserted input transposes then cancel against any
    existing trailing `Transpose[0,3,1,2]` via AbsorbConsecutiveTransposes, and
    the output transpose is absorbed into the following MultiThreshold -- so on
    the conv-fed branches nothing is actually added.
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        for node in list(graph.node):
            if node.op_type != "Concat":
                continue
            axis = next((a.i for a in node.attribute if a.name == "axis"), None)
            ishape = model.get_tensor_shape(node.input[0])
            if axis is None or ishape is None or len(ishape) != 4:
                continue
            if axis in (-1, len(ishape) - 1):
                continue                        # already channel-last
            if axis != 1:
                continue                        # only the NCHW channel case

            idx = list(graph.node).index(node)

            # NCHW -> NHWC on every input
            for k, inp in enumerate(node.input):
                sh = model.get_tensor_shape(inp)
                mid = model.make_new_valueinfo_name()
                model.set_tensor_shape(mid, [sh[0], sh[2], sh[3], sh[1]])
                model.set_tensor_datatype(mid, model.get_tensor_datatype(inp))
                graph.node.insert(idx, helper.make_node(
                    "Transpose", [inp], [mid], perm=[0, 2, 3, 1]))
                idx += 1
                node.input[k] = mid

            # concat on the (now last) channel axis
            out = node.output[0]
            osh = model.get_tensor_shape(out)
            mid_out = model.make_new_valueinfo_name()
            model.set_tensor_shape(mid_out, [osh[0], osh[2], osh[3], osh[1]])
            model.set_tensor_datatype(mid_out, model.get_tensor_datatype(out))
            for a in node.attribute:
                if a.name == "axis":
                    a.i = 3
            node.output[0] = mid_out

            # NHWC -> NCHW on the output, so downstream is untouched
            graph.node.insert(idx + 1, helper.make_node(
                "Transpose", [mid_out], [out], perm=[0, 3, 1, 2]))
            graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph(), make_deepcopy=False, cleanup=False)
            model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveScalarMulPastIm2Col(Transformation):
    """Mul(x, C) -> Im2Col   =>   Im2Col -> Mul(x, C), for scalar C.

    Im2Col is a pure gather (and zero-pads), so a scalar commutes with it exactly:
    s*0 == 0, and every output element is a copy of an input element.

    WHY IT IS NEEDED: moving a Mul past a Concat parks it in front of the next
    conv's Im2Col, and `MoveScalarLinearPastInvariants.SUPPORTED_INVARIANTS` does
    not list Im2Col -- so it strands there and keeps the tensor float
    ("Im2Col_N : Input is not int. Can't infer ConvInpGen"). Once past Im2Col,
    Streamline's own MoveScalarMulPastMatMul + AbsorbMulIntoMultiThreshold finish
    the job.
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        for node in list(graph.node):
            if node.op_type != "Im2Col":
                continue
            mul = model.find_producer(node.input[0])
            if mul is None or mul.op_type != "Mul":
                continue
            if len(model.find_consumers(mul.output[0])) != 1:
                continue                        # fork: leave it to MoveLinearPastFork
            scale, data = None, None
            for a, b in ((0, 1), (1, 0)):
                if model.get_initializer(mul.input[a]) is not None:
                    scale, data = mul.input[a], mul.input[b]
            if scale is None or model.get_initializer(scale).size != 1:
                continue                        # scalar only

            im2col_out = node.output[0]
            mid = model.make_new_valueinfo_name()
            model.set_tensor_shape(mid, model.get_tensor_shape(im2col_out))

            node.input[0] = data                # Im2Col consumes the pre-Mul tensor
            node.output[0] = mid
            mul.input[0], mul.input[1] = mid, scale
            mul.output[0] = im2col_out
            graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph(), make_deepcopy=False, cleanup=False)
            model = model.transform(InferShapes())
        return (model, graph_modified)


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


def _quant_graph(signed=0, bitwidth=8, scale=0.0039158, on_input=True):
    """x -> Quant -> out, or Relu -> Quant -> out when on_input=False."""
    from onnx import TensorProto
    from qonnx.core.modelwrapper import ModelWrapper

    shape = [1, 3, 4, 4]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)
    inits = [
        helper.make_tensor("s", TensorProto.FLOAT, [1], [scale]),
        helper.make_tensor("zp", TensorProto.FLOAT, [1], [0.0]),
        helper.make_tensor("bw", TensorProto.FLOAT, [1], [float(bitwidth)]),
    ]
    qin = "x" if on_input else "r"
    nodes = [] if on_input else [helper.make_node("Relu", ["x"], ["r"])]
    nodes.append(helper.make_node(
        "Quant", [qin, "s", "zp", "bw"], ["out"], name="Quant_0",
        domain="qonnx.custom_op.general", signed=signed, narrow=0,
        rounding_mode="ROUND"))
    g = helper.make_graph(nodes, "t", [x], [out], initializer=inits)
    proto = helper.make_model(g, producer_name="t",
                              opset_imports=[helper.make_opsetid("", 13)])
    return ModelWrapper(proto).transform(InferShapes())


def _selftest_input_quant():
    from qonnx.core.onnx_exec import execute_onnx
    np.random.seed(1)
    fails = 0
    s = 0.0039158

    def check(name, cond):
        nonlocal fails
        fails += not cond
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    # fires: unsigned Quant on the graph input
    m = _quant_graph(signed=0)
    x = np.random.rand(1, 3, 4, 4).astype(np.float32)
    before = execute_onnx(m, {"x": x})["out"]
    new, fired = InputQuantToUintDtype().apply(m)
    ops = [n.op_type for n in new.graph.node]
    # the host now feeds integer CODES, so replay with round(x/s)
    codes = np.clip(np.round(x / s), 0, 255).astype(np.float32)
    after = execute_onnx(new, {"x": codes})["out"]
    check("unsigned input Quant -> fires", fired)
    check("Quant replaced by Mul", ops == ["Mul"])
    check("input tensor annotated UINT8",
          new.get_tensor_datatype("x") == DataType["UINT8"])
    check("numerically equal when fed integer codes",
          np.allclose(before, after, atol=1e-6))

    # must not fire
    m2 = _quant_graph(signed=1)
    _, fired2 = InputQuantToUintDtype().apply(m2)
    check("signed input Quant -> no-op", not fired2)

    m3 = _quant_graph(signed=0, on_input=False)
    _, fired3 = InputQuantToUintDtype().apply(m3)
    check("Quant not on graph input -> no-op", not fired3)

    # bit width is read, not assumed
    m4 = _quant_graph(signed=0, bitwidth=4)
    new4, fired4 = InputQuantToUintDtype().apply(m4)
    check("4-bit input Quant -> UINT4",
          fired4 and new4.get_tensor_datatype("x") == DataType["UINT4"])
    return fails


def _selftest_concat_nhwc():
    """The rewrite must be a pure identity, and must land the concat on axis 3."""
    from onnx import TensorProto, helper as h
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx

    np.random.seed(2)
    fails = 0

    def check(name, cond):
        nonlocal fails
        fails += not cond
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    # NCHW channel-concat of 2 branches with different channel counts
    a = h.make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 5, 4])
    b = h.make_tensor_value_info("b", TensorProto.FLOAT, [1, 2, 5, 4])
    out = h.make_tensor_value_info("out", TensorProto.FLOAT, [1, 5, 5, 4])
    g = h.make_graph([h.make_node("Concat", ["a", "b"], ["out"], axis=1)],
                     "t", [a, b], [out])
    m = ModelWrapper(h.make_model(g, opset_imports=[h.make_opsetid("", 13)]))
    m = m.transform(InferShapes())
    feeds = {"a": np.random.rand(1, 3, 5, 4).astype(np.float32),
             "b": np.random.rand(1, 2, 5, 4).astype(np.float32)}
    before = execute_onnx(m, feeds)["out"]
    new, fired = ConcatToNHWC().apply(m)
    after = execute_onnx(new, feeds)["out"]
    cat = [n for n in new.graph.node if n.op_type == "Concat"][0]
    axis = next(x.i for x in cat.attribute if x.name == "axis")
    check("fires on NCHW axis-1 concat", fired)
    check("concat now on last axis (3)", axis == 3)
    check("output identical (pure identity)", np.allclose(before, after, atol=1e-6))
    check("output shape preserved", after.shape == before.shape == (1, 5, 5, 4))

    # must not fire twice / on an already channel-last concat
    _, fired2 = ConcatToNHWC().apply(new)
    check("idempotent — no-op on channel-last concat", not fired2)
    return fails


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
    print("\nInputQuantToUintDtype self-test")
    fails += _selftest_input_quant()
    print("\nConcatToNHWC self-test")
    fails += _selftest_concat_nhwc()
    print("\nall passed" if not fails else f"\n{fails} FAILURE(S)")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
