"""Several frames in flight through the accelerator, results in order.

`FINNExampleOverlay.execute()` starts both DMAs and waits for the output: one
frame at a time, so the PL sits idle while the CPU packs, unpacks and decodes,
and the frame rate is bounded by latency + CPU (~34 ms at 187.5 MHz) rather
than by the accelerator's interval (~5.9 ms, build_notes §11.12-11.13).

The two IODMAs are independent HLS cores, each run one transfer per start. The
input DMA goes idle as soon as the network has swallowed a frame — about one
interval — so it can be restarted on the next frame while earlier ones are
still inside; frames leave in order, so the output DMA is restarted on the
oldest frame in flight each time it finishes. Each frame has its own pair of
buffers, K of each.

This is a pipeline, NOT a batch: every result is handed out the moment its
frame leaves the PL, so no frame waits for others. Batching (the driver's
`batch_size`) would hold every result until the last frame of the batch is done.

Registers: the HLS `ap_ctrl_hs` block at 0x00 (bit 0 ap_start, bit 2 ap_idle),
buffer address at 0x10, frame count at 0x1C — the same writes
`execute_on_buffers()` makes. Idle is read as "start accepted and idle", never
from ap_done, which clears on read.
"""
from collections import deque

import numpy as np

AP_START, AP_IDLE = 0x1, 0x4


def _idle(dma):
    r = dma.read(0x00)
    return not (r & AP_START) and bool(r & AP_IDLE)


def _start(dma, buf):
    dma.write(0x10, buf.device_address)
    dma.write(0x1C, 1)
    dma.write(0x00, AP_START)


class AccelPipeline:
    """`submit()` a frame when `can_submit()`, `poll()` for results in order.

    `acc` is a FINNExampleOverlay (zynq-iodma, one input, one output). `tag` is
    anything the caller wants back with the result, e.g. the frame timestamp.
    """

    def __init__(self, acc, depth=3, alloc=None):
        if alloc is None:
            from pynq import allocate
            alloc = lambda shape: allocate(shape=shape, dtype=np.uint8,
                                           cacheable=True, target=acc.device)
        self.acc, self.depth = acc, depth
        self.idma, self.odma = acc.idma[0], acc.odma[0]
        self.ibuf = [alloc(acc.ishape_packed()) for _ in range(depth)]
        self.obuf = [alloc(acc.oshape_packed()) for _ in range(depth)]
        self.out = np.empty(acc.oshape_packed(), np.uint8)
        self.free = deque(range(depth))
        self.fly = deque()                     # (slot, tag), oldest first
        assert _idle(self.idma) and _idle(self.odma), "accelerator DMAs are not idle"

    def in_flight(self):
        return len(self.fly)

    def can_submit(self):
        return bool(self.free) and _idle(self.idma)

    def submit(self, x, tag=None):
        """x: (1, 192, 320, 3) uint8 NHWC, as for execute()."""
        s = self.free.popleft()
        np.copyto(self.ibuf[s], self.acc.pack_input(self.acc.fold_input(x)))
        self.ibuf[s].flush()
        if not self.fly:                       # output DMA always serves fly[0]
            _start(self.odma, self.obuf[s])
        self.fly.append((s, tag))
        _start(self.idma, self.ibuf[s])

    def poll(self):
        """(tag, y) for the oldest frame if it is done, else None. y is the raw
        INT21 NHWC output, exactly what execute() returns."""
        if not self.fly or not _idle(self.odma):
            return None
        s, tag = self.fly.popleft()
        if self.fly:                           # next frame's output may already be queued
            _start(self.odma, self.obuf[self.fly[0][0]])
        self.obuf[s].invalidate()
        np.copyto(self.out, self.obuf[s])
        self.free.append(s)
        y = self.acc.unfold_output(self.acc.unpack_output(self.out))
        return tag, y
