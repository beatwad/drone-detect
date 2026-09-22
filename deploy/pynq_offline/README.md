# Offline PYNQ 3.0.1 for the ZCU102 image

Assembled on the host 2026-09-21 so the board needs no network and no compiler.
`deploy/petalinux/mksd.sh` carries this directory to the board inside `deploy/`,
so on the card it is `/home/root/deploy/pynq_offline` (minus the sdist).

    pynq-3.0.1-py3-none-any.whl   what to install: a pure-Python PYNQ
    pure-python.patch             the one-line change that makes it pure Python
    wheels/                       31 dependencies, cp39 / manylinux2014_aarch64

The upstream sdist is not kept: it is 60 MB, only the wheel is ever installed,
and PyPI has it whenever the wheel needs rebuilding.

## Install

```bash
cd /home/root/deploy/pynq_offline
pip3 install --no-index --find-links wheels pynq-3.0.1-py3-none-any.whl ipython
```

**`ipython` has to be named explicitly.** It is an undeclared dependency:
`pynq.overlay` imports `pynqmetadata.frontends`, which imports
`pynqmetadata.frontends.visualisations`, which imports `IPython.display` at
module scope — and `pynqmetadata` 0.1.2 does not list it. Without it
`import pynq` dies with `ModuleNotFoundError: No module named 'IPython'`, which
looks nothing like a PYNQ problem. Resolved offline: 32 packages, verified with
a `--dry-run` install against the board's platform tags.

**Do not export `BOARD` or `PYNQ_JUPYTER_NOTEBOOKS`.** Setting `BOARD` makes the
install call `download_overlays()`, which fetches from the internet — the one
thing this bundle exists to avoid. Unset, the board overlays and notebooks are
skipped, and we use neither.

## Why a rebuilt wheel rather than the sdist

PYNQ ships source-only; there is no aarch64 wheel on PyPI. Installing from the
sdist on the board cannot work, because on aarch64 `setup.py` runs `make` on
five C libraries — `libdisplayport`, `libxhdmi`, `libaudio`, `libiic`,
`libpcam5c` — and `rootfs.manifest` has `libgcc1` and nothing else. No gcc, no
make, no python3-dev.

But none of that native code is ours. Read `setup.py`: `ext_modules` is
non-empty **only on armv7l** (Zynq-7000's `pynq.lib._video`); on aarch64 it is
already `[]`, and the five libraries are HDMI, DisplayPort, audio, IIC and PCam
drivers that live in `pynq/lib/`. `pynq/__init__.py` never imports `pynq.lib`,
and `deploy/driver_base.py` uses exactly `Overlay`, `allocate` and `ps.Clocks`.
The only modules on that path that open a shared library at all are
`pynq.buffer` and `pynq.pl_server.xrt_device`, and both want `libc.so.6`; XRT
itself is reached through `pynq/_3rdparty/xrt.py`, a vendored ctypes binding,
against the `xrt` package already in the rootfs.

So the native code is dead weight for us, and PYNQ is pure Python once
`ext_modules` is empty — which it is on any non-armv7l build host. The only
thing forcing a platform-tagged wheel was `BinaryDistribution.has_ext_modules`
returning `True` unconditionally. `pure-python.patch` returns `False`; the
result is `py3-none-any`, built on this x86 host and installable on the board
with no toolchain.

To rebuild it:

```bash
pip download pynq==3.0.1 --no-deps --no-binary :all:
tar xzf pynq-3.0.1.tar.gz && cd pynq-3.0.1
patch -p1 < ../pure-python.patch
uv run --no-project --python 3.11 --with setuptools --with wheel \
    python setup.py bdist_wheel        # python 3.12 has no distutils; setup.py needs it
```

Verified on the host under python 3.9 (the board's version): the wheel contains
no `.so`, `import pynq` succeeds, and `Overlay`, `allocate` and `Clocks` all
import. It ends at `No devices found, is the XRT environment sourced?` with
`Device.devices == []`, which is correct on a machine with no XRT — and is
exactly the symptom to expect on the board if the zocl node or CMA is missing.
Note it *warns* rather than raises: the failure would surface one line later, as
an `IndexError` on `Device.devices[0]` in `run_on_board.py`.

## What is still untested

That `Device.devices` is non-empty on the real board. That needs the image built
with the zocl node and `cma=512M` (commit `3146608`), XRT up, and the board
present. Everything upstream of it is now checked.

The tradeoff accepted here: this is a modified PYNQ. Anything that later wants
`pynq.lib.video`, `pynq.lib.audio` or the PCam driver will fail at import, and
would need the toolchain route instead — `petalinux-config -c rootfs` → Image
Features → `tools-sdk`, then the sdist.
