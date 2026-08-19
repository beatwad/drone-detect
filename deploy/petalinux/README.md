# PetaLinux image for the ZCU102

Everything needed to rebuild the board's Linux image. The build itself happens
outside the repo (`/home/alex/petalinux/`), because it produces ~80 GB of Yocto
intermediates; only the inputs live here.

Full reasoning, and every trap hit on the way, is in
[build_notes.md](../../.claude/docs/build_notes.md) §11. Read it before changing
anything here — several of these settings look arbitrary and are not.

## Why a container

The host is Ubuntu 26.04 (gcc 15, python 3.12, `/bin/sh -> dash`); PetaLinux
2022.2 is Yocto kirkstone and supports 18.04–22.04. `Dockerfile` gives the tool
a 22.04 userspace. The ~11 GB install and all projects live on a bind mount, so
rebuilding the image costs nothing.

Two packages are not optional: **libtinfo5** (or `xsct` cannot load `hsi`, and
PetaLinux reports only "Failed to generate Kconfig.syshw") and **xvfb**.

## Sequence

```bash
docker build -t petalinux-host:22.04 --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
./install.sh                                    # PetaLinux 2022.2, aarch64 only, ~11 GB
./export_xsa.sh                                 # hardware handoff from the FINN Vivado project

./plnx.sh -w projects 'petalinux-create -t project --template zynqMP -n drone'
./plnx.sh -w projects/drone 'petalinux-config --get-hw-description=<xsa dir> --silentconfig'
./configure.sh /home/alex/petalinux/projects/drone
./plnx.sh -w projects/drone 'petalinux-config --silentconfig && petalinux-config -c rootfs --silentconfig'
./plnx.sh -w projects/drone 'petalinux-build'
./plnx.sh -w projects/drone 'petalinux-package --boot --fsbl --u-boot --pmufw --force'

./mksd.sh /dev/sdX            # dry run
./mksd.sh /dev/sdX --yes      # writes the card
```

`petalinux-*` commands **exit 0 when they fail**. The real error is in
`build/config.log`, never in the console output.

## Verify before trusting the image

Decompile the device tree and check the three things that fail silently on
hardware:

```bash
dtc -I dtb -O dts -o sys.dts images/linux/system.dtb   # dtc is in build/tmp/sysroots-components/
grep -E 'dr_mode|maximum-speed' sys.dts                # "host", "super-speed"
grep -A3 gtr_sel sys.dts                               # sel0 low, sel1..3 high = SEL 1110
```

`BOOT.BIN` should be **~1.8 MB**. If it is ~28 MB the bitstream was baked in —
that is deliberately avoided, since the FINN driver loads `resizer.bit` at
runtime and keeping it out means retraining the network is a file copy.

## Board setup

| | |
|---|---|
| Boot mode | **SW6 [4:1] = off, off, off, on** (SD; default is QSPI32) |
| USB host | **J7 OPEN → ON**, **J110 1-2 → 2-3**; J109/J112/J113 unchanged |
| USB cable | SuperSpeed micro-B → Type-A. A 5-pin USB 2.0 micro OTG adapter fits and silently gives 480 Mbps |
| Console | CP2108 on J83, 115200 8N1, first of four `/dev/ttyUSB*` |
| USB check | `lsusb -t` must say **5000M**, not 480M |
