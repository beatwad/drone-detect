# PYNQ on the ZCU102

**Question:** phase 9 plans to `pip install pynq` over the PetaLinux-built XRT,
on the premise that no official PYNQ image exists for the ZCU102. Is that
premise true, and is the plan the right one?

Researched 2026-08-26. Sources are primary (pynq.io, PYNQ maintainers on
discuss.pynq.io, Xilinx Vitis tutorials) except where marked third-party.

## 1. The premise holds: there is no official ZCU102 image

[pynq.io/boards.html](http://www.pynq.io/boards.html) lists every board with a
downloadable image. As of PYNQ v3.1.1: PYNQ-Z1, PYNQ-Z2, PYNQ-ZU, ZCU104,
AUP-ZU3, Kria KV260/KR260, RFSoC 4x2, RFSoC 2x2, ZCU208, ZCU111, Ultra96/V2,
ZUBoard 1CG, TySOM-3-ZU7EV, TySOM-3A-ZU19EG. **ZCU102 is not among them**, and
it is the only ZynqMP devkit in that class that is missing.

Maintainers confirm it on
[discuss.pynq.io/t/pynq-image-for-zcu102/2993](https://discuss.pynq.io/t/pynq-image-for-zcu102/2993)
(thread runs Sept 2021 → Oct 2024). Two statements matter:

- The recommended path is to build an image yourself from a BSP plus the
  **board-agnostic Ubuntu rootfs** published on the PYNQ boards page.
- *"We do not have a process in place to enable PYNQ on the official Canonical
  Ubuntu images"* — even though Canonical does officially support the ZCU102.

As of the last posts in that thread, **multiple users report never getting a
working image**. So the gap is real and long-standing, not an oversight.

## 2. `pip install pynq` on a foreign rootfs is untested, not impossible

[discuss.pynq.io/t/import-pynq-package-on-non-pynq-image/914](https://discuss.pynq.io/t/import-pynq-package-on-non-pynq-image/914)
is the closest thing to our plan. Maintainer guidance: install from the source
distribution, `pip install <pynq_sdist>.tar.gz --upgrade --no-deps`, with the
caveat *"This flow has not been tested on other rootfs so proceed with your own
risk."* The thread predates PYNQ 3.x — it talks about `xlnk` and
`CONFIG_XILINX_APF`, both of which PYNQ dropped when buffer allocation moved to
XRT in 2.7. The transferable part is the shape of the failure: when the kernel
side is missing, the symptom is **`No Devices Found` at import**, not a build
error.

## 3. A third-party ZCU102 recipe exists and works

[github.com/steltze/ZCU102-PYNQ](https://github.com/steltze/ZCU102-PYNQ)
(third-party, not Xilinx) installs **PYNQ 3.0.1** onto **Canonical Ubuntu 22.04
LTS for Zynq UltraScale+**, modelled on Xilinx's own
[Kria-PYNQ](https://github.com/Xilinx/Kria-PYNQ) installer. `sudo bash
install.sh`, ~25 minutes; it installs debian packages, builds a venv, and
configures a Jupyter portal. Ships a DPU overlay we do not need.

This does not mean we should switch to Ubuntu — our PetaLinux image carries
board work that a stock image would not have (the PS-GTR mux hogs, USB dual-role,
UVC). But it proves PYNQ **does** run on this silicon, and it fixes the version
to target.

## 4. Version pin: PYNQ 3.0.1, not 3.1

[PYNQ SD card docs](https://pynq.readthedocs.io/en/latest/pynq_sd_card.html)
state v3.1 **requires Xilinx tools 2024.1**. Our toolchain is pinned at 2022.2
(see build_notes §7), so 3.1 is the wrong target. PYNQ 3.0.1 is the 2022.x-era
release and is what the ZCU102 recipe above uses.

## 5. Two concrete gaps in our own image — both blocking, both silent

Found by decompiling the built `images/linux/system.dtb` on 2026-08-26.

### 5.1 There is no zocl device-tree node

`grep -i zocl` on the decompiled tree: **nothing**. Yet
`deploy/petalinux/configure.sh` does enable the `zocl` rootfs package, so the
module will be present on the card and simply never probe.

This matters because **PYNQ 3.x allocates buffers through XRT**, and XRT on
embedded ZynqMP requires zocl. The
[Vitis platform tutorial](https://xilinx.github.io/Vitis-Tutorials/2022-1/build/html/docs/Vitis_Platform_Creation/Feature_Tutorials/02_petalinux_customization/README.html)
puts it plainly: *"ZOCL driver module has no associated hardware, but it's
required by XRT."*

Why it is missing is the interesting part. That same tutorial notes PetaLinux
adds the node automatically **only if the XSA is a Vitis extensible platform
project**. Ours is not — `deploy/petalinux/export_xsa.tcl` exports a plain XSA
from the FINN Vivado project. So the node has to be written by hand into
`system-user.dtsi`:

```dts
&amba {
	zyxclmm_drm {
		compatible = "xlnx,zocl";
		status = "okay";
		reg = <0x0 0xA0000000 0x0 0x10000>;
	};
};
```

Expected symptom if left out: `/dev/dri/renderD128` never appears and importing
`pynq` fails with `No Devices Found` — the same signature as §2.

### 5.2 CMA is not sized

Decompiled bootargs are:

```
earlycon console=ttyPS0,115200 clk_ignore_unused root=/dev/mmcblk0p2 rw rootwait
```

No `cma=`, and no `reserved-memory` node anywhere in the tree, so the kernel
default applies. Xilinx's guidance for the acceleration flow is `cma=512M`,
because *"CMA is used to exchange data between PS and PL kernel."* Our accelerator
takes a 184 KB input and returns a 164 KB output per frame, so 512 MB is far more
than needed — but the default is small enough to be worth setting explicitly
rather than discovering under load.

Set it in PetaLinux config → DTG settings → Kernel Bootargs, or in
`system-user.dtsi`.

## 6. Conclusion

The plan is sound and the premise behind it is correct, but it was one step short:
`pip install pynq` cannot succeed on the image as currently built, because the
kernel side XRT depends on is absent. Fix §5.1 and §5.2, rebuild, then install
PYNQ 3.0.1 from its sdist.

Fallbacks, in order of preference, if that still fails on hardware:

1. Follow the ZCU102-PYNQ recipe on Canonical Ubuntu 22.04 and re-apply our
   device-tree work (GTR hogs, USB host, UVC) on top.
2. Build a proper PYNQ image via `sdbuild` with a ZCU102 board spec and our BSP —
   the path maintainers recommend, and the one users in §1 kept failing at.
3. Drop PYNQ: FINN's `driver_base.py` needs only bitstream loading and a CMA
   allocator. Both are reachable through `pyxrt` or the FPGA manager sysfs
   interface directly. More work, but no dependency on an unsupported board.
