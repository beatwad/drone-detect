#!/bin/bash
# PetaLinux 2022.2, aarch64 sources only (ZynqMP), into a bind-mounted host dir.
# --skip_license is undocumented in --help but parsed (installer header line 76);
# without it the installer stops in `less` on the second EULA and no key sequence
# fed through a pipe will get it out. Same Xilinx terms already accepted for the
# Vivado 2022.2 install on this machine.
exec docker run --rm \
  -v /home/alex/Downloads/Xilinx_2022.2:/installer:ro \
  -v /home/alex/petalinux:/home/alex/petalinux \
  -e TMPDIR=/home/alex/petalinux/tmp \
  -w /home/alex/petalinux \
  petalinux-host:22.04 \
  bash /installer/petalinux-v2022.2-10141622-installer.run \
    --skip_license \
    --dir /home/alex/petalinux/2022.2 \
    --platform "aarch64" \
    --log /home/alex/petalinux/install.log
