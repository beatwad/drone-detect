#!/bin/bash
# Export the hardware handoff PetaLinux needs, from the Vivado project the
# zynq_drone harness built. -fixed = no reconfigurable partitions, -include_bit
# embeds top_wrapper.bit so the XSA is self-describing.
# NB: the Tcl must live in a real file. With `-source /dev/stdin` and a heredoc,
# Vivado's batch mode consumes stdin itself and sources nothing, exiting 0.
set -e
source /home/alex/Xilinx/Vivado/2022.2/settings64.sh
cd /home/alex/finn_build_mdanilow/zynq_drone
exec vivado -mode batch -nojournal -log export_xsa.log -source export_xsa.tcl
