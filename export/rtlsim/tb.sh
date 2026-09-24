#!/bin/bash
# tb.sh NODE IN_WORDS OUT_WORDS [W_PERIOD]
# Links tb.cpp against the node's existing pyverilator build and runs it, inside
# the FINN image (verilator headers live there), skipping FINN's entrypoint.
set -e
HERE=$(dirname "$(readlink -f "$0")")                     # sources: this directory (in the repo)
WORK=${WORK:-/home/alex/finn_build_mdanilow/diag}            # outputs: logs, objects (in the build tree)
mkdir -p $WORK
N=$1; shift
D=$(ls -d /home/alex/finn_build_mdanilow/pyverilator_${N}_* | head -1)
[ -d "$D" ] || { echo "no pyverilator build for $N"; exit 1; }
W=""; grep -q "weights_V_TVALID" "$D/V$N.h" && W="-DHAS_WEIGHTS"
docker run --rm --entrypoint bash -v /home/alex/finn_build_mdanilow:/home/alex/finn_build_mdanilow -v /home/alex/Repos:/home/alex/Repos \
  xilinx/finn:v0.10-215-g60ccf026.xrt_202220.2.14.354_22.04-amd64-xrt -c "
  cd $D && I=/usr/local/share/verilator/include &&
  g++ -O2 -std=c++17 -DTB_TOP=V$N $W -I. -I\$I -I\$I/vltstd \
      $HERE/tb.cpp V${N}__ALL.a verilated.o verilated_vcd_c.o \
      -o /tmp/tb_$N && /tmp/tb_$N $*"
