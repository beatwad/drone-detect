#!/bin/bash
# design.sh build          verilate the stitched drone accelerator (once, slow)
# design.sh run [args]     run design_tb with args (NF frames, +plusargs)
# Sources: the stitched project's own list, with the Xilinx axis_data_fifo IP
# swapped for axis_data_fifo_sim.sv. Runs in the FINN image (it has verilator).
set -e
B=/home/alex/finn_build_mdanilow
S=$B/vivado_stitch_proj_a5f6s728
HERE=$(dirname "$(readlink -f "$0")")                     # sources: this directory (in the repo)
WORK=${WORK:-/home/alex/finn_build_mdanilow/diag}            # outputs: logs, objects (in the build tree)
mkdir -p $WORK
O=$WORK/design_obj
IMG=xilinx/finn:v0.10-215-g60ccf026.xrt_202220.2.14.354_22.04-amd64-xrt
DOCKER="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e CCACHE_DIR=/tmp/ccache --entrypoint bash -v $B:$B -v /home/alex/Repos:/home/alex/Repos -v /home/alex/Xilinx:/home/alex/Xilinx:ro $IMG -c"
case "$1" in
build)
    grep -E "\.s?v$" $S/all_verilog_srcs.txt \
      | grep -vE "axis_data_fifo_v2_0_vl_rfs\.v|axis_infrastructure_v1_1_vl_rfs\.v" > $WORK/srcs.all
    # 28 identical copies of swg_pkg.sv, one per SWG IP: keep one, and first --
    # Verilator needs a package declared once and before its users
    { grep -m1 "swg_pkg.sv$" $WORK/srcs.all; grep -v "swg_pkg.sv$" $WORK/srcs.all; } > $WORK/srcs.txt
    echo $HERE/axis_data_fifo_sim.sv >> $WORK/srcs.txt
    INC=$(grep -E "\.vh$" $S/all_verilog_srcs.txt | xargs -n1 dirname | sort -u | sed 's/^/-I/' | tr '\n' ' ')
    echo "$(wc -l < $WORK/srcs.txt) sources"
    $DOCKER "set -e; mkdir -p $O; cd $O
      verilator --cc --exe --build -O3 -j 12 --x-assign 0 --x-initial 0 \
        -Wno-fatal -Wno-lint -Wno-style -Wno-MULTIDRIVEN -Wno-PINMISSING -Wno-TIMESCALEMOD \
        --top-module StreamingDataflowPartition_1_wrapper $INC \
        -f $WORK/srcs.txt $HERE/design_tb.cpp --Mdir $O -o design_tb 2>&1 | grep -E '%Error|error:' | head -40
      ls -la $O/design_tb"
    ;;
run)
    shift
    $DOCKER "cd $O && ./design_tb $*"
    ;;
*) echo "usage: $0 build | run [NF] [+plusargs]"; exit 2 ;;
esac
