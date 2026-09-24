#!/bin/bash
# layer.sh NODE IN_WORDS OUT_WORDS
# Verilates memstream + the node's MVU wrapper as one layer (layer_top.sv) and
# times one frame with tb.cpp. Everything is read from the node's own build.
set -e
HERE=$(dirname "$(readlink -f "$0")")                     # sources: this directory (in the repo)
WORK=${WORK:-/home/alex/finn_build_mdanilow/diag}            # outputs: logs, objects (in the build tree)
mkdir -p $WORK
N=$1; IN=$2; OUT=$3
B=/home/alex/finn_build_mdanilow
G=$(ls -d $B/code_gen_ipgen_${N}_* | head -1)
P=$(ls -d $B/pyverilator_${N}_* | head -1)
SRCS=$(grep -oE "/[^ ]+\.s?v" $P/V${N}__verFiles.dat | sort -u | tr '\n' ' ')
MS=/home/alex/Repos/finn-mdanilow/finn-rtllib/memstream/hdl
DEPTH=$(wc -l < $G/memblock.dat)
WIDTH=$(( $(head -1 $G/memblock.dat | tr -d '\n\r' | wc -c) * 4 ))   # hex chars x 4
IN_W=$(grep -oE "VL_IN[0-9W]*\(&in0_V_TDATA,[0-9]+" $P/V$N.h | grep -oE "[0-9]+$"); IN_W=$((IN_W+1))
OUT_W=$(grep -oE "VL_OUT[0-9W]*\(&out_V_TDATA,[0-9]+" $P/V$N.h | grep -oE "[0-9]+$"); OUT_W=$((OUT_W+1))
W=/tmp/layer_$N
echo "$N: memstream DEPTH=$DEPTH WIDTH=$WIDTH, in $IN_W b, out $OUT_W b"
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e CCACHE_DIR=/tmp/ccache --entrypoint bash -v $B:$B -v /home/alex/Repos:/home/alex/Repos \
  xilinx/finn:v0.10-215-g60ccf026.xrt_202220.2.14.354_22.04-amd64-xrt -c "
  set -e; mkdir -p $W && cd $W
  verilator --cc --exe --build -O3 -j 8 -Wno-fatal -Wno-lint -Wno-style -Wno-MULTIDRIVEN \
    --top-module layer_top -GDEPTH=$DEPTH -GWIDTH=$WIDTH -GINIT_FILE='\"$G/memblock.dat\"' \
    -GIN_W=$IN_W -GOUT_W=$OUT_W -DLAYER=$N \
    -I$MS -I/home/alex/Repos/finn-mdanilow/finn-rtllib/mvu \
    $HERE/layer_top.sv $MS/memstream_axi_wrapper.v $MS/memstream_axi.sv $MS/memstream.sv \
    $MS/axilite_if.v $MS/Q_srl.v $SRCS $HERE/tb.cpp \
    -CFLAGS '-DTB_TOP=Vlayer_top' -o tb >/dev/null
  ./obj_dir/tb $IN $OUT"
