#!/bin/bash
# set_depths.sh TAG NF [PERIOD] -- idx=depth ...   |   set_depths.sh TAG NF [PERIOD] -f PLANFILE [-x LO-HI]
# Runs design_tb with explicit per-FIFO depths. Builds the plusarg key exactly
# as axis_data_fifo_sim.sv does: %m inside its initial block, i.e.
#   <hier>.StreamingFIFO_rtl_<idx>.fifo.inst.unnamedblk1
# (The first version of this tooling passed '...fifo.inst=' and so silently
# matched nothing -- every per-instance override before 2026-09-24 12:00 was a no-op.)
set -e
HERE=$(dirname "$(readlink -f "$0")")                     # sources: this directory (in the repo)
WORK=${WORK:-/home/alex/finn_build_mdanilow/diag}            # outputs: logs, objects (in the build tree)
mkdir -p $WORK
H=TOP.StreamingDataflowPartition_1_wrapper.StreamingDataflowPartition_1_i.StreamingDataflowPartition_1_StreamingFIFO_rtl
TAG=$1; NF=$2; shift 2
PERIOD=""; [[ "$1" =~ ^[0-9]+$ ]] && { PERIOD=$1; shift; }
ARGS=""
if [ "$1" == "-f" ]; then
    PLAN=$2; shift 2; LO=-1; HI=-1
    [ "$1" == "-x" ] && { LO=${2%-*}; HI=${2#*-}; shift 2; }
    for a in $(cat $PLAN); do
        i=$(echo "$a" | grep -oE "StreamingFIFO_rtl_[0-9]+" | grep -oE "[0-9]+$")
        { [ "$i" -ge "$LO" ] && [ "$i" -le "$HI" ]; } || ARGS="$ARGS $a"
    done
else
    [ "$1" == "--" ] && shift
    for kv in "$@"; do ARGS="$ARGS +fifo:${H}_${kv%%=*}.fifo.inst.unnamedblk1=${kv##*=}"; done
fi
$HERE/design.sh run $NF $PERIOD +fifo_report $ARGS > $WORK/sd_$TAG.log 2>/dev/null
changed=$(grep FIFOMAX $WORK/sd_$TAG.log | awk '{d=$4; l=$6; if (l!=d) c++} END{print c+0}')
echo "$TAG: $changed FIFOs changed; intervals $(grep -E '^frame [2-9]' $WORK/sd_$TAG.log | awk '{print $8}' | tr '\n' ' ')"
