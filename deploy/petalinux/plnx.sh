#!/bin/bash
# Run a command inside the PetaLinux 2022.2 environment.
#   ./plnx.sh 'petalinux-build'                  (from projects/drone)
#   ./plnx.sh -w projects/drone 'petalinux-build'
# The tool lives on a bind mount, not in the image; see docker/Dockerfile.
WORK=/home/alex/petalinux
SUB=""
[ "$1" == "-w" ] && { SUB="$2"; shift 2; }
exec docker run --rm -i \
  -v /home/alex/petalinux:/home/alex/petalinux \
  -e TMPDIR=/home/alex/petalinux/tmp \
  -w "$WORK/$SUB" \
  petalinux-host:22.04 \
  bash -lc "source $WORK/2022.2/settings.sh >/dev/null && $*"
