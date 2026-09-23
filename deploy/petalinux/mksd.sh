#!/bin/bash
# Write the ZCU102 boot media for the drone detector.
#
#   ./mksd.sh /dev/sdX          show what would happen, change nothing
#   ./mksd.sh /dev/sdX --yes    actually repartition and write
#
# Layout (what petalinux-package --boot expects, and what SW6 = SD boot reads):
#   p1  1 GB  FAT32  BOOT   BOOT.BIN, image.ub, boot.scr
#   p2  rest  ext4   root   rootfs.tar.gz unpacked, + deploy/ in /home/root
#
# Everything this writes lives in deploy/. boot/rootfs.tar.gz is 73 MB, so it is
# gitignored rather than committed -- present in a working tree, absent from a
# fresh clone. Rebuild it from deploy/petalinux/ (boot/drone_v8.xsa is the
# input), or point ROOTFS= at a copy.
#
# The bitstream is deliberately NOT in BOOT.BIN. The FINN driver loads it at
# runtime through pynq.Overlay, so rebuilding the network is a file copy rather
# than a new boot image. resizer.bit rides in deploy/ instead.
set -euo pipefail

DEV="${1:-}"
GO="${2:-}"
HERE=$(dirname "$(readlink -f "$0")")
DEPLOY=$(dirname "$HERE")          # deploy/ -- exactly what lands in /home/root
BOOT="$DEPLOY/boot"
ROOTFS="${ROOTFS:-$BOOT/rootfs.tar.gz}"

[ -b "$DEV" ] || { echo "usage: $0 /dev/sdX [--yes]   (block device required)"; exit 1; }
[ "$(cat /sys/block/"$(basename "$DEV")"/removable 2>/dev/null)" == "1" ] \
    || { echo "REFUSING: $DEV is not a removable disk (pass the whole device, e.g. /dev/sdX)."; exit 1; }
for f in BOOT.BIN image.ub boot.scr; do
    [ -f "$BOOT/$f" ] || { echo "missing $BOOT/$f"; exit 1; }
done
[ -f "$ROOTFS" ] || { echo "missing $ROOTFS -- gitignored, see deploy/README.md; or set ROOTFS="; exit 1; }
for f in run_on_board.py resizer.bit resizer.hwh driver_base.py inputs.npz out_hw.npz; do
    [ -f "$DEPLOY/$f" ] || { echo "missing $DEPLOY/$f"; exit 1; }
done

echo "target : $DEV"
lsblk -o NAME,SIZE,MODEL,MOUNTPOINT "$DEV"
echo
echo "will write:"
ls -la "$BOOT"/{BOOT.BIN,image.ub,boot.scr} "$ROOTFS" | sed 's/^/  /'
echo "  + $DEPLOY (driver, bitstream, golden set, offline PYNQ) -> /home/root/deploy"
echo
[ "$GO" == "--yes" ] || { echo "dry run. re-run with --yes to write. THIS ERASES $DEV."; exit 0; }

read -rp "Erase $DEV completely? type ERASE: " a
[ "$a" == "ERASE" ] || { echo "aborted"; exit 1; }

sudo umount "${DEV}"* 2>/dev/null || true
sudo sgdisk --zap-all "$DEV" || true
sudo parted -s "$DEV" mklabel msdos \
    mkpart primary fat32 1MiB 1025MiB \
    mkpart primary ext4 1025MiB 100% \
    set 1 boot on
sudo mkfs.vfat -F 32 -n BOOT "${DEV}1"
sudo mkfs.ext4 -F -L root "${DEV}2"

M=$(mktemp -d)
sudo mount "${DEV}1" "$M"
sudo cp "$BOOT"/{BOOT.BIN,image.ub,boot.scr} "$M"/
sudo umount "$M"

sudo mount "${DEV}2" "$M"
sudo tar xzf "$ROOTFS" -C "$M"
sudo mkdir -p "$M/home/root/deploy"
# deploy/ minus the two host-side directories -- petalinux/ builds the image,
# boot/ has already gone to p1 -- and minus the PYNQ sdist, which is only there
# to rebuild the wheel. Everything else is board-side by construction.
sudo tar -C "$DEPLOY" --exclude=./petalinux --exclude=./boot --exclude=__pycache__ \
         --exclude=./pynq_offline/pynq-3.0.1.tar.gz \
         -cf - . | sudo tar -C "$M/home/root/deploy" -xf -
sudo sync
sudo umount "$M"
rmdir "$M"

echo
echo "done. Set SW6 [4:1] = off,off,off,on (SD boot), insert, power-cycle."
echo "Console: CP2108 on J83, 115200 8N1, first of the four /dev/ttyUSB*."
