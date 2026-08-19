#!/bin/bash
# Write the ZCU102 boot media for the drone detector.
#
#   ./mksd.sh /dev/sdX          show what would happen, change nothing
#   ./mksd.sh /dev/sdX --yes    actually repartition and write
#
# Layout (what petalinux-package --boot expects, and what SW6 = SD boot reads):
#   p1  1 GB  FAT32  BOOT   BOOT.BIN, image.ub, boot.scr
#   p2  rest  ext4   root   rootfs.tar.gz unpacked
#
# The bitstream is deliberately NOT in BOOT.BIN. The FINN driver loads it at
# runtime through pynq.Overlay, so rebuilding the network is a file copy rather
# than a new boot image. resizer.bit is put on the rootfs instead.
set -euo pipefail

DEV="${1:-}"
GO="${2:-}"
IMG=/home/alex/petalinux/projects/drone/images/linux
PKG=/home/alex/finn_build_mdanilow/drone_v8_bit/deploy

[ -b "$DEV" ] || { echo "usage: $0 /dev/sdX [--yes]   (block device required)"; exit 1; }
case "$DEV" in /dev/sda|/dev/sdb|/dev/sdc|/dev/nvme*) echo "REFUSING: $DEV looks like a system disk."; exit 1;; esac
for f in BOOT.BIN image.ub boot.scr rootfs.tar.gz; do
    [ -f "$IMG/$f" ] || { echo "missing $IMG/$f — run petalinux-package --boot first"; exit 1; }
done

echo "target : $DEV"
lsblk -o NAME,SIZE,MODEL,MOUNTPOINT "$DEV"
echo
echo "will write:"
ls -la "$IMG"/{BOOT.BIN,image.ub,boot.scr,rootfs.tar.gz} | sed 's/^/  /'
echo "  + $PKG (accelerator driver + resizer.bit) -> /home/root/deploy"
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
sudo cp "$IMG"/{BOOT.BIN,image.ub,boot.scr} "$M"/
sudo umount "$M"

sudo mount "${DEV}2" "$M"
sudo tar xzf "$IMG/rootfs.tar.gz" -C "$M"
sudo mkdir -p "$M/home/root/deploy"
sudo cp -r "$PKG"/. "$M/home/root/deploy/"
sudo sync
sudo umount "$M"
rmdir "$M"

echo
echo "done. Set SW6 [4:1] = off,off,off,on (SD boot), insert, power-cycle."
echo "Console: CP2108 on J83, 115200 8N1, first of the four /dev/ttyUSB*."
