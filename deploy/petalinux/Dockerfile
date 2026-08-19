# PetaLinux 2022.2 host environment.
#
# The real host is Ubuntu 26.04 (gcc 15, python 3.12, /bin/sh -> dash). PetaLinux
# 2022.2 is Yocto kirkstone and supports 18.04/20.04/22.04: its recipes do not
# compile under gcc 13+, and bitbake 2.0 predates python 3.12. So the tool gets
# a 22.04 userspace and the host keeps its own.
#
# The tool itself is NOT installed into the image — it goes to a bind-mounted
# host directory, so rebuilding this image costs nothing and the ~40 GB install
# survives.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN dpkg --add-architecture i386 && apt-get update && apt-get install -y \
      iproute2 gawk python3 build-essential gcc git make net-tools \
      libncurses5-dev tftpd zlib1g:i386 libssl-dev flex bison libselinux1 \
      gnupg wget diffstat chrpath socat xterm autoconf libtool tar unzip \
      texinfo zlib1g-dev gcc-multilib automake screen pax gzip cpio \
      python3-pip python3-pexpect xz-utils debianutils iputils-ping \
      python3-git python3-jinja2 libegl1-mesa libsdl1.2-dev pylint \
      rsync file locales sudo vim less bc lz4 zstd \
 && rm -rf /var/lib/apt/lists/*

# xsct (which petalinux-config uses to parse the XSA) links against ncurses 5,
# gone from Ubuntu 22.04. It also insists on an X display, hence Xvfb. Without it `package require hsi` fails and PetaLinux
# reports only "Failed to generate Kconfig.syshw" — and still exits 0. Same trap
# the host hit with Vivado (CLAUDE.md). libtinfo6 is ABI-compatible for this use.
RUN apt-get update \
 && (apt-get install -y libtinfo5 \
     || ln -sf /lib/x86_64-linux-gnu/libtinfo.so.6 /lib/x86_64-linux-gnu/libtinfo.so.5) \
 && (apt-get install -y libncurses5 || true) \
 && apt-get install -y xvfb libxrender1 libxtst6 libxi6 \
 && rm -rf /var/lib/apt/lists/*

# PetaLinux checks for bash, and Yocto's shell assumptions are bash-flavoured.
RUN echo "dash dash/sh boolean false" | debconf-set-selections \
 && dpkg-reconfigure -f noninteractive dash

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Same uid/gid as the host user: the install and the build land on a bind mount,
# and Yocto refuses to run as root anyway.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} alex || true \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash alex \
 && echo "alex ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/alex
USER alex
WORKDIR /home/alex
