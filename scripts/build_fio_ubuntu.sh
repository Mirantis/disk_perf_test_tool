#!/usr/bin/env bash
set -xe

apt update
apt -y install g++ git zlib1g-dev libaio-dev librbd-dev make bzip2
cd /tmp
git clone https://github.com/axboe/fio.git
cd fio
./configure
make -j 4
. /etc/lsb-release
# VERSION=$(cat /etc/lsb-release | grep DISTRIB_CODENAME | awk -F= '{print $2}')
chmod a-x fio
bzip2 -z -9 fio
mv fio.bz2 "fio_${DISTRIB_CODENAME}_x86_64.bz2"
