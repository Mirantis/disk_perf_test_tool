#!/bin/bash
sudo apt-get update
sudo apt-get -y install g++ git zlib1g-dev libaio-dev librbd-dev make
git clone https://github.com/axboe/fio.git
cd fio
./configure
make
