#!/bin/bash
set -x
set -e

type="iozone"

io_opts="--type $type -a write --iodepth 16 --blocksize 1m --iosize x20"
python run_rally_test.py -l -o "$io_opts" -t io-scenario $type --rally-extra-opts="--deployment $1"
