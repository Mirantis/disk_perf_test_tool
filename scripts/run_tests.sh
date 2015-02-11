#!/bin/bash

set -x
set -e

CMD1="--type iozone -a write --iodepth 8 --blocksize 4k --iosize 40M -s"
CMD2="--type fio -a write --iodepth 8 --blocksize 4k --iosize 4M -s"

python run_rally_test.py -l -o "$CMD1" -t io-scenario iozone 2>&1 | tee ceph_results.txt
python run_rally_test.py -l -o "$CMD2" -t io-scenario fio 2>&1 | tee -a ceph_results.txt

