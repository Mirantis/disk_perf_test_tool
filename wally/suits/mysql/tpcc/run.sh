#!/bin/bash
cd ~/tpcc-mysql
./tpcc_start -h127.0.0.1 -dtpcc1000 -uroot -p -w20 -c16 -r10 -l1200 > ~/tpcc-output.log
cat ~/tpcc-output.log | grep  TpmC | grep -o '[0-9,.]\+'