#!/bin/bash
set -x
set -e

type="iozone"

bsizes="1k 4k 64k 256k 1m"
ops="randwrite write"
osync="s a"
three_times="1 2 3"

for bsize in $bsizes ; do
	for op in $ops ; do 
		for sync in $osync ; do 
			for xxx in $three_times ; do
				if [[ "$sync" == "s" ]] ; then
					ssync="-s"
					factor="x500"
				else
					if [[ "$bsize" == "1k" || "$bsize" == "4k" ]] ; then
						continue
					fi

					ssync=
					factor="r2"
				fi

				python run_rally_test.py -l -o "--type $type -a $op --iodepth 16 --blocksize $bsize --iosize $factor $ssync" -t io-scenario $type --rally-extra-opts="--deployment perf-2"
			done
		done
	done
done

bsizes="4k 64k 256k 1m"
ops="randread read"

for bsize in $bsizes ; do
	for op in $ops ; do 
		for xxx in $three_times ; do
			python run_rally_test.py -l -o "--type $type -a $op --iodepth 16 --blocksize $bsize --iosize r2" -t io-scenario $type --rally-extra-opts="--deployment perf-2"
		done
	done
done
