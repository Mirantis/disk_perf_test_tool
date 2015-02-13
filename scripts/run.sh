#!/bin/bash
set -e

type="iozone"

bsizes="1k" # 4k 64k 256k 1m
ops="write" # randwrite"
osync="s" # a
num_times="1"
concurrences="1 2 4 8 16 32 64 128"
aff_group="0077d59c-bf5b-4326-8940-027e77d655ee"

for concurrence in $concurrences; do
	for bsize in $bsizes ; do
		for op in $ops ; do 
			for sync in $osync ; do 
				for counter in $(seq 1 $num_times) ; do
					# if [[ "$ops" == "write" && "$osync" == "s" ]] ; then
					# 	continue
					# fi

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

					extra_opts="user=ubuntu,keypair_name=ceph,img_name=ubuntu,flavor_name=ceph.512"
					extra_opts="${extra_opts},network_zone_name=net04,flt_ip_pool=net04_ext,key_file=ceph.pem"
					extra_opts="${extra_opts},aff_group=${aff_group}"

					io_opts="--type $type -a $op --iodepth 16 --blocksize $bsize --iosize $factor $ssync --concurrency $concurrence"

					echo $io_opts
					# python run_test.py --runner ssh -l -o "$io_opts" -t io-scenario $type --runner-extra-opts="$extra_opts"

					# io_opts="--type $type -a $op --iodepth 16 --blocksize $bsize --iosize $factor $ssync"
					# python run_test.py --runner rally -l -o "$io_opts" -t io-scenario $type --runner-extra-opts="--deployment $1"
				done
			done
		done
	done
done

# bsizes="4k 64k 256k 1m"
# ops="randread read"

# for bsize in $bsizes ; do
# 	for op in $ops ; do 
# 		for xxx in $three_times ; do
# 			io_opts="--type $type -a $op --iodepth 16 --blocksize $bsize --iosize r2"
# 			python run_rally_test.py -l -o "$io_opts" -t io-scenario $type --rally-extra-opts="--deployment $1"
# 		done
# 	done
# done

# bsizes="1k 4k"
# ops="randwrite write"
# three_times="1 2 3"

# for bsize in $bsizes ; do
# 	for op in $ops ; do 
# 		for xxx in $three_times ; do
# 			factor="r2"
# 			io_opts="--type $type -a $op --iodepth 16 --blocksize $bsize --iosize $factor"
# 			python run_rally_test.py -l -o "$io_opts" -t io-scenario $type --rally-extra-opts="--deployment $1"
# 		done
# 	done
# done

# ops="randread read"

# for op in $ops ; do 
# 	for xxx in $three_times ; do
# 		io_opts="--type $type -a $op --iodepth 16 --blocksize 1k --iosize r2"
# 		python run_rally_test.py -l -o "$io_opts" -t io-scenario $type --rally-extra-opts="--deployment $1"
# 	done
# done
