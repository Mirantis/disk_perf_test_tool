#!/bin/bash
set -x

TEST_FILE=$1
OUT_FILE=$2
NUM_CYCLES=7
# TESTS_PER_CYCLE=9

# COUNTER=0
# (( NUM_TESTS=$NUM_CYCLES * $TESTS_PER_CYCLE))

# function next() {
# 	echo "Done $COUNTER tests from $NUM_TESTS"
# 	(( COUNTER=$COUNTER + 1 ))
# }

function super_sync() {
	sync
	echo 3 > /proc/sys/vm/drop_caches
}

function run_tests(){
	OPTS="--test-file $TEST_FILE --type fio --iodepth 1 --iosize 10G"
	OPERS="read write randread randwrite"
	CONCS="1 4 8 64"
	SIZES="4k 16k 64k 256k 1m 2m"

	# num cycles = 6 * 4 * 7 * 4 + 7 * 4 * 4 == 784 == 13 hours

	super_sync ; dd if=/dev/zero of=$TEST_FILE bs=1048576 count=10240

	for cycle in $(seq $NUM_CYCLES) ; do
		for conc in $CONCS ; do
			for bsize in $SIZES ; do
				for operation in $OPERS ; do
					super_sync ; python io.py $OPTS -a $operation --blocksize $bsize -d --concurrency $conc
				done
			done
		done
	done

	for cycle in $(seq $NUM_CYCLES) ; do
		for conc in $CONCS ; do
			for operation in $OPERS ; do
				super_sync ; python io.py $OPTS -a $operation --blocksize 4k -s --concurrency $conc
			done
		done
	done

	super_sync ; python io.py $OPTS -a write --blocksize 2m --concurrency 1
	super_sync ; python io.py $OPTS -a write --blocksize 2m --concurrency 1
	super_sync ; python io.py $OPTS -a write --blocksize 2m --concurrency 1

	OPTS="--test-file $TEST_FILE --type fio --iodepth 1 --iosize 1G"
	for cycle in $(seq $NUM_CYCLES) ; do
		super_sync ; python io.py $OPTS -a randwrite --blocksize 4k -d --concurrency 1
	done

	OPTS="--test-file $TEST_FILE --type fio --iodepth 1 --iosize 10G"
	# need to test different file sizes
	# need to test different timeouts - maybe we can decrease test time
}

run_tests "$FILE_1" 2>&1 | tee "$OUT_FILE"


