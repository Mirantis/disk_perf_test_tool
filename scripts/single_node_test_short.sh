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

function run_tests(){
	OPTS="--test-file $TEST_FILE --type fio --iodepth 1 --iosize 10G"

	sync ; echo 3 > /proc/sys/vm/drop_caches ; python io.py $OPTS -a randwrite --blocksize 4k -d --concurrency 1

	sync ; echo 3 > /proc/sys/vm/drop_caches ; dd if=/dev/zero of=$TEST_FILE bs=1048576 count=10240
	sync ; echo 3 > /proc/sys/vm/drop_caches ; dd if=/dev/zero of=$TEST_FILE bs=1048576 count=10240

	for cycle in $(seq $NUM_CYCLES) ; do
		for conc in 1 4 8 ; do
			sync ; echo 3 > /proc/sys/vm/drop_caches ; python io.py $OPTS -a randwrite --blocksize 4k -d --concurrency $conc
		done

		for conc in 1 4 8 ; do
			sync ; echo 3 > /proc/sys/vm/drop_caches ; python io.py $OPTS -a randread  --blocksize 4k -d --concurrency $conc
		done

		sync ; echo 3 > /proc/sys/vm/drop_caches ; python io.py $OPTS -a randwrite --blocksize 4k -s --concurrency 1

		sync ; echo 3 > /proc/sys/vm/drop_caches ; python io.py $OPTS -a read      --blocksize 2m -d --concurrency 1
		sync ; echo 3 > /proc/sys/vm/drop_caches ; python io.py $OPTS -a write     --blocksize 2m -d --concurrency 1
	done
}

run_tests "$FILE_1" 2>&1 | tee "$OUT_FILE"


