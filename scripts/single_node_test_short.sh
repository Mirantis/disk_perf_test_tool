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
	OPTS="--type=fio"
	sync ; echo 3 > /proc/sys/vm/drop_caches ; python tests/io.py tasks/io_task_randwrite_4kb_1с.cfg --type=fio

	sync ; echo 3 > /proc/sys/vm/drop_caches ; dd if=/dev/zero of=$TEST_FILE bs=1048576 count=10240
	sync ; echo 3 > /proc/sys/vm/drop_caches ; dd if=/dev/zero of=$TEST_FILE bs=1048576 count=10240

	for cycle in $(seq $NUM_CYCLES) ; do
        sync ; echo 3 > /proc/sys/vm/drop_caches ; python tests/io.py tasks/io_task_randwrite_4kb_1с.cfg --type=fio
        sync ; echo 3 > /proc/sys/vm/drop_caches ; python tests/io.py tasks/io_task_randwrite_4kb_4с.cfg --type=fio
        sync ; echo 3 > /proc/sys/vm/drop_caches ; python tests/io.py tasks/io_task_randwrite_4kb_8с.cfg --type=fio

        sync ; echo 3 > /proc/sys/vm/drop_caches ; python tests/io.py tasks/io_task_randread_4kb_1с.cfg --type=fio
        sync ; echo 3 > /proc/sys/vm/drop_caches ; python tests/io.py tasks/io_task_randread_4kb_1с.cfg --type=fio
        sync ; echo 3 > /proc/sys/vm/drop_caches ; python tests/io.py tasks/io_task_randread_4kb_1с.cfg --type=fio

		sync ; echo 3 > /proc/sys/vm/drop_caches ; python tests/io.py tasks/io_task_randwrite_4kb_1с.cfg --type=fio

		sync ; echo 3 > /proc/sys/vm/drop_caches ; python tests/io.py tasks/io_task_reade_2mb.cfg --type=fio
		sync ; echo 3 > /proc/sys/vm/drop_caches ; python tests/io.py tasks/io_task_write_2mb.cfg --type=fio
	done
}

run_tests "$FILE_1" 2>&1 | tee "$OUT_FILE"

# sudo bash scripts/single_node_test_short.sh file_to_test result.txt
