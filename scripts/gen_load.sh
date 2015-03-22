#!/bin/bash

TESTER="--tester-type fio"
CACHE="--cache-modes d"
REPEATS="--repeats 3"

# python generate_load.py $TESTER --size 4k --opers randwrite $CACHE --concurrences 1 --direct-default-size x1000
# python generate_load.py $TESTER --size 4k --opers randwrite $CACHE --concurrences 1 --direct-default-size x2000
# python generate_load.py $TESTER --size 4k --opers randwrite $CACHE --concurrences 1 --direct-default-size x4000
# python generate_load.py $TESTER --size 4k --opers randwrite $CACHE --concurrences 1 --direct-default-size x8000
# python generate_load.py $TESTER --size 4k --opers randwrite $CACHE --concurrences 1 --direct-default-size x16000
# python generate_load.py $TESTER --size 4k --opers randwrite $CACHE --concurrences 1 --direct-default-size x32000
# python generate_load.py $TESTER --size 4k --opers randwrite $CACHE --concurrences 1 --direct-default-size x64000
# python generate_load.py $TESTER --size 4k --opers randwrite $CACHE --concurrences 1 --direct-default-size x128000

python generate_load.py $TESTER --size 4k --opers randwrite $CACHE --concurrences 1 4 8 $REPEATS --io-size 10G
python generate_load.py $TESTER --size 4k --opers randread $CACHE --concurrences 1 4 8 $REPEATS --io-size 10G

python generate_load.py $TESTER --size 4k --opers randwrite --cache-modes s --concurrences 1 $REPEATS --io-size 10G
python generate_load.py $TESTER --size 4k --opers randread randwrite $CACHE --concurrences 1 $REPEATS --io-size 10G
python generate_load.py $TESTER --size 2m --opers read write $CACHE --concurrences 1 $REPEATS --io-size 10G
