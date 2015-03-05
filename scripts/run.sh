#!/bin/bash
set -e

# don't forget to change it in prepare.sh as well
my_dir="$(dirname "$0")"
source "$my_dir/config.sh"

BLOCK_SIZES="1k 4k 64k 256k 1m"
OPERATIONS="randwrite write randread read"
SYNC_TYPES="s a d"
REPEAT_COUNT="3"
CONCURRENCES="1 8 64"
IODEPTHS="16"


SYNC_FACTOR="x500"
DIRECT_FACTOR="x500"
ASYNC_FACTOR="r2"


function get_file_size_opts() {
    SYNC_TYPE="$1"
    if [ "$SYNC_TYPE" = "s" ] ; then
        echo "--iosize $SYNC_FACTOR -s"
    elif [ "$SYNC_TYPE" = "d" ] ; then
        echo "--iosize $DIRECT_FACTOR -d"
    else
        echo "--iosize $ASYNC_FACTOR"
    fi
}

function echo_combinations() {
    for IODEPTH in $IODEPTHS ; do
        for CONCURRENCE in $CONCURRENCES ; do
            for BSIZE in $BLOCK_SIZES ; do
                for OPERATION in $OPERATIONS ; do 
                    for SYNC_TYPE in $SYNC_TYPES ; do

                        # filter out too slow options
                        if [ "$BSIZE" = "1k" -o "$BSIZE" = "4k" ] ; then
                            if [ "$SYNC_TYPE" = "a" ] ; then
                                continue
                            fi
                        fi 

                        # filter out sync reads
                        if [ "$OPERATION" = "read" -o "$OPERATION" = "randread" ] ; then
                            if [ "$SYNC_TYPE" = "s" ] ; then
                                continue
                            fi
                        fi 

                        FILE_SIZE_AND_SYNC=$(get_file_size_opts "$SYNC_TYPE")


                        IO_OPTS="--type $TESTER_TYPE "
                        IO_OPTS="$IO_OPTS -a $OPERATION "
                        IO_OPTS="$IO_OPTS --iodepth $IODEPTH "
                        IO_OPTS="$IO_OPTS --blocksize $BSIZE "
                        IO_OPTS="$IO_OPTS $FILE_SIZE_AND_SYNC "
                        IO_OPTS="$IO_OPTS --concurrency $CONCURRENCE"

                        for COUNTER in $(seq 1 $REPEAT_COUNT) ; do
                            echo $IO_OPTS
                        done
                    done
                done
            done
        done
    done
}


function run_test() {
    OPTION_FILE="$1"

    if [ ! -f "$OPTION_FILE" ] ; then
        echo "Path to file with io.py options list should be passed"
        exit 1
    fi

    if [ "$RUNNER" = "ssh" ] ; then
        GROUP_ID=$(nova server-group-list | grep " $SERV_GROUP " | awk '{print $2}' )
        EXTRA_OPTS="user=$IMAGE_USER"
        EXTRA_OPTS="${EXTRA_OPTS},keypair_name=$KEYPAIR_NAME"
        EXTRA_OPTS="${EXTRA_OPTS},img_name=$IMAGE_NAME"
        EXTRA_OPTS="${EXTRA_OPTS},flavor_name=$FLAVOR_NAME"
        EXTRA_OPTS="${EXTRA_OPTS},network_zone_name=$NETWORK_ZONE_NAME"
        EXTRA_OPTS="${EXTRA_OPTS},flt_ip_pool=$FL_NETWORK_ZONE_NAME"
        EXTRA_OPTS="${EXTRA_OPTS},key_file=$KEY_FILE_NAME"
        EXTRA_OPTS="${EXTRA_OPTS},aff_group=$GROUP_ID"
        EXTRA_OPTS="${EXTRA_OPTS},count=$VM_COUNT"
    else
        echo "Unsupported runner $RUNNER"
        exit 1
    fi

    RUN_TEST_OPTS="-t io -l --runner $RUNNER"
    set -x
    python run_test.py $RUN_TEST_OPTS --create-vms-opts="$EXTRA_OPTS" -f "$OPTION_FILE" $TESTER_TYPE
    set +x
}

if [ "$1" = '--prepare-opts' ] ; then
    echo_combinations
else
    run_test $1
fi

