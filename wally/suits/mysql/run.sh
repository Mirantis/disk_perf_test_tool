#!/bin/bash
set -e
set -x

while [[ $# > 1 ]]
do
key="$1"

case $key in
    warehouses)
    WAREHOUSES="$2"
    shift
    ;;
    *)
    echo "Unknown option $key"
    exit 1
    ;;
esac
shift
done

DATABASE_PASSWORD=wally
DATBASE_USER=root
DB_NAME=tpcc

cd ~/tpcc-mysql
./tpcc_start -h127.0.0.1 "-d$DB_NAME" "-u$DATBASE_USER" "-p$DATABASE_PASSWORD" -w"WAREHOUSES" -c16 -r10 -l1200 > ~/tpcc-output.log
echo "TpmC:" `cat ~/tpcc-output.log | grep  TpmC | grep -o '[0-9,.]\+'`
