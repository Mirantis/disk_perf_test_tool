#!/bin/bash
set -e

while [[ $# > 1 ]]
do
key="$1"

case $key in
    num_clients)
    CLIENTS="$2"
    shift
    ;;
    transactions_per_client)
    TRANSACTINOS_PER_CLIENT="$2"
    shift
    ;;
    *)
    echo "Unknown option $key"
    exit 1
    ;;
esac
shift
done

CLIENTS=$(echo $CLIENTS | tr ',' '\n')
TRANSACTINOS_PER_CLIENT=$(echo $TRANSACTINOS_PER_CLIENT | tr ',' '\n')


sudo -u postgres createdb -O postgres pgbench &> /dev/null
sudo -u postgres pgbench -i -U postgres pgbench &> /dev/null


for num_clients in $CLIENTS; do
    for trans_per_cl in $TRANSACTINOS_PER_CLIENT; do
        tps_all=''
        for i in 1 2 3 4 5 6 7 8 9 10; do
            echo -n "$num_clients $trans_per_cl:"
            sudo -u postgres pgbench -c $num_clients -n -t $trans_per_cl -j 4 -r -U postgres pgbench |
            grep "(excluding connections establishing)" | awk {'print $3'}
        done
    done
done

sudo -u postgres dropdb pgbench &> /dev/null

exit 0

