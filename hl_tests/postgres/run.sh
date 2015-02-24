#!/bin/bash
set -e

CLIENTS=${CLIENTS:-"4 8"}
TRANSACTINOS_PER_CLIENT=${TRANSACTINOS_PER_CLIENT:-"1 2"}


sudo -u postgres createdb -O postgres pgbench
sudo -u postgres pgbench -i -U postgres pgbench


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

sudo -u postgres dropdb pgbench

exit 0

