#!/bin/bash

CLIENTS=${CLIENTS:-"16 32 48 64 80 96"}
TRANSACTINOS_PER_CLIENT=${TRANSACTINOS_PER_CLIENT:-"100 200 300 400 500 600 700 800 900 1000"}
RESULTS_FILE=${RESULTS_FILE:-"results.json"}


createdb -O postgres pgbench
pgbench -i -U postgres pgbench

echo "{" > $RESULTS_FILE

for num_clients in $CLIENTS; do
    for trans_per_cl in $TRANSACTINOS_PER_CLIENT; do
        tps_all=''
        for i in 1 2 3 4 5 6 7 8 9 10; do
            tps=$(pgbench -c $num_clients -n -t $trans_per_cl -j 4 -r -U postgres pgbench |
            grep "(excluding connections establishing)" | awk {'print $3'})
            tps_all="$tps_all\n$tps"
        done
        # calculate average and deviation
        echo "$num_clients $trans_per_cl: " >> $RESULTS_FILE
        echo -e $tps_all | awk  '{ col=1; array[NR]=$col; sum+=$col; print "col="$col,"sum="sum} END {for(x=1;x<=NR;x++){sumsq+=((array[x]-(sum/NR))^2);} print "[" sum/NR "," sqrt(sumsq/(NR-1)) "], " }' >> $RESULTS_FILE
    done
done

echo "}" >> $RESULTS_FILE


