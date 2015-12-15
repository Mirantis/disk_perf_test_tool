#!/bin/bash

set -e


while [[ $# > 1 ]]
do
key="$1"

case $key in
    url)
    URL="$2"
    shift
    ;;
    times)
    TIMES="$2"
    shift
    ;;
    *)
    echo "Unknown option $key"
    exit 1
    ;;
esac
shift
done

OMGPATN=/tmp

cd "$OMGPATN"
source venv/bin/activate

cd omgbenchmark/rally_plugin

sed -i -e "s+rabbit:\/\/guest:guest@localhost\/+$URL+g" deployment.json
sed -i -e "s,timeout\": 60,timeouts\": $TIMES,g" task_timeout.json

rally --plugin-paths . deployment create --file=deployment.json --name=test &> /dev/null
rally --plugin-paths . task start task.json &> ~/omg.log

cat ~/omg.log | grep  "Messages count" | grep -o '[0-9,.]\+'
cat ~/omg.log | grep  "total" | grep -o '[0-9,.]\+%' | grep -o '[0-9,.]\+'
cat ~/omg.log | grep  "Load duration" | grep -o '[0-9,.]\+'
