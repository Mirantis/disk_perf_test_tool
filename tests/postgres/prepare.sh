#!/bin/bash
set -e

if [ ! -d /etc/postgresql ]; then
    apt-get update
    apt-get install -y postgresql postgresql-contrib
    err=$(pg_createcluster 9.3 main --start 2>&1  /dev/null )
    if [ $? -ne 0 ]; then
        echo "There was an error while creating cluster"
        exit 1
    fi
fi

sed -i 's/^local\s\+all\s\+all\s\+peer/local all all trust/g' /etc/postgresql/9.3/main/pg_hba.conf
sudo sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/g" /etc/postgresql/9.3/main/postgresql.conf

service postgresql restart

exit 0