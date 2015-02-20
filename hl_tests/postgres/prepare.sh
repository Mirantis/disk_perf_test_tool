#!/bin/bash

# install postgres
apt-get update
apt-get install postgresql postgresql-contrib

# check if postrges cluster created

if [ ! -d /etc/postgresql ]; then
    err=$(pg_createcluster 9.3 main --start 2>&1  /dev/null )
    if [ $? -ne 0 ]; then
        echo "There was an error while creating cluster"
        exit 1
    fi
fi

sed -i 's/^local\s\+all\s\+all\s\+peer/local all all trust/g' /etc/postgresql/9.3/main/pg_hba.conf
echo "listen_address = '*'" >> /etc/postgresql/9.3/main/postgresql.conf


