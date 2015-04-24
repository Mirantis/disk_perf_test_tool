#!/bin/bash
# install and configure mysql
DATABASE_PASSWORD=wally
DATBASE_USER=root
DB_NAME=tpcc

# not prompting db password
debconf-set-selections <<MYSQL_PRESEED
mysql-server mysql-server/root_password password $DATABASE_PASSWORD
mysql-server mysql-server/root_password_again password $DATABASE_PASSWORD
mysql-server mysql-server/start_on_boot boolean true
MYSQL_PRESEED

apt-get install -y mysql-server
apt-get install -y libmysqld-dev
apt-get install -y make

# disable mysql profile in apparmor
sudo ln -s /etc/apparmor.d/usr.sbin.mysqld /etc/apparmor.d/disable/
sudo apparmor_parser -R /etc/apparmor.d/usr.sbin.mysqld

# allows us not to access mysql without specifying password
cat <<EOF >$HOME/.my.cnf
[client]
user=$DATABASE_USER
password=$DATABASE_PASSWORD
host=$DATABASE_HOST
EOF

cd ~
apt-get install bzr
bzr branch lp:~percona-dev/perconatools/tpcc-mysql
cd tpcc-mysql/src
make

cd ..
mysql -e "CREATE DATABASE $DB_NAME;"
mysql "$DB_NAME" < create_table.sql
mysql "$DB_NAME" < add_fkey_idx.sql

./tpcc_load localhost "$DB_NAME" "$DATBASE_USER" "$DATABASE_PASSWORD" 20