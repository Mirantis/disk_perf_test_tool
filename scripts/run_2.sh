#!/bin/bash
set -x

type="iozone"

# nova image-list | grep ' ubuntu ' >/dev/null
# if [ $? -ne 0 ] ; then
# 	url="https://cloud-images.ubuntu.com/trusty/current/trusty-server-cloudimg-amd64-disk1.img"
# 	glance image-create --name 'ubuntu' --disk-format qcow2 --container-format bare --is-public true --copy-from $url
# fi

# nova flavor-list | grep ' ceph.512 ' >/dev/null
# if [ $? -ne 0 ] ; then
# 	nova flavor-create ceph.512 ceph.512 512 50 1
# fi

# nova server-group-list | grep ' ceph ' >/dev/null
# if [ $? -ne 0 ] ; then
# 	nova server-group-create --policy anti-affinity ceph
# fi

# nova keypair-list | grep ' ceph ' >/dev/null
# if [ $? -ne 0 ] ; then
# 	nova keypair-add ceph > ceph.pem
# fi

set -e

io_opts="--type $type -a write --iodepth 16 --blocksize 1m --iosize x20"
python run_test.py --runner ssh -l -o "$io_opts" -t io-scenario $type --runner-extra-opts="ubuntu ceph.pem"

# io_opts="--type $type -a write --iodepth 16 --blocksize 1m --iosize x20"
# python run_test.py --runner rally -l -o "$io_opts" -t io-scenario $type --runner-extra-opts="--deployment perf-1"
