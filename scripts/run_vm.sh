#!/bin/bash
MASTER_IP=$1
FUEL_PASSWD=$2

# VM_IP=$(nova floating-ip-create "$FLOATIN_NET" | grep "$FLOATIN_NET" | awk '{print $2}')
VM_IP=172.16.53.26
OS_ORIGIN_IP=192.168.0.2
OS_EXT_IP=172.16.53.2
VM_NAME=koder-disk-test

FIXED_NET_NAME="net04"
FLOATING_NET="net04_ext"

my_dir="$(dirname "$0")"
source "$my_dir/config.sh"
SSH_OVER_MASTER="sshpass -p${FUEL_PASSWD} ssh root@${MASTER_IP}"
VOLUME_NAME="test-volume"
VOLUME_SIZE=20
VOLUME_DEVICE="/dev/vdb"


function get_openrc() {
	OPENRC=`tempfile`
	CONTROLLER_NODE=$($SSH_OVER_MASTER fuel node | grep controller | awk '-F|' '{gsub(" ", "", $5); print $5}')
	$SSH_OVER_MASTER ssh $CONTROLLER_NODE cat openrc 2>/dev/null | sed "s/$OS_ORIGIN_IP/$OS_EXT_IP/g" > $OPENRC
	echo $OPENRC
}

function wait_vm_active() {
	vm_state="none"
	vm_name=$1

	while [ "$vm_state" != "ACTIVE" ] ; do
		sleep 1
		vm_state=$(nova list | grep $vm_name | awk '{print $6}')
	done
}

function boot_vm() {
	FIXED_NET_ID=$(nova net-list | grep "\b${FIXED_NET_NAME}\b" | awk '{print $2}')
	VOL_ID=$(cinder create --display-name $VOLUME_NAME $VOLUME_SIZE | grep '\bid\b' | awk '{print $4}')
	nova boot --flavor "$FLAVOR_NAME" --image "$IMAGE_NAME" --key-name "$KEYPAIR_NAME" --security-groups default --nic net-id=$FIXED_NET_ID $VM_NAME >/dev/null
	wait_vm_active $VM_NAME

	nova floating-ip-associate $VM_NAME $VM_IP
	nova volume-attach $VM_NAME $VOL_ID $VOLUME_DEVICE >/dev/null
}

function prepare_vm() {
	scp -i "$KEY_FILE_NAME" -r ../io_scenario ubuntu@${VM_IP}:/tmp >/dev/null
	scp -i "$KEY_FILE_NAME" $DEBS ubuntu@${VM_IP}:/tmp >/dev/null
	scp -i "$KEY_FILE_NAME" single_node_test_short.sh ubuntu@${VM_IP}:/tmp >/dev/null
	ssh -i "$KEY_FILE_NAME" ubuntu@${VM_IP} sudo dpkg -i $DEBS >/dev/null
}

function prepare_node() {
	COMPUTE_NODE=$($SSH_OVER_MASTER fuel node | grep compute | awk '-F|' '{gsub(" ", "", $5); print $5}')

	sshpass -p${FUEL_PASSWD} scp -r ../io_scenario root@${MASTER_IP}:/tmp
	$SSH_OVER_MASTER scp -r /tmp/io_scenario $COMPUTE_NODE:/tmp >/dev/null

	sshpass -p${FUEL_PASSWD} scp $DEBS root@${MASTER_IP}:/tmp

	$SSH_OVER_MASTER scp $DEBS $COMPUTE_NODE:/tmp
	$SSH_OVER_MASTER ssh $COMPUTE_NODE dpkg -i $DEBS

	sshpass -p${FUEL_PASSWD} scp single_node_test_short.sh root@${MASTER_IP}:/tmp
	$SSH_OVER_MASTER scp /tmp/single_node_test_short.sh $COMPUTE_NODE:/tmp
}

function download_debs() {
	pushd /tmp >/dev/null
	rm -f *.deb >/dev/null
	aptitude download libibverbs1 librdmacm1 libaio1 fio >/dev/null
	popd >/dev/null
	echo /tmp/*.deb
}

# OPENRC=`get_openrc`
# source $OPENRC
# rm $OPENRC

# boot_vm
# prepare_vm


