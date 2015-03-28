#!/bin/bash
MASTER_IP=$1
FUEL_PASSWD=$2
NEW_IP=$3
VM_NAME=disk-io-test2

# VM_IP=$(nova floating-ip-create "$FLOATIN_NET" | grep "$FLOATIN_NET" | awk '{print $2}')
VM_IP=172.16.55.23
OS_ORIGIN_IP=10.20.0.129
OS_EXT_IP=172.16.53.66



FIXED_NET_NAME="net04"
FLOATING_NET="net04_ext"

my_dir="$(dirname -- "$0")"
source "$my_dir/config.sh"
SSH_OVER_MASTER="sshpass -p${FUEL_PASSWD} ssh root@${MASTER_IP}"
VOLUME_NAME="test-volume"
VOLUME_SIZE=20
VOLUME_DEVICE="/dev/vdb"


function get_openrc() {
	OPENRC=`tempfile`
	CONTROLLER_NODE=$($SSH_OVER_MASTER fuel node | grep controller | awk '-F|' '{gsub(" ", "", $5); print $5}')
	$SSH_OVER_MASTER ssh $CONTROLLER_NODE cat openrc 2>/dev/null | \
	    sed -r 's/(\b[0-9]{1,3}\.){3}[0-9]{1,3}\b'/$NEW_IP/ > $OPENRC
	echo $OPENRC
}

function wait_vm_active() {
	vm_state="none"
	vm_name=$VM_NAME
    counter=0

	while [ $vm_state != "ACTIVE" ] ; do
		sleep 1
		vm_state=$(nova list | grep $vm_name | awk '{print $6}')
		counter=$((counter + 1))

		if [ $counter -eq $TIMEOUT ]
        then
            echo "Time limit exceed"
            break
        fi
	done
}

function boot_vm() {
	FIXED_NET_ID=$(nova net-list | grep "\b${FIXED_NET_NAME}\b" | awk '{print $2}')
	VOL_ID=$(cinder create --display-name $VOLUME_NAME $VOLUME_SIZE | grep '\bid\b' | grep available | awk '{print $4}')

    if [ -z $VOL_ID ]; then
        VOL_ID=$(cinder list | grep test-volume | grep available| awk '{print $2}'| head -1)
    fi

	nova boot --flavor "$FLAVOR_NAME" --image "$IMAGE_NAME" --key-name "$KEYPAIR_NAME" --security-groups default --nic net-id=$FIXED_NET_ID $VM_NAME >/dev/null
	wait_vm_active $VM_NAME

	nova floating-ip-associate $VM_NAME $VM_IP

	nova volume-attach $VM_NAME $VOL_ID $VOLUME_DEVICE >/dev/null
	echo "VOL_ID=$VOL_ID"
}

function prepare_vm() {
	echo "Copy io scenario folded"
	scp -i "$KEY_FILE_NAME" -r ../io_scenario ubuntu@${VM_IP}:/tmp >/dev/null

	echo "Copy DEBS packages"
	scp -i "$KEY_FILE_NAME" $DEBS ubuntu@${VM_IP}:/tmp >/dev/null

	echo "Copy single_node_test_short"
	scp -i "$KEY_FILE_NAME" single_node_test_short.sh ubuntu@${VM_IP}:/tmp >/dev/null

    echo "dpkg on vm"
	ssh $SSH_OPTS -i "$KEY_FILE_NAME" ubuntu@${VM_IP} sudo dpkg -i $DEBS >/dev/null
}

function prepare_node() {
	# set -e
	# set -o pipefail
    echo "Preparing node"
	COMPUTE_NODE=$($SSH_OVER_MASTER fuel node | grep compute | awk '-F|' '{gsub(" ", "", $5); print $5}')

	echo "Copying io_scenario to compute node"
	sshpass -p${FUEL_MASTER_PASSWD} scp -r ../io_scenario root@${FUEL_MASTER_IP}:/tmp
	$SSH_OVER_MASTER scp -r /tmp/io_scenario $COMPUTE_NODE:/tmp >/dev/null

	echo "Copying debs to compute node"
	sshpass -p${FUEL_MASTER_PASSWD} scp $DEBS root@${FUEL_MASTER_IP}:/tmp

	$SSH_OVER_MASTER scp $DEBS $COMPUTE_NODE:/tmp
	$SSH_OVER_MASTER ssh $COMPUTE_NODE dpkg -i $DEBS

    echo "Copying single_node_test.sh to compute node"
	sshpass -p${FUEL_MASTER_PASSWD} scp single_node_test_short.sh root@${FUEL_MASTER_IP}:/tmp
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


