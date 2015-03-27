FUEL_MASTER_IP=$1
FUEL_MASTER_PASSWD=$2
EXTERNAL_IP=$3
KEY_FILE_NAME=$4


if [ ! -z $5 ]
then
    FILE_TO_TEST=$5
else
    FILE_TO_TEST="bbb.txt"
fi

if [ ! -z $6 ]
then
    FILE_TO_STORE_RESULTS=$6
else
    FILE_TO_STORE_RESULTS="aaa.txt"
fi

if [ ! -z $7 ]
then
    TIMEOUT=$7
else
    TIMEOUT=60
fi


echo "Fuel master IP: $FUEL_MASTER_IP"
echo "Fuel master password: $FUEL_MASTER_PASSWD"
echo "External IP: $EXTERNAL_IP"
echo "Key file name: $KEY_FILE_NAME"

# note : function will works properly only when image dame is single string without spaces that can brake awk
function wait_image_active() {
	image_state="none"
	image_name="$IMAGE_NAME"
    counter=0

	while [ ! $image_state eq "active" ] ; do
		sleep 1
		image_state=$(glance image-list | grep $image_name | awk '{print $12}')
		echo $image_state
		counter=$((counter + 1))

		if [ "$counter" -eq "$TIMEOUT" ]
        then
            echo "Time limit exceed"
            break
        fi
	done
}


function wait_floating_ip() {
    sleep 10
	floating_ip="|"
	vm_name=$VM_NAME
    counter=0

	while [ $floating_ip != "|" ] ; do
		sleep 1
		floating_ip=$(nova floating-ip-list | grep $vm_name | awk '{print $13}' | head -1)
		counter=$((counter + 1))

		if [ $counter -eq $TIMEOUT ]
        then
            echo "Time limit exceed"
            break
        fi
	done
}


function wait_vm_deleted() {
	vm_name=$(nova list| grep $VM_NAME| awk '{print $4}'| head -1)
    counter=0

	while [ ! -z $vm_name ] ; do
		sleep 1
		vm_name=$(nova list| grep $VM_NAME| awk '{print $4}'| head -1)
		counter=$((counter + 1))

		if [ $counter -eq $TIMEOUT ]
        then
            echo "Time limit exceed"
            break
        fi
	done
}

bash run_vm.sh $FUEL_MASTER_IP $FUEL_MASTER_PASSWD $EXTERNAL_IP
source `get_openrc`
list=$(nova list)
if [ "$list" == "" ]; then
    echo "openrc variables are unset or set to the empty string"
fi

IP=$(nova floating-ip-list | grep $FLOATING_NET | awk '{if ($5 == "-") print $2}' | head -n1)

if [ -z $IP ]; then
    IP=$(nova floating-ip-create net04_ext| awk '{print $2}')

    if [ -z $list ]; then
        echo "Cannot allocate new floating ip"
    fi
fi

VM_IP=$IP
echo "VM IP: $VM_IP"

# sends images to glance
bash prepare.sh
wait_image_active
echo "Image has been saved"
VOL_ID=$(boot_vm)
echo "VM has been booted"
wait_floating_ip
echo "Floating IP has been obtained"
source `prepare_vm`
echo "VM has been prepared"

# sudo bash ../single_node_test_short.sh $FILE_TO_TEST $FILE_TO_STORE_RESULTS

ssh $SSH_OPTS -i $KEY_FILE_NAME ubuntu@$VM_IP \
    "cd /tmp/io_scenario; echo 'results' > $FILE_TO_STORE_RESULTS; \
    curl -X POST -d @$FILE_TO_STORE_RESULTS http://http://172.16.52.80/api/test --header 'Content-Type:application/json'"

# nova delete $VM_NAME
# wait_vm_deleted
# echo "$VM_NAME has been deleted successfully"
# cinder delete $VOL_ID
# echo "Volume has been deleted $VOL_ID"
