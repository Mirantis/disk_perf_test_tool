function get_arguments() {

    export FUEL_MASTER_IP=$1

    if [ -z "${FUEL_MASTER_IP}" ]; then echo "Fuel master node ip is not provided"; fi

    export EXTERNAL_IP=$2

    if [ -z "${EXTERNAL_IP}" ]; then echo "Fuel external ip is not provided"; fi

    export KEY_FILE_NAME=$3

    if [ -z "${KEY_FILE_NAME}" ]; then echo "Key file name is not provided"; fi

    export FILE_TO_TEST=$4

    if [ -z "${KEY_FILE_NAME}" ]; then echo "Key file name is not provided"; fi

    if [ ! -f $KEY_FILE_NAME ];
    then
       echo "File $KEY_FILE_NAME does not exist."
    fi

    export RESULT_FILE=$5

    if [ -z "${RESULT_FILE}" ]; then echo "Result file name is not provided"; fi

    export FUEL_MASTER_PASSWD=${6:-test37}
    export TIMEOUT=${7:-360}


    echo "Fuel master IP: $FUEL_MASTER_IP"
    echo "Fuel master password: $FUEL_MASTER_PASSWD"
    echo "External IP: $EXTERNAL_IP"
    echo "Key file name: $KEY_FILE_NAME"
    echo  "Timeout: $TIMEOUT"
}

# note : function will works properly only when image dame is single string without spaces that can brake awk
function wait_image_active() {
	image_state="none"
	image_name="$IMAGE_NAME"
    counter=0

	while [ ["$image_state" == "active"] ] ; do
		sleep 1
		image_state=$(glance image-list | grep "$image_name" | awk '{print $12}')
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
	floating_ip="|"
	vm_name=$VM_NAME
    counter=0

	while [ "$floating_ip" != "|" ] ; do
		sleep 1
		floating_ip=$(nova floating-ip-list | grep "$vm_name" | awk '{print $13}' | head -1)
		counter=$((counter + 1))

		if [ $counter -eq $TIMEOUT ]
        then
            echo "Time limit exceed"
            break
        fi
	done
}


function wait_vm_deleted() {
	vm_name=$(nova list| grep "$VM_NAME"| awk '{print $4}'| head -1)
    counter=0

	while [ ! -z $vm_name ] ; do
		sleep 1
		vm_name=$(nova list| grep "$VM_NAME"| awk '{print $4}'| head -1)
		counter=$((counter + 1))

		if [ "$counter" -eq $TIMEOUT ]
        then
            echo "Time limit exceed"
            break
        fi
	done
}


function get_floating_ip() {
    IP=$(nova floating-ip-list | grep "$FLOATING_NET" | awk '{if ($5 == "-") print $2}' | head -n1)

    if [ -z "$IP" ]; then # fix net name
        IP=$(nova floating-ip-create "$FLOATING_NET"| awk '{print $2}')

        if [ -z "$list" ]; then
            echo "Cannot allocate new floating ip"
            # exit
        fi
    fi

    echo $FLOATING_NET
    export VM_IP=$IP
    echo "VM_IP: $VM_IP"
}

function run_openrc() {
    source run_vm.sh "$FUEL_MASTER_IP" "$FUEL_MASTER_PASSWD" "$EXTERNAL_IP" novanetwork nova
    source `get_openrc`

    list=$(nova list)
    if [ "$list" == "" ]; then
        echo "openrc variables are unset or set to the empty string"
    fi

    echo "AUTH_URL: $OS_AUTH_URL"
}

get_arguments $@

echo "getting openrc from controller node"
run_openrc
nova list

echo "openrc has been activated on your machine"
get_floating_ip

echo "floating ip has been found"
bash prepare.sh
echo "Image has been sended to glance"
wait_image_active
echo "Image has been saved"

BOOT_LOG_FILE=`tempfile`
boot_vm | tee "$BOOT_LOG_FILE"
VOL_ID=$(cat "$BOOT_LOG_FILE" | grep "VOL_ID=" | sed 's/VOL_ID=//')
rm "$BOOT_LOG_FILE"

echo "VM has been booted"
wait_floating_ip
echo "Floating IP has been obtained"
source `prepare_vm`
echo "VM has been prepared"

# sudo bash ../single_node_test_short.sh $FILE_TO_TEST $RESULT_FILE

ssh $SSH_OPTS -i $KEY_FILE_NAME ubuntu@$VM_IP \
     "cd /tmp/io_scenario;"

# echo 'results' > $RESULT_FILE; \
#     curl -X POST -d @$RESULT_FILE http://http://172.16.52.80/api/test --header 'Content-Type:application/json'

# nova delete $VM_NAME
# wait_vm_deleted
# echo "$VM_NAME has been deleted successfully"
# cinder delete $VOL_ID
# echo "Volume has been deleted $VOL_ID"
