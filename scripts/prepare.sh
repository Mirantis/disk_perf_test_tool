#!/bin/bash
set -e

my_dir="$(dirname "$0")"
source "$my_dir/config.sh"

# settings
FL_RAM=256
FL_HDD=20
FL_CPU=1


function lookup_for_objects() {
    set +e

    echo -n "Looking for image $IMAGE_NAME ... "
    export img_id=$(nova image-list | grep " $IMAGE_NAME " | awk '{print $2}')
    if [ ! -z "$img_id" ] ; then
        echo " Found"
    else
        echo " Not Found"
    fi

    echo -n "Looking for flavor $FLAVOR_NAME ... "
    export flavor_id=$(nova flavor-list | grep " $FLAVOR_NAME " | awk '{print $2}')
    if [ ! -z "$flavor_id" ] ; then
        echo " Found"
    else
        echo " Not Found"
    fi

    echo -n "Looking for server-group $SERV_GROUP ... "
    export group_id=$(nova server-group-list | grep " $SERV_GROUP " | awk '{print $2}' )
    if [ ! -z "$group_id" ] ; then
        echo " Found"
    else
        echo " Not Found"
    fi

    echo -n "Looking for keypair $KEYPAIR_NAME ... "
    export keypair_id=$(nova keypair-list | grep " $KEYPAIR_NAME " | awk '{print $2}' )
    if [ ! -z "$keypair_id" ] ; then
        echo " Found"
    else
        echo " Not Found"
    fi

    set -e
}

function clean() {
    lookup_for_objects

    if [ ! -z "$img_id" ] ; then
        echo "Deleting $IMAGE_NAME image"
        nova image-delete "$img_id" >/dev/null
    fi

    if [ ! -z "$flavor_id" ] ; then
        echo "Deleting $FLAVOR_NAME flavor"
        nova flavor-delete "$flavor_id" >/dev/null
    fi

    if [ ! -z "$group_id" ] ; then
        echo "Deleting server-group $SERV_GROUP"
        nova server-group-delete "$group_id" >/dev/null
    fi

    if [ ! -z "$keypair_id" ] ; then
        echo "deleting keypair $KEYPAIR_NAME"
        nova keypair-delete "$KEYPAIR_NAME" >/dev/null
    fi

    if [ -f "$KEY_FILE_NAME" ] ; then
        echo "deleting keypair file $KEY_FILE_NAME"
        rm -f "$KEY_FILE_NAME"
    fi
}

function prepare() {
    lookup_for_objects

    if [ -z "$img_id" ] ; then
        echo "Creating $IMAGE_NAME  image"
        opts="--disk-format qcow2 --container-format bare --is-public true"
        glance image-create --name "$IMAGE_NAME" $opts --copy-from "$IMAGE_URL" >/dev/null
        echo "Image created, but may need a time to became active"
    fi

    if [ -z "$flavor_id" ] ; then
        echo "Creating flavor $FLAVOR_NAME"
        nova flavor-create "$FLAVOR_NAME" "$FLAVOR_NAME" "$FL_RAM" "$FL_HDD" "$FL_CPU" >/dev/null
    fi

    if [ -z "$group_id" ] ; then
        echo "Creating server group $SERV_GROUP"
        nova server-group-create --policy anti-affinity "$SERV_GROUP" >/dev/null
    fi

    if [ -z "$keypair_id" ] ; then
        echo "Creating server group $SERV_GROUP. Key would be stored into $KEY_FILE_NAME"
        nova keypair-add "$KEYPAIR_NAME" > "$KEY_FILE_NAME"
        chmod og= "$KEY_FILE_NAME"
    fi

    echo "Adding rules for ping and ssh"
    set +e
    nova secgroup-add-rule default icmp -1 -1 0.0.0.0/0 >/dev/null
    nova secgroup-add-rule default tcp 22 22 0.0.0.0/0 >/dev/null
    set -e
}

if [ "$1" = "--clear" ] ; then
    clean
else
    prepare
fi
