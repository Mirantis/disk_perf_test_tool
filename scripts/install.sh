#!/bin/bash

FULL="$1"

pushd $(dirname "$0") > /dev/null
SCRIPTPATH=$(pwd -P)
popd > /dev/null

function install_apt() {
    apt-get install -y python-openssl python-pip
}

function install_yum() {
    yum -y install pyOpenSSL python-pip python-ecdsa
}

if which apt-get >/dev/null; then
    install_apt
else
    if which yum >/dev/null; then
        install_yum
    else
        echo "Error: Neither apt-get, not yum installed. Can't install binary dependencies."
        exit 1
    fi
fi

pip install -r "$SCRIPTPATH/../requirements.txt"

if [ "$FULL" == "--full" ] ; then
    pip install -r "$SCRIPTPATH/../requirements_extra.txt"
fi
