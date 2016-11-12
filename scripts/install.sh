#!/bin/bash

FULL="$1"

pushd $(dirname "$0") > /dev/null
SCRIPTPATH=$(pwd -P)
popd > /dev/null

function install_apt() {
    MODULES="python-openssl  python-faulthandler python-pip"
    if [ "$FULL" == "--full" ] ; then
        MODULES="$MODULES python-scipy python-numpy python-matplotlib python-psutil"
    fi
    apt-get install -y $MODULES
}


function install_yum() {
    MODULES="pyOpenSSL python-pip python-ecdsa"
    if [ "$FULL" == "--full" ] ; then
        MODULES="$MODULES scipy numpy python-matplotlib python-psutil"
    fi
    yum -y install $MODULES
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
