#!/bin/bash

FULL="$1"

pushd $(dirname "$0") > /dev/null
SCRIPTPATH=$(pwd -P)
popd > /dev/null

function install_apt() {
    apt-get install -y python-openssl python-novaclient python-cinderclient \
                       python-keystoneclient python-glanceclient python-faulthandler \
                       python-pip

    if [ "$FULL" == "--full" ] ; then
        apt-get install -y python-scipy python-numpy python-matplotlib python-psutil
    fi
}


function install_yum() {
    yum -y install pyOpenSSL python-novaclient python-cinderclient \
                   python-keystoneclient python-glanceclient \
                   python-pip python-ecdsa

    if [ "$FULL" == "--full" ] ; then
        yum -y install scipy numpy python-matplotlib python-psutil
    fi
}

if which apt-get >/dev/null; then
    install_apt
else
    if which yum >/dev/null; then
        install_yum
    else
        echo "Error: Neither apt-get, not yum installed. Can't install deps"
        exit 1
    fi
fi

pip install -r "$SCRIPTPATH/../requirements.txt"

if [ "$FULL" == "--full" ] ; then
    pip install oktest iso8601==0.1.10
fi
