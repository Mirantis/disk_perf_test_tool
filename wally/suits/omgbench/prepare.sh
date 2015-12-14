#!/bin/bash

set -e
set -x

OMGPATN=/tmp

mkdir -p "$OMGPATN"
cd "$OMGPATN"

git clone https://github.com/openstack/rally
git clone https://github.com/Yulya/omgbenchmark

mkdir venv
cd rally
./install_rally.sh -d "$OMGPATN"/venv -y

cd "$OMGPATN"
source venv/bin/activate
apt-get -y install python-scipy libblas-dev liblapack-dev libatlas-base-dev gfortran
pip install oslo.messaging petname scipy
