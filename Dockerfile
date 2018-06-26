FROM ubuntu:14.04
MAINTAINER Kostiantyn Danylov <koder.mail@gmail.com>

RUN apt-get update

RUN apt-get install -y python-dev python-pip python-virtualenv \
    libevent-dev python-libvirt

RUN apt-get install -y libssl-dev libffi-dev

RUN apt-get install -y python-setuptools git vim curl wget

RUN git clone https://github.com/Mirantis/disk_perf_test_tool.git \
    /opt/disk_perf_tool

RUN cd /opt/disk_perf_tool; bash scripts/install.sh --full

RUN ["/bin/bash"]
