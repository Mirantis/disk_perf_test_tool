FROM ubuntu:14.04
MAINTAINER Petr Lomakin <plomakin@mirantis.com>

RUN apt-get update

RUN apt-get install -y python-dev python-pip python-virtualenv libevent-dev python-libvirt
RUN apt-get install -y libssl-dev libffi-dev

RUN apt-get install -y git python-setuptools git vim; \
    easy_install pip

RUN apt-get install -y python-novaclient

RUN git clone https://github.com/Mirantis/disk_perf_test_tool.git /opt/disk_perf_tool

RUN cd /opt/disk_perf_tool; pip install -r requirements.txt

RUN ["/bin/bash"]
