# docker build -t ubuntu1604py36
FROM ubuntu:16.04

MAINTAINER Kostiantyn Danylov <koder.mail@gmail.com>

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:jonathonf/python-3.6 && \
    apt-get update &&  \
    apt-get install -y vim git build-essential python3.6 python3.6-dev python3-pip python3.6-venv curl wget

COPY . /opt/disk_perf_tool

# git clone https://github.com/Mirantis/disk_perf_test_tool.git /opt/disk_perf_tool && \
# git checkout v2.0 && \

RUN git clone https://github.com/koder-ua/cephlib.git /opt/cephlib && \
    git clone https://github.com/koder-ua/xmlbuilder3.git /opt/xmlbuilder3 && \
    git clone https://github.com/koder-ua/agent.git /opt/agent && \
    mkdir /opt/wally_libs && \
    ln -s /opt/agent/agent /opt/wally_libs && \
    ln -s /opt/xmlbuilder3/xmlbuilder3 /opt/wally_libs && \
    ln -s /opt/cephlib/cephlib /opt/wally_libs

RUN python3.6 -m pip install pip --upgrade
RUN cd /opt/disk_perf_tool &&  python3.6 -m pip install wheel && python3.6 -m pip install -r requirements.txt

CMD /bin/bash
