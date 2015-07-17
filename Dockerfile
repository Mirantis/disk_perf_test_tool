FROM ubuntu:14.04
MAINTAINER Petr Lomakin <plomakin@mirantis.com>

RUN apt-get update

# RUN apt-get install -y python-dev python-pip python-virtualenv libevent-dev python-libvirt
# RUN apt-get install -y libssl-dev libffi-dev

RUN apt-get install -y git vim
RUN git clone https://github.com/Mirantis/disk_perf_test_tool.git /opt/wally
RUN cd /opt/wally; ./install.sh --full
RUN ["/bin/bash"]
