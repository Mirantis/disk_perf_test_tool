Installation
============

Requirements:
    * python 3.6+ (all packages: python3 python3-dev python3-pip python3-venv python3-tk)

Manual:
-------

    git clone https://github.com/Mirantis/disk_perf_test_tool.git
    git clone https://github.com/koder-ua/cephlib.git
    git clone https://github.com/koder-ua/xmlbuilder3.git
    git clone https://github.com/koder-ua/agent.git
    cd disk_perf_test_tool
    python3.6 -m pip install wheel
    python3.6 -m pip install -r requirements.txt
    python3.6 -m wally --help


Docker:
-------

Build:

    git clone https://github.com/Mirantis/disk_perf_test_tool.git
    docker build -t <username>/wally .

To run container use:

    docker run -ti <username>/wally /bin/bash
    wally --help

