Introduction
------------

Wally is a tool to measure performance of block storages of different kinds
in distributed ways and provide comprehensive reports. It's designed to
run in distributed and cloud environments, but can measure single disk as well.

Wally put much effort in run test in controlled way, process result correctly
from statistical point of view and provide numbers you can rely on, argue about,
and understand.

Wally is not load geenrating tool. It uses well-known load generators - [fio],
to test system and provides wrappers around it, which greatly helps with
discovering cluster features and settings, install sensors, preparing system for
test, run complex test suites from several test nodes in parallel and visualize
results.

The main features:
* Cluster and storage preparation to obtain as much reproducible results, as possible
* Integration with [openstack], [ceph] and [fuel]
* Distributed test execution
* Tight integration with [fio]
* VM spawning in OS for test
* Sensors subsystem, which collect load on cluster devices during test
* Simple yet flexible config files, which allow to specify cluster structure and
  select information to collect during load
* Comprehensive visual report
* Resource consumption report, which allows to see how much cluster resources used
  to provide service to client
* Bottleneck analizer, which helps to find parts, which affect results the most
* yaml/cvs based storage for all results, so you can easily plug them into you result
  processing pipeline
* wally can restart test in case of failure from failed stage

What wally can't do:
* Deploy/configure storage - test system must be ready for test
* Update report during test execution. Wally is completely cli tool with no UI,
  reports are generated after test complete
* Provide interactive reports. All images/tables are generated with matplotlib
  and static.


Basic architecture overview
---------------------------

Wally code is consists of 3 main parts - [agent library], [cephlib] and [wally] itself.
Agent library is responsible for providing [RPC] connection to cluster and test nodes.
Cephlib contain the most of storage, discovery, sensors, data processing and visualization code.
Wally itself provides cli, load tools integration, report generation and other parts.

fio is a main load tool, which tigtly integrated inside wally. Wally have own fio version,
build for some linux distributives. Wally can use system fio as well, but needs one of latest
version to be installed. Fio config files is located in wally/suits/io folder wiht cfg extension.
default_qd.cfg is file with default settings, it includes in mostly all other configs. ceph.cfg,
hdd.cfg, cinder_iscsi.cfg is a primary test suites. cfg files is a fio config files, with a bit
of extra features, provided by wally. Before test wally insert provided settings in selected cfg
file, unroll cycles, split it into jobs and synchronously execute jobs one-by-one from test nodes.

While fio provides some of such features wally don't use them to control results more precisely.

To run test wally need a config file, which contains cluster information, sensors settings,
test config and some other variables to control test execution and results processing.
Examples of config files are located in conf0g-examples folder. All config files in this folder
includes default.yaml, which, in his turn, includes logging.yaml. In most cases you don't need
to change anything in default.yaml/logging.yaml files. Configuration files described in details
below.

Wally execution consist of stages, most of stages maps to config file blocks. Main stages are:

* Cluster discovery
* Connecting to nodes via ssh
* Instrumenting nodes with rpc servers
* Installing sensors, accordingly to config file
* Run test
* Generating report
* Cleanup

How to run a test
-----------------

To run a test you need to prepare cluster and config file.


Configuration file
==================


* `SSHURI` - string in format [user[:passwd]@]host[:port][:key_file]
    Where:
    - `user` - str, user name, current user by default
    - `passwd` - str, ssh password, can be ommited if `key_file_rr` is provided or default key is used
    - `host` - str, the only required field. Host name or ip address
    - `port` - int, ssh server port to connect, 22 is default
    - `key_file` - str, path to ssh key private file. `~/.ssh/id_rsa` by default. In case if `port` is ommited,
       but `key_file` is provided - it must be separated from host with two colums.
    
    `passwd` and `key_file` must not be used at the same time. Examples:
    
    - root@master
    - root@127.0.0.1:44
    - root@10.20.20.30::/tmp/keyfile
    - root:rootpasswd@10.20.30.40
    
* `[XXX]` - list of XXX type
* `{XXX: YYY}` - mapping from type XXX(key) to type YYY(value)
    

default settings
----------------

Many config settings already has usable default values in config-examples/default.yaml file and
in most cases use can reuse them. For those put next include line in you config file:

`include: default.yaml`

You can be override selected settings in you config file.

Plain settings
--------------

* `discover`: [str]
   
    Possible values in list: `ceph`, `openstack`, `fuel`, `ignore_errors`. Example:
   
    `discover: openstack,ceph`

    Give wally list of clusters to discover. Cluster discovery used to find cluster nodes along with
    they roles to simplify settings configuration and some other steps. You can always define or redefine nodes roles
    in `explicit` section. Each cluster requires additional config section. `ignore_errors` mean to ignore
    missing clusters.

* `run_sensors`: bool
    
    Set to true, to allow wally to collect load information during test. This greatly increase result size,
    but allows wally to provide much more sophisticated report.

* `results_storage`: str

    Default folder to put results. For each test wally will generate unique name and create subfolder in this
    directory, where all results and settings would be stored. Wally must have `rwx` access to this folder. 

    Example: `results_storage: /var/wally`

* `sleep`: int

    Tell wally to do nothing for X seconds. Useful if you only need to collect sensors.
    
    Example: `sleep: 60`

Section `ceph`
--------------

Provides settings to discover ceph cluster nodes

* `root_node`: str
    
    Required. Ssh url of root node. This can be any node, which has ceph client key (any node, where you can run
    `ceph` cli command).

* `cluster`: str
    
    Ceph cluster name. `ceph` by default.

* `conf`: str
    
    Path to cluster config file. /etc/ceph/{cluster_name}.conf by default.

* `key`: str
    
    Path to `client.admin` key file. /etc/ceph/{cluster_name}.client.admin.keyring by default.

* `ip_remap`: {IP: IP}
    
    Used in case if OSD and Monitor nodes registered in ceph using internal ip addresses, which is not visible from
    master node.  Allows to map non-routable ip addresses to routable. Example:

    ```
    ip_remap:
        10.8.0.4: 172.16.164.71
        10.8.0.3: 172.16.164.72
        10.8.0.2: 172.16.164.73
    ```

Section `nodes`
---------------

{SSHURI: [str]} - contains mapping of sshuri to list of roles for selected node. Helps wally in case if it can't
detect cluster nodes. Also all testnodes are provided via this section and at least one node with role testnode
must be provided. Example:

```
nodes:
    user@ceph-client: testnode
```

Section `tests`
---------------



fio task files
--------------


