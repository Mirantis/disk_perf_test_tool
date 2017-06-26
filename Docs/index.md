How to read this document
=========================

* For fast start go directly to **Howto install wally** section and then to appropriate subsection of 
  **Howto test** section


Overview
========

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

Wally motivation
----------------

Major testing problems and how wally fix them for you

Howto install wally
===================

Container
---------


Local installation
------------------

apt install g++ ....

pip install XXX
python -m wally prepare << download fio, compile


Howto run a test
================

To run a test you need to prepare cluster and config file.

How to run wally: using container, directly


Configuration
=============

* `SSHURI` - string in format [user[:passwd]@]host[:port][:key_file]
    Where:
    - `user` - str, user name, current user by default
    - `passwd` - str, ssh password, can be ommited if `key_file_rr` is provided or default key is used
    - `host` - str, the only required field. Host name or ip address
    - `port` - int, ssh server port to connect, 22 is default
    - `key_file` - str, path to ssh key private file. `~/.ssh/id_rsa` by default. In case if `port` is ommited,
       but `key_file` is provided - it must be separated from host with two colums.
    
    `passwd` and `key_file` must not be used at the same time.
    
    Examples:
    
    - `11.12.23.10:37` - ip and ssh port, current user and `~/.ssh/id_rsa` key used
    - `ceph-1` - only host name, default port, current user and `~/.ssh/id_rsa` key used
    - `ceph-12::~/.ssh/keyfile` - current user and `~/.ssh/keyfile` key used
    - `root@master` - login as root with `~/.ssh/id_rsa` key
    - `root@127.0.0.1:44` - login as root, using 44 port and key from `~/.ssh/id_rsa`
    - `user@10.20.20.30::/tmp/keyfile` - login as root using key from `/tmp/keyfile` 
    - `root:rootpassword@10.20.30.40` - login as root using `rootpassword` as an ssh password 
    
* `[XXX]` - list of XXX type
* `{XXX: YYY}` - mapping from type XXX(key) to type YYY(value)
* `SIZE` - integer number with one of usual K/M/G/T/P suffixes, or without. Be aware that 1024 base is used,
    for 10M really mean 10MiB == 10485760 Bytes and so on.

Default settings
----------------

Many config settings already has usable default values in `config-examples/default.yaml` file and
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

* `sleep`: int, default is zero

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
    
    Use in case if OSD and Monitor nodes registered in ceph via internal ip addresses, which is not visible from
    node,where you run wally.  Allows to map non-routable ip addresses to routable. Example:

```yaml
    ip_remap:
        10.8.0.4: 172.16.164.71
        10.8.0.3: 172.16.164.72
        10.8.0.2: 172.16.164.73
```

Example:

```yaml
ceph:
    root_node: ceph-client
    cluster: ceph    # << optional
    ip_remap:        # << optional
        10.8.0.4: 172.16.164.71
        10.8.0.3: 172.16.164.72
        10.8.0.2: 172.16.164.73
```

Section `openstack`
-------------------

Provides openstack settings, used to discover OS cluster and to spawn/find test vm.

* `skip_preparation`: bool

    Default: `true`, wally need prepared openstack to spawn virtual machines. If you OS cluster was prepared
    previously you can set this setting to `false` to save some time on checks.
    
* `openrc`: either str ir {str: str}
    
    Specify source for [openstack connection settings]. 
    
    - `openrc: ENV` - get OS credentials from environment variables. You need to export openrc setting
       before start wally, like this
       ```
       $ source openrc
       $ RUN_WALLY
       ```
       
       or
       
       ```
       $ env OS_USER=.. OS_PROJECT=..  RUN_WALLY
       ```
     
    - `openrc: str` - use openrc file, located at provided path to get OS connection settings. Example:
    `openrc: /tmp/my_cluster_openrc`
    
    - `openrc: {str: str}` - provide connection settings directly in config file.

        Example:

```yaml
        openrc:
            OS_USERNAME: USER
            OS_PASSWORD: PASSWD
            OS_TENANT_NAME: KEY_FILE
            OS_AUTH_URL: URL
```

* `insecure`: bool - override OS_INSECURE settings, provided in `openrc` section.

* `vms`: [SSHURI]
    List of vm sshuri, except that instead of hostname/ip vm name prefix is used. Wally will found all vm,
    which has a name with this prefix and use them as test nodes.
    
    Example:

```yaml
        vms:
            - wally@wally_vm
            - root:rootpasswd@test_vm
```

This will found all vm named like `wally_vm*` and `test_vm` and try to reuse them for test with provided credentials.
Note that by default for vm wally use openstack ssh key, not `~/.ssh/id_rsa`. See **Openstack vm config** section
for details.

* VM spawning options. This options control how many new vm to spawn for test and what profile to use.
  All spawned vm would automatically get `testnode` role and would be used for tests.
  Wally try to spaw vm evenly across all compute nodes, using anti-affinity groups.

   - `count`: str or int. Control how many vm to spawn, possible values:
       - `=X`, where X is int - spawn as many vm as needed to make total testnodes count not less that X.
         As example - if you already have 1 explicit test node, provided via `nodes` anso wally found 2 vm's left
         from previous test run and you set `count: =4` so wally will spawn one additional vm.
       - `X`, where X is integer. Spawn exactly X new vm.
       - `xX`, where X is integer. Spawn X vm per compute.
        Example: `copunt: x3` - spawn 3 vm per each compute.       
   
   - `cfg_name`: str, vm config. By default only `wally_1024` config are available. This config uses image from
     `https://cloud-images.ubuntu.com/trusty/current/trusty-server-cloudimg-amd64-disk1.img` as vm image,
     1GiB of ram, 2 vCPU and 100GiB volume. See **Openstack vm config** for details.
   
   - `network_zone_name`: str. Network pool for internal ip v4. Usually `net04`
   - `flt_ip_pool`: str. Network pool for floating ip v4. Usually `net04_ext`
   - `skip_preparation`: bool, false by default. By default before spawn vm wally check that all required prerequisites,
      like vm flavour, image, aa-groups, ssh rules are ready and create them is something is missed. This tell
      wally to skip this stage. You may set it if you sure, that openstack is prepared and like to save some time
      on this stage, but better to keep it off to prevent issues.


Section `nodes`
---------------

{SSHURI: [str]} - mapping of `sshuri` to list of roles for selected node. Helps wally in case if it can't
detect cluster nodes. Also all testnodes are provided via this section, except for reused VM.

Example:

```yaml
nodes:
    user@ceph-client: testnode
```

*Note: you need to define at least one node with `testnode` role here, unless you reuse VM in `openstack` section*

Section `tests`
---------------
    
This section define list of test suites to be executed. Each section is a map from suite type to suite config.
See details for different suites below.

fio suite config
----------------
* `load`: str - required option, name of load profile. 

   By default next profiles are available:

   - `ceph` - for all kind of ceph-backed block devices
   - `hdd` - local hdd drive
   - `cinder_iscsi` - cinder lvm-over-iscsi volumes
   - `check_distribution` - check how IOPS/latency are distributed
   
   See **fio task files** section for details.

* `params`: {str: Any} - list of parameters for load profile.
    Subparams:
    - `FILENAME`: str, required by all profiles. It will be used as test file for fio.
        In case if test file name is different on different test nodes you need to create (sym)links with same
        names on all them before start test and use link name here.
    - `FILESIZE`: SIZE, file size parameter. While in most cases wally can correctly detect device/file size
        often you don't need to test whole file. Also this parameter is required if file doesn't exists yet.

    Non-standard loads may need some additional parameters, see **fio task files** section for details.
  
* `use_system_fio`: bool, false by default. Tell `wally` to use testnode local fio binary, instead of one shipped
   with wally. You might need this in case if wally has no prebuild fio for you distribution. By default it's
   better to use wally's fio, as ones with distribution is often outdated. See
   **HOWTO**/`Supply fio for you distribution` for details.

* `use_sudo`: bool, false by default. Wally will run fio on testnodes with sudo. Often this requires if you local
  testnode user is not root, but you need to test device.

* `force_prefill`: bool. false by default. Tell wally to unconditionally fill test file/device with pseudo-random data
   before test. By default wally first check that target is already contains random data and skip filling step.
   On this step wally fill entire device, so it might takes a long.

* `skip_prefill`: bool, false by default. Force wally to don't fill target with pseudorandom data. Use this if you
   are testing local hdd/ssd/cinder iscsi, but not if you testing ceph backed device or any device, which 
   backed by system with delayed space allocation.

Example:

```yaml
  - fio:
      load: ceph
      params:
          FILENAME: /dev/vdb
          FILESIZE: 100G
```

Key `test_profile`: str
-----------------------

This section allows to use some predefined set of settings for spawning VM and run tests.
Available profiles with they settings are listed in config-examples/default.yaml file.
Next profiles are available by default:

* `openstack_ceph` - spawn 1 VM per each compute and run `ceph` fio profile against /dev/vdb
* `openstack_cinder` - spawn 1 VM per each compute and run `ceph_iscsi_vdb` fio profile against /dev/vdb
* `openstack_nova` - spawn 1 VM per each compute and run `hdd` fio profile against /opt/test.bin

Example:

```yaml
include: default.yaml
discover: openstack,ceph
run_sensors: true
results_storage: /var/wally_results

ceph:
    root_node: localhost

openstack:
    openrc: ENV  # take creds from environment variable

test_profile: openstack_ceph
```

CLI
===
.....


Test suites description and motivation
======================================

Other useful information
========================

fio task files
--------------

Openstack vm config
-------------------

image/flavour/ssh keys, etc

Howto test
==========

Local block device
------------------
Use `config-examples/local_block_device.yml` as a template. Replace `{STORAGE_FOLDER}` with path to folder where put
result directory. Make sure, that wally has read/write access to this folder, or can create it. You can either test
device directly,
or test a file on already mounted device. Replace `{STORAGE_DEV_OR_FILE_NAME}` with correct path. In most cases wally
can detect file or block device size correctly, but usually better to set `{STORAGE_OR_FILE_SIZE}` directly. The larger
file you will use, the less affect on result will cause different caches, but also longer would be initial filling it
with pseudo-random data.

Example of testings `sdb` device:

```yaml
include: default.yaml
run_sensors: false
results_storage: /var/wally

nodes:
    localhost: testnode

tests:
  - fio:
      load: hdd
      params:
          FILENAME: /dev/sdb
          FILESIZE: 100G
```

Example of testings device, mounted to `/opt` folder:

```yaml
include: default.yaml
run_sensors: false
results_storage: /var/wally

nodes:
    localhost: testnode

tests:
  - fio:
      load: hdd
      params:
          FILENAME: /opt/some_test_file.bin
          FILESIZE: 100G
```

**Be aware, that wally will not remove file after test complete.**

Ceph without openstack, or other NAS/SAN
----------------------------------------

Wally support only rbd/cephfs testing, object protocols, such as rados and RGW is not supported.
Cephfs testing doesn't requires any special preparation except usual mounting it on test nodes, consult
[ceph fs quick start] for details.

Ceph linear read/write is usually limited by network. As example if you have 10 SATA drives used as storage drives in
you cluster than aggregated linear read speed can reach ~1Gibps or 8Gibps, which is close to 10Gib network limitation.
So unless you have a test node with wide enough network it's usually better to test ceph cluster from several test
nodes in parallel.

Ceph has generally low performance on low QD as in this mode you work with only one OSD at a time.
Meanwhile ceph can scale to much larger QD values than hdd/ssd drives, as in this case you spread IO requests
across all OSD daemons. You need up to (16 * OSD_count) QD for 4k random  reads and about
(12 * OSD_COUNT / REPLICATION_FACTOR) QD for 4k random writes to touch cluster limitations.
For other blocks and modes you might need different settings. You don't need to care about this, if you are using
default `ceph` profile.

There are three ways of testing RBD - direct, by mounting it to node using [krbd] and via virtual machine,
with volume provided by rbd driver, built into qemu. For the last one consult **Ceph with openstack** section
or documentation to you hypervisor.

**TODO: add example**

Using testnode mounted rbd device
---------------------------------

First you need a pool to be used as target for rbd. You can use default `rbd` pool, or create you own for test.
Pool need to have many PG to have good performance. Ballpark estimation is (100 * OSD_COUNT / REPLICATION_FACTOR).
After creation ceph may warn about "too many PG", this message can be safely ignored. Here is ceph documentation:
[PLACEMENT GROUPS].

* Create a pool (consult [ceph pools documentation] for details).
```bash
    $ ceph osd pool create {pool-name} {pg-num} 
```

* Wait till crestion complete and all PG became `active+clean`. 
* Create rbd volume in this pool, volume size need to be selected large enough to mitigate unavoidable OSD nodes
  FS caches. Usually (SUM_RAM_SIZE_ON_ALL_OSD * 3) works good and results in only ~20% cache hit on reads:

```bash
    $ rbd create {vol-name} --size {size} --pool {pool-name}
```

* Mount rbd via kernel rbd device. This is a tricky part. Kernels usually has old version of rbd driver and doesn't
  support newest rbd features. This will results in errors during mounting rbd. First try to mount rbd device:
  
```bash
    $ rbd map {vol-name} --pool {pool-name}
```

If it fails - you need to run `rbd info --pool {pool-name} {vol-name}`, and disable features via
`rbd feature disable --pool {pool-name} {vol-name} {feature name}`. Then try to mount once again.

* wally need to have read/write access to result rbd device.

Direct rbd testing
------------------
Direct rbd testing run via rbd driver, built inside fio. Using this driver fio can generate
requests to RBD directly, without external rbd driver. This is the fastest and the most reliable way
of testing RBD, but with internal rbd driver you bypassing some code layers, which cen be used in production
environment. fio version shipped with wally has no rbd support, as it can't be build statically. In order to use
it you need to build fio with rbd support, see **Use you fio binary** part of **Howto** section for instruction.

**TODO**

Ceph with openstack
-------------------

The easiest way is to use predefined `openstack_ceph` profile. It spawn one VM per each compute node and run `ceph`
test suite on all of them.

Example:

```yaml
include: default.yaml
discover: openstack,ceph
run_sensors: true
results_storage: /var/wally_results

ceph:
    root_node: localhost

openstack:
    openrc: ENV  # take creds from environment variable

test_profile: openstack_ceph
```

Cinder lvm volumes
------------------

Howto
=====

* Use you fio binary

  You need to download fio source, compile it for linux distribution on test nodes, compress with bz2, name
  `fio_{DISTRNAME}_{ARCH}.bz2` and put into `fio_binaries` folder. `ARCH` is output of `arch` command on target system.
  `DISTRNAME` should be same as `lsb_release -c -s` output. 
  
  Here is a tupical steps to compile latest fio from master:
  
```bash
    $ git clone 
    $ cd fio
    $ ./configure --build-static 
    $ make -jXXX  # Replace XXX with you CPU core count to decrease compilation time
    $ bzip2 fio
    $ mv fio.bz2 WALLY_FOLDER/fio_binaries/fio_DISTRO_ARCH.bz2
```

Storage structure
=================

Wally save all input configurations, all collected data and test results into single subfolder of `results_storage`
settings directory. All files are either csv(results/sensor files), yaml/js for configuration and non-numeric
information, png/svg for images and couple of raw text files like logs and some outputs.

Here is a description what each file contains:

* `cli` - txt, wally cli in semi-raw formal
* `config.yaml` - yaml, full final config, build from original wally config, passed as cli parameter by processing all
   replacement and inclusions.
* `log` - txt, wally execution log. Merged log of all wally runs for this test including restarts and report
   generations.
* `result_code` - yaml, contains exit code of last wally execution with 'test' subcommand on this folder.
* `run_interval` - yaml, list of [begin_time, end_time] of last wally execution with 'test' subcommand on this folder.
* `meta` - folder. Contains cached values for statistical calculations.
* `nodes` - folder, information about test cluster
    - `all.yml` - yaml. Contains information for all nodes, except for node parameters
    - `nodes/parameters.js` - js. Contains node parameters. Parameters are stores separatelly, as they can be very large
       for ceph nodes and js files parsed much faster in python than yaml.
* `report` - folder, contains report html/css files and all report images. Can be copied to other place.
    - `index.html` - report start page.
    - `main.css` - report css file.
    - `XXX/YYY.png or .svg` - image files for report
* `results` - folder with all fio results
    - `fio_{SUITE_NAME}_{IDX1}.yml` - yaml, full config for each executed suite.
    - `fio_{SUITE_NAME}_{IDX1}.{JOB_SHORT_DESCR}_{IDX2}` - folder with all data for each job in suite
        * `{NODE_IP}:{NODE_SSH_PORT}.fio.{TS}.(csv|json)` - fio output file. TS is parsed timeseries name - either 
          `bw` or `lat` or `stdout` for output.
        * `info.yml` - 

Development docs
================

Source code structure
---------------------

Source code style
-----------------

Tests
-----

v2 changes
==========
....

wishful thinking about v3
-------------------------


[ceph fs quick start]: http://docs.ceph.com/docs/master/start/quick-cephfs/
[PLACEMENT GROUPS]: http://docs.ceph.com/docs/master/rados/operations/placement-groups/
[ceph pools documentation]: http://docs.ceph.com/docs/kraken/rados/operations/pools/
