TODO:

    * revise type checking
    * use overloading module

Steps:
    Discover/reuse - returns NodeInfo
    Connect - returns Node from NodeInfo
    Node contains ssh, rpc interface and basic API
    Add aio rpc client

    * Make storage class with dict-like interface
        - map path to value, e.g.  'cluster_info': yaml
        - 'results/r20w80b60kQD10VM2/iops': [iops]
        - should support both binary and text(yaml) formats, maybe store in both
        - store all results in it
        - before call a stage/part check that it results is not awailable yet,
          or chek this inside stage. Skip stage if data already awailable
        - use for discovery, tests, etc
    * aio?
    * Update to newest fio
    * Add fio build script to download fio from git and build it
    * Add integration tests with nbd
    * Move from threads to QD to mitigate fio issues
    * Use agent to communicate with remote node
    * fix existing folder detection
    * fio load reporters
    * Results stored in archived binary format for fast parsing (sqlite)?
    * Split all code on separated modules:
        * logging
        * Test run class
        * Test stage class
    * Results are set of timeseries with attached descriptions

    * move agent and ssh code to separated library
    * plugins for agent
    * evaluate bokeh for visualization
        https://github.com/cronburg/ceph-viz/tree/master/histogram

* Statistical result check and report:
    - check results distribution
    - warn for non-normal results
    - correct comparison
    - detect internal pattern
    https://habrahabr.ru/post/311092/
    https://blog.cloudera.com/blog/2015/12/common-probability-distributions-the-data-scientists-crib-sheet/
    http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.mstats.normaltest.html
    http://profitraders.com/Math/Shapiro.html
    http://www.machinelearning.ru/wiki/index.php?title=%D0%9A%D1%80%D0%B8%D1%82%D0%B5%D1%80%D0%B8%D0%B9_%D1%85%D0%B8-%D0%BA%D0%B2%D0%B0%D0%B4%D1%80%D0%B0%D1%82
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html#numpy.fft.fft
    https://en.wikipedia.org/wiki/Log-normal_distribution
    http://stats.stackexchange.com/questions/25709/what-distribution-is-most-commonly-used-to-model-server-response-time
    http://www.lognormal.com/features/
    http://blog.simiacryptus.com/2015/10/modeling-network-latency.html


* Collect and store cluster info
* Resume stopped/paused run
* Difference calculation
* Resource usage calculator/visualizer
* Bottleneck hunter
* Comprehensive report with results histograms and other
* python3.5
* Docker/lxd public container as default distribution way
* Allow to reuse vm from previous run (store connection config, keys and vm id's in run info)
* Simplify settings
* Save pictures from report in jpg in separated folder
* Node histogram distribution
* Integration with ceph report tool

* Automatically scale QD till saturation

* Integrate vdbench/spc/TPC/TPB
* Runtime visualization
