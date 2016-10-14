* Update to newest fio
* Add fio build/test code
* Add integration tests with nbd
* Move from threads to QD to mitigate fio issues
* Use agent to communicate with remote node
* fix existing folder detection
* fio load reporters

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
