Wally состоит из частей, которые стоит
разделить и унифицировать с другими тулами
------------------------------------------

* Сделать ceph-lib, вынести ее в отдельный проект,
  должна поддерживать и 2.7 и 3.5 и не иметь строгих внешних
  бинарных зависимостейю В нее вынести:
    * Cluster detector
    * Cluster info collector
    * Monitoring
    * FSStorage

* Openstack VM spawn
* Load generators
* Load results visualizator
* Cluster load visualization
* Поиск узких мест
* Расчет потребляемых ресурсов
* Сопрягающий код
* Хранилища должны легко подключаться

* Расчет потребления ресурсов сделать конфигурируемым -
  указывать соотношения чего с чем считать
* В конфиге задавать storage plugin


Ресурсы:
--------
На выходе из сенсоров есть 

NODE_OR_ROLE.DEVICE.SENSOR

create namespace with all nodes/roles as objects with specially overloaded
__getattr__ method to handle device and then handle sensor.
Make eval on result


(CLUSTER.DISK.riops + CLUSTER.DISK.wiops) / (VM.DISK.riops + VM.DISK.wiops)


Remarks:
--------

* With current code impossible to do vm count scan test

TODO next
---------

* Merge FSStorage and serializer into
  ObjStorage, separate TSStorage.
* Build WallyStorage on top of it, use only WallyStorage in code
* check that OS key match what is stored on disk 
* unit tests for math functions
* CEPH PERFORMANCE COUNTERS
* Sync storage_structure
* fix fio job summary
* Use disk max QD as qd limit?
* Cumulative statistic table for all jobs
* Add column for job params, which show how many cluster resource consumed
* show extra outliers with arrows
* Hide cluster load if no nodes available
* Show latency skew and curtosis
* Sort engineering report by result tuple
* Name engineering reports by long summary
* Latency heatmap and violin aren't consistent
* profile violint plot
* Fix plot layout, there to much unused space around typical plot
* iops boxplot as function from QD
* collect device types mapping from nodes - device should be block/net/...
* Optimize sensor communication with ceph, can run fist OSD request for
  data validation only on start.
* Update Storage test, add tests for stat and plot module
* Aggregated sensors boxplot
* Hitmap for aggregated sensors
* automatically find what to plot from storage data (but also allow to select via config)

Have to think:
--------------
* Send data to external storage
* Each sensor should collect only one portion of data. During
  start it should scan all available sources and tell upper code to create separated funcs for them.
* store statistic results in storage
* During prefill check io on file
* Store percentiles levels in TS, separate 1D TS and 2D TS to different classes, store levels in 2D TS
* weight average and deviation
* C++/Go disk stat sensors to measure IOPS/Lat on milliseconds

* TODO large
------------
* Force to kill running fio on ctrl+C and correct cleanup or cleanup all previous run with 'wally cleanup PATH'

* Code:
-------
* RW mixed report
* RPC reconnect in case of errors
* store more information for node - OSD settings, FS on test nodes, target block device settings on test nodes
* Sensors
    - Revise sensors code. Prepack on node side, different sensors data types
    - perf
    - [bcc](https://github.com/iovisor/bcc)
    - ceph sensors
* Config validation
* Add sync 4k write with small set of thcount
* Flexible SSH connection creds - use agent, default ssh settings or part of config
* Remove created temporary files - create all tempfiles via func from .utils, which track them
* Use ceph-monitoring from wally
* Use warm-up detection to select real test time.
* Report code:
    - Compatible report types set up by config and load??
* Calculate statistic for previous iteration in background
        
* UT
----
* UT, which run test with predefined in yaml cluster (cluster and config created separatelly, not with tests)
  and check that result storage work as expected. Declare db sheme in seaprated yaml file, UT should check.
* White-box event logs for UT
* Result-to-yaml for UT

* Infra:
--------
* Add script to download fio from git and build it
* Docker/lxd public container as default distribution way
* Update setup.py to provide CLI entry points

* Statistical result check and report:
--------------------------------------
* KDE on latency, than found local extremums and estimate
  effective cache sizes from them
* [Q-Q plot](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot)
* Check results distribution
* Warn for non-normal results
* Check that distribution of different parts is close. Average performance should be steady across test
* Node histogram distribution
* Interactive report, which shows different plots and data,
  depending on selected visualization type
* Offload simple report table to cvs/yaml/json/test/ascii_table
* fio load reporters (visualizers), ceph report tool
    [ceph-viz-histo](https://github.com/cronburg/ceph-viz/tree/master/histogram)
* evaluate bokeh for visualization
* [flamegraph](https://www.youtube.com/watch?v=nZfNehCzGdw) for 'perf' output
* detect internal pattern:
    - FFT
    - http://mabrek.github.io/
    - https://github.com/rushter/MLAlgorithms
    - https://github.com/rushter/data-science-blogs
    - https://habrahabr.ru/post/311092/
    - https://blog.cloudera.com/blog/2015/12/common-probability-distributions-the-data-scientists-crib-sheet/
    - http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.mstats.normaltest.html
    - http://profitraders.com/Math/Shapiro.html
    - http://www.machinelearning.ru/wiki/index.php?title=%D0%9A%D1%80%D0%B8%D1%82%D0%B5%D1%80%D0%B8%D0%B9_%D1%85%D0%B8-%D0%BA%D0%B2%D0%B0%D0%B4%D1%80%D0%B0%D1%82
    - http://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html#numpy.fft.fft
    - https://en.wikipedia.org/wiki/Log-normal_distribution
    - http://stats.stackexchange.com/questions/25709/what-distribution-is-most-commonly-used-to-model-server-response-time
    - http://www.lognormal.com/features/
    - http://blog.simiacryptus.com/2015/10/modeling-network-latency.html
* For HDD read/write - report caches hit ratio, maps of real read/writes, FS counters
* Report help page, link for explanations
* checkboxes for show/hide part of image
* pop-up help for part of picture
* pop-up text values for bars/lines
* waterfall charts for ceph request processing
* correct comparison between different systems

* Maybe move to 2.1:
--------------------
* Add sensor collection time to them
* Make collection interval configurable per sensor type, make collection time separated for each sensor
* DB <-> files conversion, or just store all the time in files as well
* Automatically scale QD till saturation
* Runtime visualization
* Integrate vdbench/spc/TPC/TPB
* Add aio rpc client
* Add integration tests with nbd
* fix existing folder detection
* Simple REST API for external in-browser UI
