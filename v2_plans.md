* Code:
    * RPC reconnect in case of errors
    * Download sensors results between test iterations
    * store more information for node - OSD settings, etc
    * Fast-to-parse storage. Maybe BSDDB for all non-raw keys
    * Unit-tests
    * Sensors
        - Revise sensors code. Prepack on node side, different sensors data types
        - perf
        - [bcc](https://github.com/iovisor/bcc)
        - ceph sensors
    * Config revised:
        * Result config then validated
    * Add sync 4k write with small set of thcount
    * Flexible SSH connection creds - use agent, default ssh settings or part of config
    * Remove created temporary files
    * Remove warm-up time from fio. Use warm-up detection to select real test time,
      also fio/OS log files should be used to get test results, not directly
      calculated by fio.

* UT
    * White-box event logs for UT
    * Result-to-yaml for UT

* Infra:
    * Add script to download fio from git and build it
    * Docker/lxd public container as default distribution way
    * Update setup.py to provide CLI entry points

* Statistical result check and report:
    * Comprehensive report with results histograms and other, [Q-Q plot](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot)
    * Check results distribution
    * Warn for non-normal results
    * Check that distribution of different parts is close. Average
      performance should be steady across test
    * Graphs for raw data over time
    * Save pictures from report in jpg in separated folder
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

* Report structure
    * Overall report
    * Extended engineering report
    * Cluster information
    * Loads. For each load:
        - IOPS distribution, stat analisys
        - LAT heatmap/histo, stat analisys
        - Bottleneck analisys
    * Changes for load groups - show how IOPS/LAT histo is chages with thread count
    * Report help page, link for explanations

* Report pictures:
    * checkboxes for show/hide part of image
    * pop-up help for part of picture
    * pop-up text values for bars/lines
    * waterfall charts for ceph request processing

* Intellectual postprocessing:
    * Difference calculation
    * Resource usage calculator/visualizer, bottleneck hunter
    * correct comparison between different systems

* Maybe move to 2.1:
    * Automatically scale QD till saturation
    * Runtime visualization
    * Integrate vdbench/spc/TPC/TPB
    * Add aio rpc client
    * Add integration tests with nbd
    * fix existing folder detection
    * Simple REST API for external in-browser UI