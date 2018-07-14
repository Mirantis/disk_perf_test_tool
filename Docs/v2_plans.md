TODO today:
-----------

* Для сефа выводить нагрузку на пулы и PG





* QDIOTimeHeatmap поломано получаение девайса по роли (aka ceph-storage)
* ceph-osd ноды должны верифицироваться как storage также (в фильтрации девайсов)
* дефолтный ключ в конфиг для всех новых нод
* Проверять что файл ключа для нод существует. Есл иего нет валли ведет себя некорректно
* Сторадж - все json - назвать js, все yaml - yml, всем без расширения дать расширения
* Download boostrap & font css files and store them in report folder. Better to keep all in git repo
* Add cluster config and tasks configs with compiled loads to report
* add fiorbd default_qd and ceph profiles
* Ceph profile - define QD from OSD count
* Add cleanup cli, which get test output folder and clean everything
* Add os cleanup cli, which removes all vm, flavours, images, keys from it
* Add cli option to remove test file after test completed.
* Make a python module, set all deps, clib/fio compilation, add UT for installation
* openstack VM reuse - allow to use full sshuri with name prefix instead of vm name
* fiorbd???? should be quite easy
* allow to provide key for reused OS vms
* openstack vm reuse - маркать виртуалки (wally_test_XXX) и находить их по этому имени и автоматически переиспользовать
* Согласовать имя секции для 'fio' - в default.yaml используется 'io', в конфигах - 'fio'
* Добавить UT
* Сделать полноценные тесты постобработки с генерацией псевдослучаныйх данных с определенным распределением
* Прогнать ceph/hdd тесты
* Написать документацию. Описать опции CLI, config, default_config, как формируются fio таски и структуру хранилища
* Сделать docker image
* Сделать 2.0 релиз

* Оформить примеры доступа к данным
* Обдумать gtod_cpu опцию
* Переделать код ресурсов под новые селекторы
* Не все роли в классификаторе
* Модель поиска через регулярки не работает для больших кластеров.
  Проблема в избыточном применении re.match. Нужно делать предвыборку
  по известным полям (как это сделать универсально). Задача эквивалентна
  индексу в многомерной таблице. В простейшем случае сделать несколько
  многовложенных словарей с разным порядком полей. Перебор происходит по
  ноде и девайсу, но не по сенсору и метрике - 
  {(sensor, metric): [(dev, node), ...]} или
  {sensor: {metric: [(dev, node), ...]}}

* Расширить локатор устройств, используя роли ноды и ее hw_info
* Построить для всех лоадов одну таблицу - сервис за X секунд vs. ресурсы
* Cluster summary
* Хранить csv данные агрегированные по устройсту???
* для чтения нужно строить heatmap 'storage read block size', не write
* Storage nodes cpu non-idle heatmap
* Check ceph fs flush time is larger that tests time
* Расчет кеш-попаданий при чтении
* scipy.stats.shapiro для теста на нормальность
* Проверить стоп по ctrl+c
* bottleneck table
* Рассмотреть pandas.dataframe как универсальный посредник для
  ф-ций визуализации
* scipy.stats.probplot - QQ plot
* Маркать девайсы на нодах по ролям при диагностике нод
* Унифицировать имена настроек - e.g. hmap_XXX для хитмапа.
* Может шестигранники вместо heatmap?
* Проверить и унифицировать все кеши. Отдельно поиск TS, который всегда
  выдает Union[DataSource, Iterable[DataSource]]. Отдельно одна
  унифицированная функция загрузки/постобработки, в которой все базовые
  кеши. Дополнительно слой постобработки(агрегация по ролям, девайсам,
  CPU) со своими кешами. Хранить original в TS.
* Собирать репорт с кластера

Проблемы
--------

* Посмотреть почему тест дикки-фуллера так фигово работает
* Что делать с IDLE load?

Wally состоит из частей, которые стоит разделить и унифицировать с другими тулами:
----------------------------------------------------------------------------------

* Оптимизировать как-то сбор 'ops in fly', проверить как это влияет на сеф
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
* check that OS key match what is stored on disk 
* CEPH PERFORMANCE COUNTERS
* Hide cluster load if no nodes available
* iops boxplot as function from QD
* Optimize sensor communication with ceph, can run fist OSD request for data validation only on start.
* Update Storage test, add tests for stat and plot module
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



# ----------------------------------------------------------------------------------------------------------------------


2.0:
	* Сравнения билдов - по папкам из CLI, текcтовое
	* Занести интервал усреднения в конфиг
	* починить SW & HW info, добавить настройки qemu и все такое
	* Перед началом теста проверять наличие его результатов и скипать
	* продолжение работы при большинстве ошибок
	* Починить процессор
	* Починить боттлнеки
	* Юнит-тесты
	* Make python module
	* putget/ssbench tests
	* rbd с нод без виртуалок
	* отдельный тенант на все и очистка полная
	* Per-vm stats & between vm dev
	* Логи визуальные
	* psql, mssql, SPC-1
	* Тестирование кешей

Done:
	* собрать новый fio под основные платформы и положить в git
	* Все тесты - в один поток
	* Перейти на анализ логов fio
	* Делать один больщой тест на несколько минут и мерять по нему все параметры
	* печатать fio параметры в лог

Мелочи:
	* Зарефакторить запуск/мониторинг/оставнов процесса по SSH, запуск в фоне с чеком - в отдельную ф-цию
	* prefill запускать в фоне и чекать периодически
	* починить все подвисания во всех потоках - дампить стеки при подвисании и таймаут
	* fadvise_hint=0
	* Изменить в репорте сенсоров все на % от суммы от тестнод
	* посмотреть что с сетевыми картами
	* Intellectual granular sensors
