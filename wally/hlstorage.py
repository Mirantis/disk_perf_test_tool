import os
import pprint
import logging
from typing import cast, Iterator, Tuple, Type, Dict, Optional, List, Any

import numpy

from .suits.job import JobConfig
from .result_classes import SuiteConfig, TimeSeries, DataSource, StatProps, IResultStorage, ArrayData
from .storage import Storage
from .utils import StopTestError
from .suits.all_suits import all_suits


logger = logging.getLogger('wally')


class DB_re:
    node_id = r'\d+.\d+.\d+.\d+:\d+'
    job_id = r'[-a-zA-Z0-9_]+_\d+'
    suite_id = r'[a-z_]+_\d+'
    sensor = r'[-a-z_]+'
    dev = r'[-a-zA-Z0-9_]+'
    tag = r'[a-z_.]+'
    metric = r'[a-z_.]+'


class DB_paths:
    suite_cfg_r = r'results/{suite_id}\.info\.yml'

    job_root = r'results/{suite_id}\.{job_id}/'
    job_cfg_r = job_root + r'info\.yml'

    # time series, data from load tool, sensor is a tool name
    ts_r = job_root + r'{node_id}\.{sensor}\.{metric}\.{tag}'

    # statistica data for ts
    stat_r = job_root + r'{node_id}\.{sensor}\.{metric}\.stat\.yaml'

    # sensor data
    sensor_data_r = r'sensors/{node_id}_{sensor}\.{dev}\.{metric}\.{tag}'
    sensor_time_r = r'sensors/{node_id}_collected_at\.csv'

    report_root = 'report/'
    plot_r = r'{suite_id}\.{job_id}/{node_id}\.{sensor}\.{dev}\.{metric}\.{tag}'
    txt_report = report_root + '{suite_id}_report.txt'

    job_extra = 'meta/{suite_id}.{job_id}/{tag}'

    job_cfg = job_cfg_r.replace("\\.", '.')
    suite_cfg = suite_cfg_r.replace("\\.", '.')
    ts = ts_r.replace("\\.", '.')
    stat = stat_r.replace("\\.", '.')
    sensor_data = sensor_data_r.replace("\\.", '.')
    sensor_time = sensor_time_r.replace("\\.", '.')
    plot = plot_r.replace("\\.", '.')


DB_rr = {name: r"(?P<{}>{})".format(name, rr)
         for name, rr in DB_re.__dict__.items()
         if not name.startswith("__")}


def fill_path(path: str, **params) -> str:
    for name, val in params.items():
        if val is not None:
            path = path.replace("{" + name + "}", val)
    return path


class ResultStorage(IResultStorage):
    # TODO: check that all path components match required patterns

    ts_header_size = 64
    ts_header_format = "!IIIcc"
    ts_arr_tag = 'csv'
    csv_file_encoding = 'ascii'

    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self.cache = {}  # type: Dict[str, Tuple[int, int, ArrayData]]

    def sync(self) -> None:
        self.storage.sync()

    #  -----------------  SERIALIZATION / DESERIALIZATION  -------------------------------------------------------------
    def read_headers(self, fd) -> Tuple[str, List[str], List[str], Optional[numpy.ndarray]]:
        header = fd.readline().decode(self.csv_file_encoding).rstrip().split(",")
        dtype, has_header2, header2_dtype, *ext_header = header

        if has_header2 == 'true':
            ln = fd.readline().decode(self.csv_file_encoding).strip()
            header2 = numpy.fromstring(ln, sep=',', dtype=header2_dtype)
        else:
            assert has_header2 == 'false', \
                "In file {} has_header2 is not true/false, but {!r}".format(fd.name, has_header2)
            header2 = None
        return dtype, ext_header, header, header2

    def load_array(self, path: str) -> ArrayData:
        """
        Load array from file, shoult not be called directly
        :param path: file path
        :return: ArrayData
        """
        with self.storage.get_fd(path, "rb") as fd:
            fd.seek(0, os.SEEK_SET)

            stats = os.fstat(fd.fileno())
            if path in self.cache:
                size, atime, arr_info = self.cache[path]
                if size == stats.st_size and atime == stats.st_atime_ns:
                    return arr_info

            data_dtype, header, _, header2 = self.read_headers(fd)
            assert data_dtype == 'uint64', path
            dt = fd.read().decode(self.csv_file_encoding).strip()

        if len(dt) != 0:
            arr = numpy.fromstring(dt.replace("\n", ','), sep=',', dtype=data_dtype)
            lines = dt.count("\n") + 1
            assert len(set(ln.count(',') for ln in dt.split("\n"))) == 1, \
                "Data lines in {!r} have different element count".format(path)
            arr.shape = [lines] if lines == arr.size else [lines, -1]
        else:
            arr = None

        arr_data = ArrayData(header, header2, arr)
        self.cache[path] = (stats.st_size, stats.st_atime_ns, arr_data)
        return arr_data

    def put_array(self, path: str, data: numpy.array, header: List[str], header2: numpy.ndarray = None,
                  append_on_exists: bool = False) -> None:

        header = [data.dtype.name] + \
                 (['false', ''] if header2 is None else ['true', header2.dtype.name]) + \
                 header

        exists = append_on_exists and path in self.storage
        vw = data.view().reshape((data.shape[0], 1)) if len(data.shape) == 1 else data
        mode = "cb" if not exists else "rb+"

        with self.storage.get_fd(path, mode) as fd:
            if exists:
                data_dtype, _, full_header, curr_header2 = self.read_headers(fd)

                assert data_dtype == data.dtype.name, \
                    "Path {!r}. Passed data type ({!r}) and current data type ({!r}) doesn't match"\
                        .format(path, data.dtype.name, data_dtype)

                assert header == full_header, \
                    "Path {!r}. Passed header ({!r}) and current header ({!r}) doesn't match"\
                        .format(path, header, full_header)

                assert header2 == curr_header2, \
                    "Path {!r}. Passed header2 != current header2: {!r}\n{!r}"\
                        .format(path, header2, curr_header2)

                fd.seek(0, os.SEEK_END)
            else:
                fd.write((",".join(header) + "\n").encode(self.csv_file_encoding))
                if header2 is not None:
                    fd.write((",".join(map(str, header2)) + "\n").encode(self.csv_file_encoding))

            numpy.savetxt(fd, vw, delimiter=',', newline="\n", fmt="%lu")

    def load_ts(self, ds: DataSource, path: str) -> TimeSeries:
        """
        Load time series, generated by fio or other tool, should not be called directly,
        use iter_ts istead.
        :param ds: data source path
        :param path: path in data storage
        :return: TimeSeries
        """
        (units, time_units), header2, data = self.load_array(path)
        times = data[:,0].copy()
        ts_data = data[:,1:]

        if ts_data.shape[1] == 1:
            ts_data.shape = (ts_data.shape[0],)

        return TimeSeries("{}.{}".format(ds.dev, ds.sensor),
                          raw=None,
                          data=ts_data,
                          times=times,
                          source=ds,
                          units=units,
                          time_units=time_units,
                          histo_bins=header2)

    def load_sensor_raw(self, ds: DataSource) -> bytes:
        path = DB_paths.sensor_data.format(**ds.__dict__)
        with self.storage.get_fd(path, "rb") as fd:
            return fd.read()

    def load_sensor(self, ds: DataSource) -> TimeSeries:
        # sensors has no shape
        path = DB_paths.sensor_time.format(**ds.__dict__)
        collect_header, must_be_none, collected_at = self.load_array(path)

        # cut 'collection end' time
        # .copy needed to really remove 'collection end' element to make c_interpolate_.. works correctly
        collected_at = collected_at[::2].copy()

        # there must be no histogram for collected_at
        assert must_be_none is None, "Extra header2 {!r} in collect_at file at {!r}".format(must_be_none, path)
        node, tp, units = collect_header
        assert node == ds.node_id and tp == 'collected_at' and units in ('ms', 'us'),\
            "Unexpected collect_at header {!r} at {!r}".format(collect_header, path)
        assert len(collected_at.shape) == 1, "Collected_at must be 1D at {!r}".format(path)

        data_path = DB_paths.sensor_data.format(**ds.__dict__)
        data_header, must_be_none, data  = self.load_array(data_path)

        # there must be no histogram for any sensors
        assert must_be_none is None, "Extra header2 {!r} in sensor data file {!r}".format(must_be_none, data_path)

        data_units = data_header[2]
        assert data_header == [ds.node_id, ds.metric_fqdn, data_units], \
            "Unexpected data header {!r} at {!r}".format(data_header, data_path)
        assert len(data.shape) == 1, "Sensor data must be 1D at {!r}".format(data_path)

        return TimeSeries(ds.metric_fqdn,
                          raw=None,
                          data=data,
                          times=collected_at,
                          source=ds,
                          units=data_units,
                          time_units=units)

    # -------------  CHECK DATA IN STORAGE  ----------------------------------------------------------------------------

    def check_plot_file(self, source: DataSource) -> Optional[str]:
        path = DB_paths.plot.format(**source.__dict__)
        fpath = self.storage.resolve_raw(DB_paths.report_root + path)
        return path if os.path.exists(fpath) else None

    # -------------   PUT DATA INTO STORAGE   --------------------------------------------------------------------------

    def put_or_check_suite(self, suite: SuiteConfig) -> None:
        path = DB_paths.suite_cfg.format(suite_id=suite.storage_id)
        if path in self.storage:
            db_cfg = self.storage.load(SuiteConfig, path)
            if db_cfg != suite:
                logger.error("Current suite %s config is not equal to found in storage at %s", suite.test_type, path)
                logger.debug("Current: \n%s\nStorage:\n%s", pprint.pformat(db_cfg), pprint.pformat(suite))
                raise StopTestError()
        else:
            self.storage.put(suite, path)

    def put_job(self, suite: SuiteConfig, job: JobConfig) -> None:
        path = DB_paths.job_cfg.format(suite_id=suite.storage_id, job_id=job.storage_id)
        self.storage.put(job, path)

    def put_ts(self, ts: TimeSeries) -> None:
        assert ts.data.dtype == ts.times.dtype, "Data type {!r} != time type {!r}".format(ts.data.dtype, ts.times.dtype)
        assert ts.data.dtype.kind == 'u', "Only unsigned ints are accepted"
        assert ts.source.tag == self.ts_arr_tag, "Incorrect source tag == {!r}, must be {!r}".format(ts.source.tag,
                                                                                                     self.ts_arr_tag)
        csv_path = DB_paths.ts.format(**ts.source.__dict__)
        header = [ts.units, ts.time_units]

        tv = ts.times.view().reshape((-1, 1))
        if len(ts.data.shape) == 1:
            dv = ts.data.view().reshape((ts.times.shape[0], -1))
        else:
            dv = ts.data

        result = numpy.concatenate((tv, dv), axis=1)
        if ts.histo_bins is not None:
            self.put_array(csv_path, result, header, header2=ts.histo_bins)
        else:
            self.put_array(csv_path, result, header)

        if ts.raw:
            raw_path = DB_paths.ts.format(**ts.source(tag=ts.raw_tag).__dict__)
            self.storage.put_raw(ts.raw, raw_path)

    def put_extra(self, data: bytes, source: DataSource) -> None:
        self.storage.put_raw(data, DB_paths.ts.format(**source.__dict__))

    def put_stat(self, data: StatProps, source: DataSource) -> None:
        self.storage.put(data, DB_paths.stat.format(**source.__dict__))

    # return path to file to be inserted into report
    def put_plot_file(self, data: bytes, source: DataSource) -> str:
        path = DB_paths.plot.format(**source.__dict__)
        self.storage.put_raw(data, DB_paths.report_root + path)
        return path

    def put_report(self, report: str, name: str) -> str:
        return self.storage.put_raw(report.encode(self.csv_file_encoding), DB_paths.report_root + name)

    def put_sensor_raw(self, data: bytes, ds: DataSource) -> None:
        path = DB_paths.sensor_data.format(**ds.__dict__)
        with self.storage.get_fd(path, "cb") as fd:
            fd.write(data)

    def append_sensor(self, data: numpy.array, ds: DataSource, units: str, histo_bins: numpy.ndarray = None) -> None:
        if ds.metric == 'collected_at':
            path = DB_paths.sensor_time
            metrics_fqn = 'collected_at'
        else:
            path = DB_paths.sensor_data
            metrics_fqn = ds.metric_fqdn

        if ds.metric == 'lat':
            assert len(data.shape) == 2, "Latency should be histo array"
            assert histo_bins is not None, "Latency should have histo bins"

        path = path.format(**ds.__dict__)
        self.put_array(path, data, [ds.node_id, metrics_fqn, units], header2=histo_bins, append_on_exists=True)

    # -------------   GET DATA FROM STORAGE   --------------------------------------------------------------------------

    def get_stat(self, stat_cls: Type[StatProps], source: DataSource) -> StatProps:
        return self.storage.load(stat_cls, DB_paths.stat.format(**source.__dict__))

    # -------------   ITER OVER STORAGE   ------------------------------------------------------------------------------

    def iter_paths(self, path_glob) -> Iterator[Tuple[bool, str, Dict[str, str]]]:
        path = path_glob.format(**DB_rr).split("/")
        yield from self.storage._iter_paths("", path, {})

    def iter_suite(self, suite_type: str = None) -> Iterator[SuiteConfig]:
        for is_file, suite_info_path, groups in self.iter_paths(DB_paths.suite_cfg_r):
            assert is_file
            suite = self.storage.load(SuiteConfig, suite_info_path)
            # suite = cast(SuiteConfig, self.storage.load(SuiteConfig, suite_info_path))
            assert suite.storage_id == groups['suite_id']
            if not suite_type or suite.test_type == suite_type:
                yield suite

    def iter_job(self, suite: SuiteConfig) -> Iterator[JobConfig]:
        job_glob = fill_path(DB_paths.job_cfg_r, suite_id=suite.storage_id)
        job_config_cls = all_suits[suite.test_type].job_config_cls
        for is_file, path, groups in self.iter_paths(job_glob):
            assert is_file
            job = cast(JobConfig, self.storage.load(job_config_cls, path))
            assert job.storage_id == groups['job_id']
            yield job

    # iterate over test tool data
    def iter_ts(self, suite: SuiteConfig, job: JobConfig, **filters) -> Iterator[TimeSeries]:
        filters.update(suite_id=suite.storage_id, job_id=job.storage_id)
        ts_glob = fill_path(DB_paths.ts_r, **filters)
        for is_file, path, groups in self.iter_paths(ts_glob):
            tag = groups["tag"]
            if tag != 'csv':
                continue
            assert is_file
            groups = groups.copy()
            groups.update(filters)
            ds = DataSource(suite_id=suite.storage_id,
                            job_id=job.storage_id,
                            node_id=groups["node_id"],
                            sensor=groups["sensor"],
                            dev=None,
                            metric=groups["metric"],
                            tag=tag)
            yield self.load_ts(ds, path)

    def iter_sensors(self, node_id: str = None, sensor: str = None, dev: str = None, metric: str = None) -> \
            Iterator[Tuple[str, DataSource]]:
        vls = dict(node_id=node_id, sensor=sensor, dev=dev, metric=metric)
        path = fill_path(DB_paths.sensor_data_r, **vls)
        for is_file, path, groups in self.iter_paths(path):
            cvls = vls.copy()
            cvls.update(groups)
            yield path, DataSource(**cvls)

    def get_txt_report(self, suite: SuiteConfig) -> Optional[str]:
        path = DB_paths.txt_report.format(suite_id=suite.storage_id)
        if path in self.storage:
            return self.storage.get_raw(path).decode('utf8')

    def put_txt_report(self, suite: SuiteConfig, report: str) -> None:
        path = DB_paths.txt_report.format(suite_id=suite.storage_id)
        self.storage.put_raw(report.encode('utf8'), path)

    def put_job_info(self, suite: SuiteConfig, job: JobConfig, key: str, data: Any) -> None:
        path = DB_paths.job_extra.format(suite_id=suite.storage_id, job_id=job.storage_id, tag=key)
        self.storage.put(data, path)

    def get_job_info(self, suite: SuiteConfig, job: JobConfig, key: str) -> Any:
        path = DB_paths.job_extra.format(suite_id=suite.storage_id, job_id=job.storage_id, tag=key)
        return self.storage.get(path, None)
