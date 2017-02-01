import os
import logging
from typing import cast, Iterator, Tuple, Type, Dict, Optional

import numpy

from .suits.job import JobConfig
from .result_classes import SuiteConfig, TimeSeries, DataSource, StatProps, IResultStorage
from .storage import Storage, csv_file_encoding
from .utils import StopTestError, str2shape, shape2str
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

    job_root = r'results/{suite_id}.{job_id}/'
    job_cfg_r = job_root + r'info\.yml'

    # time series, data from load tool, sensor is a tool name
    ts_r = job_root + r'{node_id}\.{sensor}\.{metric}.{tag}'

    # statistica data for ts
    stat_r = job_root + r'{node_id}\.{sensor}\.{metric}\.stat.yaml'

    # sensor data
    sensor_data_r = r'sensors/{node_id}_{sensor}\.{dev}\.{metric}\.csv'
    sensor_time_r = r'sensors/{node_id}_collected_at\.csv'

    report_root = 'report/'
    plot_r = r'report/{suite_id}\.{job_id}/{node_id}\.{sensor}\.{dev}\.{metric}\.{tag}'

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

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def sync(self) -> None:
        self.storage.sync()

    #  -----------------  SERIALIZATION / DESERIALIZATION  -------------------------------------------------------------

    def load_ts(self, ds: DataSource, path: str) -> TimeSeries:

        with self.storage.get_fd(path, "rb") as fd:
            header = fd.readline().decode(csv_file_encoding).strip().split(",")
            shape, dtype, units, time_units = header
            arr = numpy.loadtxt(fd, delimiter=',', dtype=dtype)

        return TimeSeries("{}.{}".format(ds.dev, ds.sensor),
                          raw=None,
                          data=arr[:,1:].reshape(str2shape(shape)),
                          times=arr[:,0],
                          source=ds,
                          units=units,
                          time_units=time_units)

    def load_sensor(self, ds: DataSource) -> TimeSeries:
        collect_header, collected_at = self.storage.get_array(DB_paths.sensor_time.format(**ds.__dict__))
        assert collect_header == [ds.node_id, 'collected_at', 'us'], repr(collect_header)

        data_header, data = self.storage.get_array(DB_paths.sensor_data.format(**ds.__dict__))

        data_units = data_header[2]
        assert data_header == [ds.node_id, ds.metric_fqdn, data_units]

        return TimeSeries(ds.metric_fqdn,
                          raw=None,
                          data=data,
                          times=collected_at,
                          source=ds,
                          units=data_units,
                          time_units='us')

    # -------------  CHECK DATA IN STORAGE  ----------------------------------------------------------------------------

    def check_plot_file(self, source: DataSource) -> Optional[str]:
        path = DB_paths.plot.format(**source.__dict__)
        fpath = self.storage.resolve_raw(path)
        return path if os.path.exists(fpath) else None

    # -------------   PUT DATA INTO STORAGE   --------------------------------------------------------------------------

    def put_or_check_suite(self, suite: SuiteConfig) -> None:
        path = DB_paths.suite_cfg.format(suite_id=suite.storage_id)
        if path in self.storage:
            db_cfg = self.storage.get(path)
            if db_cfg != suite:
                logger.error("Current suite %s config is not equal to found in storage at %s", suite.test_type, path)
                raise StopTestError()

        self.storage.put(suite, path)

    def put_job(self, suite: SuiteConfig, job: JobConfig) -> None:
        path = DB_paths.job_cfg.format(suite_id=suite.storage_id, job_id=job.storage_id)
        self.storage.put(job, path)

    def put_ts(self, ts: TimeSeries) -> None:
        assert ts.data.dtype == ts.times.dtype
        assert ts.data.dtype.kind == 'u'
        assert ts.source.tag == self.ts_arr_tag

        csv_path = DB_paths.ts.format(**ts.source.__dict__)
        header = [shape2str(ts.data.shape),
                  ts.data.dtype.name,
                  ts.units,
                  ts.time_units]

        with self.storage.get_fd(csv_path, "cb") as fd:
            tv = ts.times.view().reshape((-1, 1))

            if len(ts.data.shape) == 1:
                dv = ts.data.view().reshape((ts.times.shape[0], -1))
            else:
                dv = ts.data

            result = numpy.concatenate((tv, dv), axis=1)
            fd.write((",".join(map(str, header)) + "\n").encode(csv_file_encoding))
            numpy.savetxt(fd, result, delimiter=',', newline="\n", fmt="%lu")

        if ts.raw:
            raw_path = DB_paths.ts.format(**ts.source(tag=ts.raw_tag).__dict__)
            self.storage.put_raw(ts.raw, raw_path)

    def put_extra(self, data: bytes, source: DataSource) -> None:
        self.storage.put(data, DB_paths.ts.format(**source.__dict__))

    def put_stat(self, data: StatProps, source: DataSource) -> None:
        self.storage.put(data, DB_paths.stat.format(**source.__dict__))

    # return path to file to be inserted into report
    def put_plot_file(self, data: bytes, source: DataSource) -> str:
        path = DB_paths.plot.format(**source.__dict__)
        return cast(str, self.storage.put_raw(data, path))

    def put_report(self, report: str, name: str) -> str:
        return self.storage.put_raw(report.encode("utf8"), DB_paths.report_root + name)

    def append_sensor(self, data: numpy.array, ds: DataSource, units: str) -> None:
        if ds.metric == 'collected_at':
            path = DB_paths.sensor_time
            metrics_fqn = 'collected_at'
        else:
            path = DB_paths.sensor_data
            metrics_fqn = ds.metric_fqdn
        self.storage.append([ds.node_id, metrics_fqn, units], data, path.format(**ds.__dict__))

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
            assert is_file
            groups = groups.copy()
            groups.update(filters)
            ds = DataSource(suite_id=suite.storage_id,
                            job_id=job.storage_id,
                            node_id=groups["node_id"],
                            sensor=groups["sensor"],
                            dev=None,
                            metric=groups["metric"],
                            tag=groups["tag"])
            yield self.load_ts(ds, path)

    def iter_sensors(self, node_id: str = None, sensor: str = None, dev: str = None, metric: str = None) -> \
            Iterator[Tuple[str, Dict[str, str]]]:

        path = fill_path(DB_paths.sensor_data_r, node_id=node_id, sensor=sensor, dev=dev, metric=metric)
        for is_file, path, groups in self.iter_paths(path):
            assert is_file
            yield path, groups


