import re
import os
import array
import struct
import logging
from typing import cast, Iterator, Tuple, Type, Dict, Set, List, Optional

import numpy

from .result_classes import (TestSuiteConfig, TestJobConfig, TimeSeries, DataSource,
                             StatProps, IResultStorage)
from .storage import Storage
from .utils import StopTestError
from .suits.all_suits import all_suits


logger = logging.getLogger('wally')


class DB_re:
    node_id = r'\d+.\d+.\d+.\d+:\d+'
    job_id = r'[-a-zA-Z0-9]+_\d+'
    sensor = r'[a-z_]+'
    dev = r'[-a-zA-Z0-9_]+'
    suite_id = r'[a-z]+_\d+'
    tag = r'[a-z_.]+'


class DB_paths:
    suite_cfg_r = r'results/{suite_id}_info\.yml'
    suite_cfg = suite_cfg_r.replace("\\.", '.')

    job_cfg_r = r'results/{suite_id}\.{job_id}/info\.yml'
    job_cfg = job_cfg_r.replace("\\.", '.')

    job_extra_r = r'results/{suite_id}\.{job_id}/{node_id}/{dev}\.{sensor}\.{tag}'
    job_extra = job_extra_r.replace("\\.", '.')

    ts_r = r'results/{suite_id}\.{job_id}/{node_id}/{dev}\.{sensor}\.{tag}'
    ts = ts_r.replace("\\.", '.')

    stat_r = r'results/{suite_id}\.{job_id}/{node_id}/{dev}\.{sensor}\.{tag}'
    stat = stat_r.replace("\\.", '.')

    plot_r = r'report/{suite_id}\.{job_id}/{node_id}/{dev}\.{sensor}\.{tag}'
    plot = plot_r.replace("\\.", '.')

    report = r'report/'


DB_rr = {name: r"(?P<{}>{})".format(name, rr) for name, rr in DB_re.__dict__.items() if not name.startswith("__")}


class ResultStorage(IResultStorage):
    # TODO: check that all path components match required patterns

    ts_header_format = "!IIIcc"
    ts_arr_tag = 'bin'
    ts_raw_tag = 'txt'

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def sync(self) -> None:
        self.storage.sync()

    def put_or_check_suite(self, suite: TestSuiteConfig) -> None:
        path = DB_paths.suite_cfg.format(suite_id=suite.storage_id)
        if path in self.storage:
            db_cfg = self.storage.get(path)
            if db_cfg != suite:
                logger.error("Current suite %s config is not equal to found in storage at %s", suite.test_type, path)
                raise StopTestError()

        self.storage.put(suite, path)

    def put_job(self, suite: TestSuiteConfig, job: TestJobConfig) -> None:
        path = DB_paths.job_cfg.format(suite_id=suite.storage_id, job_id=job.storage_id)
        self.storage.put(job, path)

    def put_ts(self, ts: TimeSeries) -> None:
        data = cast(List[int], ts.data)
        times = cast(List[int], ts.times)

        if len(data) % ts.second_axis_size != 0:
            logger.error("Time series data size(%s) is not propotional to second_axis_size(%s).",
                         len(data), ts.second_axis_size)
            raise StopTestError()

        if len(data) // ts.second_axis_size != len(times):
            logger.error("Unbalanced data and time srray sizes. %s", ts)
            raise StopTestError()

        bin_path = DB_paths.ts.format(**ts.source(tag=self.ts_arr_tag).__dict__)

        with self.storage.get_fd(bin_path, "cb") as fd:
            header = struct.pack(self.ts_header_format,
                                 ts.second_axis_size,
                                 len(data),
                                 len(times),
                                 ts.data.typecode.encode("ascii"),
                                 ts.times.typecode.encode("ascii"))
            fd.write(header)
            ts.data.tofile(fd)  # type: ignore
            ts.times.tofile(fd)  # type: ignore

        if ts.raw:
            raw_path = DB_paths.ts.format(**ts.source(tag=self.ts_raw_tag).__dict__)
            self.storage.put_raw(ts.raw, raw_path)

    def put_extra(self, data: bytes, source: DataSource) -> None:
        path = DB_paths.job_cfg.format(**source.__dict__)
        self.storage.put_raw(data, path)

    def put_stat(self, data: StatProps, source: DataSource) -> None:
        path = DB_paths.stat.format(**source.__dict__)
        self.storage.put(data, path)

    def get_stat(self, stat_cls: Type[StatProps], source: DataSource) -> StatProps:
        path = DB_paths.stat.format(**source.__dict__)
        return self.storage.load(stat_cls, path)

    def iter_paths(self, path_glob) -> Iterator[Tuple[bool, str, Dict[str, str]]]:
        path = path_glob.format(**DB_rr).split("/")
        yield from self.storage._iter_paths("", path, {})

    def iter_suite(self, suite_type: str = None) -> Iterator[TestSuiteConfig]:
        for is_file, suite_info_path, groups in self.iter_paths(DB_paths.suite_cfg_r):
            assert is_file
            suite = cast(TestSuiteConfig, self.storage.load(TestSuiteConfig, suite_info_path))
            assert suite.storage_id == groups['suite_id']
            if not suite_type or suite.test_type == suite_type:
                yield suite

    def iter_job(self, suite: TestSuiteConfig) -> Iterator[TestJobConfig]:
        job_glob = DB_paths.job_cfg_r.replace('{suite_id}', suite.storage_id)
        job_config_cls = all_suits[suite.test_type].job_config_cls

        for is_file, path, groups in self.iter_paths(job_glob):
            assert is_file
            job = cast(TestJobConfig, self.storage.load(job_config_cls, path))
            assert job.storage_id == groups['job_id']
            yield job

    def iter_datasource(self, suite: TestSuiteConfig, job: TestJobConfig) -> Iterator[Tuple[DataSource, Dict[str, str]]]:
        ts_glob = DB_paths.ts_r.replace('{suite_id}', suite.storage_id).replace('{job_id}', job.storage_id)
        ts_found = {}  # type: Dict[Tuple[str, str, str], Dict[str, str]]

        for is_file, path, groups in self.iter_paths(ts_glob):
            assert is_file
            key = (groups['node_id'], groups['dev'], groups['sensor'])
            ts_found.setdefault(key, {})[groups['tag']] = path

        for (node_id, dev, sensor), tag2path in ts_found.items():
            if self.ts_arr_tag in tag2path:
                yield DataSource(suite_id=suite.storage_id,
                                 job_id=job.storage_id,
                                 node_id=node_id,
                                 dev=dev, sensor=sensor, tag=None), tag2path

    def load_ts(self, ds: DataSource, path: str) -> TimeSeries:
        with self.storage.get_fd(path, "rb") as fd:
            header = fd.read(struct.calcsize(self.ts_header_format))
            second_axis_size, data_sz, time_sz, data_typecode, time_typecode = \
                struct.unpack(self.ts_header_format, header)

            data = array.array(data_typecode.decode("ascii"))
            times = array.array(time_typecode.decode("ascii"))

            data.fromfile(fd, data_sz)  # type: ignore
            times.fromfile(fd, time_sz)  # type: ignore

        return TimeSeries("{}.{}".format(ds.dev, ds.sensor),
                          raw=None,
                          data=numpy.array(data, dtype=numpy.dtype('float32')),
                          times=numpy.array(times),
                          second_axis_size=second_axis_size,
                          source=ds)

    def iter_ts(self, suite: TestSuiteConfig, job: TestJobConfig, **filters) -> Iterator[TimeSeries]:
        for ds, tag2path in self.iter_datasource(suite, job):
            for name, val in filters.items():
                if val != getattr(ds, name):
                    break
            else:
                ts = self.load_ts(ds, tag2path[self.ts_arr_tag])
                if self.ts_raw_tag in tag2path:
                    ts.raw = self.storage.get_raw(tag2path[self.ts_raw_tag])

                yield ts

    # return path to file to be inserted into report
    def put_plot_file(self, data: bytes, source: DataSource) -> str:
        path = DB_paths.plot.format(**source.__dict__)
        return cast(str, self.storage.put_raw(data, path))

    def check_plot_file(self, source: DataSource) -> Optional[str]:
        path = DB_paths.plot.format(**source.__dict__)
        fpath = self.storage.resolve_raw(path)
        if os.path.exists(fpath):
            return fpath
        return None

    def put_report(self, report: str, name: str) -> str:
        return self.storage.put_raw(report.encode("utf8"), DB_paths.report + name)
