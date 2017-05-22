import os
import json
import pprint
import logging
from typing import cast, Iterator, Tuple, Type, Optional, Any, Union, List

import numpy

from cephlib.wally_storage import WallyDB
from cephlib.sensor_storage import SensorStorage
from cephlib.statistic import StatProps
from cephlib.numeric_types import DataSource, TimeSeries
from cephlib.node import NodeInfo

from .suits.job import JobConfig
from .result_classes import SuiteConfig, IWallyStorage
from .utils import StopTestError
from .suits.all_suits import all_suits

from cephlib.storage import Storage

logger = logging.getLogger('wally')


def fill_path(path: str, **params) -> str:
    for name, val in params.items():
        if val is not None:
            path = path.replace("{" + name + "}", val)
    return path


class WallyStorage(IWallyStorage, SensorStorage):
    def __init__(self, storage: Storage) -> None:
        SensorStorage.__init__(self, storage, WallyDB)

    def flush(self) -> None:
        self.storage.flush()

    # -------------  CHECK DATA IN STORAGE  ----------------------------------------------------------------------------
    def check_plot_file(self, source: DataSource) -> Optional[str]:
        path = self.db_paths.plot.format(**source.__dict__)
        fpath = self.storage.get_fname(self.db_paths.report_root + path)
        return path if os.path.exists(fpath) else None

    # -------------   PUT DATA INTO STORAGE   --------------------------------------------------------------------------
    def put_or_check_suite(self, suite: SuiteConfig) -> None:
        path = self.db_paths.suite_cfg.format(suite_id=suite.storage_id)
        if path in self.storage:
            db_cfg = self.storage.load(SuiteConfig, path)
            if db_cfg != suite:
                logger.error("Current suite %s config is not equal to found in storage at %s", suite.test_type, path)
                logger.debug("Current: \n%s\nStorage:\n%s", pprint.pformat(db_cfg), pprint.pformat(suite))
                raise StopTestError()
        else:
            self.storage.put(suite, path)

    def put_job(self, suite: SuiteConfig, job: JobConfig) -> None:
        path = self.db_paths.job_cfg.format(suite_id=suite.storage_id, job_id=job.storage_id)
        self.storage.put(job, path)

    def put_extra(self, data: bytes, source: DataSource) -> None:
        self.storage.put_raw(data, self.db_paths.ts.format(**source.__dict__))

    def put_stat(self, data: StatProps, source: DataSource) -> None:
        self.storage.put(data, self.db_paths.stat.format(**source.__dict__))

    # return path to file to be inserted into report
    def put_plot_file(self, data: bytes, source: DataSource) -> str:
        path = self.db_paths.plot.format(**source.__dict__)
        self.storage.put_raw(data, self.db_paths.report_root + path)
        return path

    def put_report(self, report: str, name: str) -> str:
        return self.storage.put_raw(report.encode(self.csv_file_encoding), self.db_paths.report_root + name)

    def put_txt_report(self, suite: SuiteConfig, report: str) -> None:
        path = self.db_paths.txt_report.format(suite_id=suite.storage_id)
        self.storage.put_raw(report.encode('utf8'), path)

    def put_job_info(self, suite: SuiteConfig, job: JobConfig, key: str, data: Any) -> None:
        path = self.db_paths.job_extra.format(suite_id=suite.storage_id, job_id=job.storage_id, name=key)
        if isinstance(data, bytes):
            self.storage.put_raw(data, path)
        else:
            self.storage.put(data, path)

    # -------------   GET DATA FROM STORAGE   --------------------------------------------------------------------------

    def get_stat(self, stat_cls: Type[StatProps], source: DataSource) -> StatProps:
        return self.storage.load(stat_cls, self.db_paths.stat.format(**source.__dict__))

    def get_txt_report(self, suite: SuiteConfig) -> Optional[str]:
        path = self.db_paths.txt_report.format(suite_id=suite.storage_id)
        if path in self.storage:
            return self.storage.get_raw(path).decode('utf8')
        return None

    def get_job_info(self, suite: SuiteConfig, job: JobConfig, key: str) -> Any:
        path = self.db_paths.job_extra.format(suite_id=suite.storage_id, job_id=job.storage_id, name=key)
        return self.storage.get(path, None)
    # -------------   ITER OVER STORAGE   ------------------------------------------------------------------------------

    def iter_suite(self, suite_type: str = None) -> Iterator[SuiteConfig]:
        for is_file, suite_info_path, groups in self.iter_paths(self.db_paths.suite_cfg_r):
            assert is_file
            suite = self.storage.load(SuiteConfig, suite_info_path)
            assert suite.storage_id == groups['suite_id']
            if not suite_type or suite.test_type == suite_type:
                yield suite

    def iter_job(self, suite: SuiteConfig) -> Iterator[JobConfig]:
        job_glob = fill_path(self.db_paths.job_cfg_r, suite_id=suite.storage_id)
        job_config_cls = all_suits[suite.test_type].job_config_cls
        for is_file, path, groups in self.iter_paths(job_glob):
            assert is_file
            job = cast(JobConfig, self.storage.load(job_config_cls, path))
            assert job.storage_id == groups['job_id']
            yield job

    def load_nodes(self) -> List[NodeInfo]:
        try:
            return self.storage.other_caches['wally']['nodes']
        except KeyError:
            nodes = self.storage.load_list(NodeInfo, WallyDB.all_nodes)
            if WallyDB.nodes_params in self.storage:
                params = json.loads(self.storage.get_raw(WallyDB.nodes_params).decode('utf8'))
                for node in nodes:
                    node.params = params.get(node.node_id, {})
            self.storage.other_caches['wally']['nodes'] = nodes
            return nodes

    #  -----------------  TS  ------------------------------------------------------------------------------------------
    def get_ts(self, ds: DataSource) -> TimeSeries:
        path = self.db_paths.ts.format_map(ds.__dict__)
        (units, time_units), header2, content = self.storage.get_array(path)
        times = content[:,0].copy()
        data = content[:,1:]

        if data.shape[1] == 1:
            data.shape = (data.shape[0],)

        return TimeSeries(data=data, times=times, source=ds, units=units, time_units=time_units, histo_bins=header2)

    def put_ts(self, ts: TimeSeries) -> None:
        assert ts.data.dtype == ts.times.dtype, "Data type {!r} != time type {!r}".format(ts.data.dtype, ts.times.dtype)
        assert ts.data.dtype.kind == 'u', "Only unsigned ints are accepted"
        assert ts.source.tag == self.ts_arr_tag, \
            "Incorrect source tag == {!r}, must be {!r}".format(ts.source.tag, self.ts_arr_tag)

        if ts.source.metric == 'lat':
            assert len(ts.data.shape) == 2, "Latency should be 2d array"
            assert ts.histo_bins is not None, "Latency should have histo_bins field not empty"

        csv_path = self.db_paths.ts.format_map(ts.source.__dict__)
        header = [ts.units, ts.time_units]

        tv = ts.times.view().reshape((-1, 1))

        if len(ts.data.shape) == 1:
            dv = ts.data.view().reshape((ts.times.shape[0], -1))
        else:
            dv = ts.data

        result = numpy.concatenate((tv, dv), axis=1)
        self.storage.put_array(csv_path, result, header, header2=ts.histo_bins, append_on_exists=False)

    def iter_ts(self, **ds_parts: str) -> Iterator[DataSource]:
        return self.iter_objs(self.db_paths.ts_r, **ds_parts)
