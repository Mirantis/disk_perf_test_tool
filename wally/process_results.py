# put all result preprocessing here
# selection, aggregation

import logging


from .stage import Stage, StepOrder
from .test_run_class import TestRun
from .statistic import calc_norm_stat_props, calc_histo_stat_props
from .result_classes import TestJobConfig
from .suits.itest import ResultStorage
from .suits.io.fio_hist import get_lat_vals, expected_lat_bins
from .utils import StopTestError

logger = logging.getLogger("wally")

import matplotlib

# have to be before pyplot import to avoid tkinter(default graph frontend) import error
matplotlib.use('svg')

import matplotlib.pyplot as plt


class CalcStatisticStage(Stage):
    priority = StepOrder.TEST + 1

    def run(self, ctx: TestRun) -> None:
        rstorage = ResultStorage(ctx.storage, TestJobConfig)

        for suite_cfg, path in rstorage.list_suites():
            if suite_cfg.test_type != 'fio':
                continue

            for job_cfg, path, _ in rstorage.list_jobs_in_suite(path):
                results = {}
                for node_id, dev, sensor_name in rstorage.list_ts_in_job(path):
                    ts = rstorage.load_ts(path, node_id, dev, sensor_name)
                    if dev == 'fio' and sensor_name == 'lat':
                        if ts.second_axis_size != expected_lat_bins:
                            logger.error("Sensor %s.%s on node %s has" +
                                         "second_axis_size=%s. Can only process sensors with second_axis_size=%s.",
                                         dev, sensor_name, node_id, ts.second_axis_size, expected_lat_bins)
                            continue
                        ts.bins_edges = get_lat_vals(ts.second_axis_size)
                        stat_prop = calc_histo_stat_props(ts)

                    elif ts.second_axis_size != 1:
                        logger.warning("Sensor %s.%s on node %s provide 2D data with " +
                                       "ts.second_axis_size=%s. Can't process it.",
                                       dev, sensor_name, node_id, ts.second_axis_size)
                        continue
                    else:
                        stat_prop = calc_norm_stat_props(ts)

                    results[(node_id, dev, sensor_name)] = stat_prop

        raise StopTestError()
