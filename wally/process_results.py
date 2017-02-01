# put all result preprocessing here
# # selection, aggregation
#
# from io import BytesIO
# import logging
# from typing import Any
#
# from .stage import Stage, StepOrder
# from .test_run_class import TestRun
# from .statistic import calc_norm_stat_props, calc_histo_stat_props
# from .result_classes import StatProps, DataSource, TimeSeries
# from .hlstorage import ResultStorage
# from .suits.io.fio_hist import get_lat_vals, expected_lat_bins
# from .suits.io.fio import FioTest
# from .utils import StopTestError
#
# import matplotlib
# matplotlib.use('svg')
# import matplotlib.pyplot as plt
#
#
# logger = logging.getLogger("wally")
#
#
# class CalcStatisticStage(Stage):
#     priority = StepOrder.TEST + 1
#
#     def run(self, ctx: TestRun) -> None:
#         rstorage = ResultStorage(ctx.storage)
#
#         for suite in rstorage.iter_suite(FioTest.name):
#             for job in rstorage.iter_job(suite):
#                 for ts in rstorage.iter_ts(suite, job):
#                     if ts.source.sensor == 'lat':
#                         if ts.data.shape[1] != expected_lat_bins:
#                             logger.error("Sensor %s.%s on node %s has" +
#                                          "shape=%s. Can only process sensors with shape=[X,%s].",
#                                          ts.source.dev, ts.source.sensor, ts.source.node_id,
#                                          ts.data.shape, expected_lat_bins)
#                             continue
#
#                         ts.bins_edges = get_lat_vals(ts.data.shape[1])
#                         stat_prop = calc_histo_stat_props(ts)  # type: StatProps
#
#                     elif len(ts.data.shape) != 1:
#                         logger.warning("Sensor %s.%s on node %s provide 2+D data. Can't process it.",
#                                        ts.source.dev, ts.source.sensor, ts.source.node_id)
#                         continue
#                     else:
#                         stat_prop = calc_norm_stat_props(ts)
#
#         raise StopTestError()
