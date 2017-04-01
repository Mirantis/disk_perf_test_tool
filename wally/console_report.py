import numpy

from cephlib.common import float2str

from . import texttable
from .hlstorage import ResultStorage
from .stage import Stage, StepOrder
from .test_run_class import TestRun
from .suits.io.fio import FioTest
from .statistic import calc_norm_stat_props, calc_histo_stat_props
from .suits.io.fio_hist import get_lat_vals

class ConsoleReportStage(Stage):

    priority = StepOrder.REPORT

    def run(self, ctx: TestRun) -> None:
        rstorage = ResultStorage(ctx.storage)
        for suite in rstorage.iter_suite(FioTest.name):
            table = texttable.Texttable(max_width=200)

            table.header(["Description", "IOPS ~ Dev", 'Skew/Kurt', 'lat med', 'lat 95'])
            table.set_cols_align(('l', 'r', 'r', 'r', 'r'))

            for job in sorted(rstorage.iter_job(suite), key=lambda job: job.params):
                bw_ts, = list(rstorage.iter_ts(suite, job, metric='bw'))
                props = calc_norm_stat_props(bw_ts)
                avg_iops = props.average // job.params.params['bsize']
                iops_dev = props.deviation // job.params.params['bsize']

                lat_ts, = list(rstorage.iter_ts(suite, job, metric='lat'))
                bins_edges = numpy.array(get_lat_vals(lat_ts.data.shape[1]), dtype='float32') / 1000  # convert us to ms
                lat_props = calc_histo_stat_props(lat_ts, bins_edges)

                table.add_row([job.params.summary,
                               "{} ~ {}".format(float2str(avg_iops), float2str(iops_dev)),
                               "{}/{}".format(float2str(props.skew), float2str(props.kurt)),
                               float2str(lat_props.perc_50), float2str(lat_props.perc_95)])

            print(table.draw())
