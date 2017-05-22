import logging
from typing import cast, Iterator, List, Union

import numpy

from cephlib.common import float2str
from cephlib.texttable import Texttable
from cephlib.statistic import calc_norm_stat_props, calc_histo_stat_props

from .stage import Stage, StepOrder
from .test_run_class import TestRun
from .result_classes import SuiteConfig
from .suits.io.fio import FioTest
from .suits.io.fio_job import FioJobParams
from .suits.io.fio_hist import get_lat_vals
from .data_selectors import get_aggregated
from .result_storage import IWallyStorage


logger = logging.getLogger("wally")



console_report_headers = ["Description", "IOPS ~ Dev", "BW, MiBps", 'Skew/Kurt', 'lat med, ms', 'lat 95, ms']
console_report_align = ['l', 'r', 'r', 'r', 'r', 'r']

def get_console_report_table(suite: SuiteConfig, rstorage: IWallyStorage) -> List[Union[List[str], Texttable.HLINE]]:
    table = []  # type: List[Union[List[str], Texttable.HLINE]]
    prev_params = None
    for job in sorted(rstorage.iter_job(suite), key=lambda job: job.params):
        fparams = cast(FioJobParams, job.params)
        fparams['qd'] = None

        if prev_params is not None and fparams.char_tpl != prev_params:
            table.append(Texttable.HLINE)

        prev_params = fparams.char_tpl

        bw_ts = get_aggregated(rstorage, suite.storage_id, job.storage_id, metric='bw',
                               trange=job.reliable_info_range_s)
        props = calc_norm_stat_props(bw_ts)
        avg_iops = props.average // job.params.params['bsize']
        iops_dev = props.deviation // job.params.params['bsize']

        lat_ts = get_aggregated(rstorage, suite.storage_id, job.storage_id, metric='lat',
                                trange=job.reliable_info_range_s)
        bins_edges = numpy.array(get_lat_vals(lat_ts.data.shape[1]), dtype='float32') / 1000  # convert us to ms
        lat_props = calc_histo_stat_props(lat_ts, bins_edges)
        table.append([job.params.summary,
                      "{:>6s} ~ {:>6s}".format(float2str(avg_iops), float2str(iops_dev)),
                      float2str(props.average / 1024),  # Ki -> Mi
                      "{:>5.1f}/{:>5.1f}".format(props.skew, props.kurt),
                      float2str(lat_props.perc_50), float2str(lat_props.perc_95)])
    return table


class ConsoleReportStage(Stage):

    priority = StepOrder.REPORT

    def run(self, ctx: TestRun) -> None:
        for suite in ctx.rstorage.iter_suite(FioTest.name):
            table = Texttable(max_width=200)
            table.set_deco(Texttable.VLINES | Texttable.BORDER | Texttable.HEADER)
            tbl = ctx.rstorage.get_txt_report(suite)
            if tbl is None:
                table.header(console_report_headers)
                table.set_cols_align(console_report_align)
                for line in get_console_report_table(suite, ctx.rstorage):
                    table.add_row(line)
                tbl = table.draw()
                ctx.rstorage.put_txt_report(suite, tbl)
            print(tbl)
