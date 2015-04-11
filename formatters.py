import itertools
from collections import defaultdict

import texttable

from statistic import med_dev

# [{u'__meta__': {u'raw_cfg': u'[writetest * 2]\ngroup_reporting\nnumjobs=4\nwait_for_previous\nramp_time=5\nblocksize=4k\nfilename={FILENAME}\nrw=randwrite\ndirect=1\nbuffered=0\niodepth=1\nsize=100Mb\nruntime=10\ntime_based\n\n[readtest * 2]\ngroup_reporting\nnumjobs=4\nwait_for_previous\nramp_time=5\nblocksize=4k\nfilename={FILENAME}\nrw=randread\ndirect=1\nbuffered=0\niodepth=1\nsize=100Mb\nruntime=10\ntime_based\n'},
#   u'res': {u'readtest': {u'action': u'randread',
#                          u'blocksize': u'4k',
#                          u'bw_mean': [349.61, 276.54],
#                          u'clat': [11987.16, 15235.08],
#                          u'concurence': 4,
#                          u'direct_io': True,
#                          u'iops': [316, 251],
#                          u'jobname': u'readtest',
#                          u'lat': [11987.52, 15235.46],
#                          u'slat': [0.0, 0.0],
#                          u'sync': False,
#                          u'timings': [u'10', u'5']},
#            u'writetest': {u'action': u'randwrite',
#                           u'blocksize': u'4k',
#                           u'bw_mean': [72.03, 61.84],
#                           u'clat': [113525.86, 152836.42],
#                           u'concurence': 4,
#                           u'direct_io': True,
#                           u'iops': [35, 23],
#                           u'jobname': u'writetest',
#                           u'lat': [113526.31, 152836.89],
#                           u'slat': [0.0, 0.0],
#                           u'sync': False,
#                           u'timings': [u'10', u'5']}}},
#  {u'__meta__': {u'raw_cfg': u'[writetest * 2]\ngroup_reporting\nnumjobs=4\nwait_for_previous\nramp_time=5\nblocksize=4k\nfilename={FILENAME}\nrw=randwrite\ndirect=1\nbuffered=0\niodepth=1\nsize=100Mb\nruntime=10\ntime_based\n\n[readtest * 2]\ngroup_reporting\nnumjobs=4\nwait_for_previous\nramp_time=5\nblocksize=4k\nfilename={FILENAME}\nrw=randread\ndirect=1\nbuffered=0\niodepth=1\nsize=100Mb\nruntime=10\ntime_based\n'},
#   u'res': {u'readtest': {u'action': u'randread',
#                          u'blocksize': u'4k',
#                          u'bw_mean': [287.62, 280.76],
#                          u'clat': [15437.57, 14741.65],
#                          u'concurence': 4,
#                          u'direct_io': True,
#                          u'iops': [258, 271],
#                          u'jobname': u'readtest',
#                          u'lat': [15437.94, 14742.04],
#                          u'slat': [0.0, 0.0],
#                          u'sync': False,
#                          u'timings': [u'10', u'5']},
#            u'writetest': {u'action': u'randwrite',
#                           u'blocksize': u'4k',
#                           u'bw_mean': [71.18, 61.62],
#                           u'clat': [116382.95, 153486.81],
#                           u'concurence': 4,
#                           u'direct_io': True,
#                           u'iops': [32, 22],
#                           u'jobname': u'writetest',
#                           u'lat': [116383.44, 153487.27],
#                           u'slat': [0.0, 0.0],
#                           u'sync': False,
#                           u'timings': [u'10', u'5']}}}]


def get_test_descr(data):
    rw = {"randread": "rr",
          "randwrite": "rw",
          "read": "sr",
          "write": "sw"}[data["action"]]

    if data["direct_io"]:
        sync_mode = 'd'
    elif data["sync"]:
        sync_mode = 's'
    else:
        sync_mode = 'a'

    th_count = int(data['concurence'])

    return "{0}{1}{2}_th{3}".format(rw, sync_mode,
                                    data['blocksize'], th_count)


def format_results_for_console(test_set):
    data_for_print = []
    tab = texttable.Texttable()
    tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER | tab.HLINES)
    tab.set_cols_align(["l", "r", "r", "r", "r"])

    for test_name, data in test_set['res'].items():
        descr = get_test_descr(data)

        iops, _ = med_dev(data['iops'])
        bw, bwdev = med_dev(data['bw_mean'])

        # 3 * sigma
        dev_perc = int((bwdev * 300) / bw)

        params = (descr, int(iops), int(bw), dev_perc,
                  int(med_dev(data['lat'])[0]) // 1000)
        data_for_print.append(params)

    header = ["Description", "IOPS", "BW KBps", "Dev * 3 %", "LAT ms"]
    tab.add_row(header)
    tab.header = header

    map(tab.add_row, data_for_print)

    return tab.draw()


# def format_pgbench_stat(res):
#     """
#     Receives results in format:
#     "<num_clients> <num_transactions>: <tps>
#      <num_clients> <num_transactions>: <tps>
#      ....
#     "
#     """
#     if res:
#         data = {}
#         grouped_res = itertools.groupby(res, lambda x: x[0])
#         for key, group in grouped_res:
#             results = list(group)
#             sum_res = sum([r[1] for r in results])
#             mean = sum_res/len(results)
#             sum_sq = sum([(r[1] - mean) ** 2 for r in results])
#             if len(results) > 1:
#                 dev = (sum_sq / (len(results) - 1))
#             else:
#                 dev = 0
#             data[key] = (mean, dev)
#         return data
