import texttable

from utils import ssize_to_b
from statistic import med_dev
from disk_perf_test_tool.tests.disk_test_agent import get_test_summary


def key_func(k_data):
    _, data = k_data

    bsz = ssize_to_b(data['blocksize'])
    tp = data['rw']
    return tp, data['sync_mode'], bsz, data['concurence']


def format_results_for_console(test_set):
    data_for_print = []
    tab = texttable.Texttable()
    tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER)
    tab.set_cols_align(["l", "r", "r", "r", "r"])

    items = sorted(test_set['res'].items(), key=key_func)
    prev_k = None

    for test_name, data in items:
        curr_k = key_func((test_name, data))[:3]

        if prev_k is not None:
            if prev_k != curr_k:
                data_for_print.append(["---"] * 5)

        prev_k = curr_k

        descr = get_test_summary(data)

        iops, _ = med_dev(data['iops'])
        bw, bwdev = med_dev(data['bw'])

        # 3 * sigma
        dev_perc = int((bwdev * 300) / bw)

        params = (descr, int(iops), int(bw), dev_perc,
                  int(med_dev(data['lat'])[0]) // 1000)
        data_for_print.append(params)

    header = ["Description", "IOPS", "BW KBps", "Dev * 3 %", "LAT ms"]
    tab.header(header)

    map(tab.add_row, data_for_print)

    return tab.draw()
