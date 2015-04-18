import texttable

from wally.utils import ssize_to_b
from wally.statistic import med_dev
from wally.suits.io.agent import get_test_summary


def key_func(k_data):
    _, data = k_data

    return (data['rw'],
            data['sync_mode'],
            ssize_to_b(data['blocksize']),
            data['concurence'])


def format_results_for_console(test_set):
    """
    create a table with io performance report
    for console
    """
    tab = texttable.Texttable()
    tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER)
    tab.set_cols_align(["l", "r", "r", "r", "r"])

    prev_k = None
    items = sorted(test_set['res'].items(), key=key_func)

    for test_name, data in items:
        curr_k = key_func((test_name, data))[:3]

        if prev_k is not None:
            if prev_k != curr_k:
                tab.add_row(["---"] * 5)

        prev_k = curr_k

        descr = get_test_summary(data)

        iops, _ = med_dev(data['iops'])
        bw, bwdev = med_dev(data['bw'])

        # 3 * sigma
        dev_perc = int((bwdev * 300) / bw)

        params = (descr, int(iops), int(bw), dev_perc,
                  int(med_dev(data['lat'])[0]) // 1000)
        tab.add_row(params)

    header = ["Description", "IOPS", "BW KiBps", "Dev * 3 %", "clat ms"]
    tab.header(header)

    return tab.draw()
