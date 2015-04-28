import texttable

from wally.utils import ssize_to_b
from wally.suits.io.agent import get_test_summary
from wally.statistic import med_dev, round_deviation, round_3_digit


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
    tab = texttable.Texttable(max_width=120)
    tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER)
    tab.set_cols_align(["l", "r", "r", "r", "r", "r", "r"])

    prev_k = None
    vm_count = test_set['__test_meta__']['testnodes_count']
    items = sorted(test_set['res'].items(), key=key_func)
    header = ["Description", "iops\ncum", "KiBps\ncum",
              "iops\nper vm", "KiBps\nper vm", "Cnf\n%", "lat\nms"]

    for test_name, data in items:

        curr_k = key_func((test_name, data))[:3]

        if prev_k is not None:
            if prev_k != curr_k:
                tab.add_row(
                    ["--------", "-----", "------",
                     "-----", "------", "---", "-----"])

        prev_k = curr_k

        descr = get_test_summary(data)

        iops, _ = round_deviation(med_dev(data['iops']))
        bw, bwdev = round_deviation(med_dev(data['bw']))

        # 3 * sigma
        if 0 == bw:
            assert 0 == bwdev
            dev_perc = 0
        else:
            dev_perc = int((bwdev * 300) / bw)

        med_lat, _ = round_deviation(med_dev(data['lat']))
        med_lat = int(med_lat) // 1000

        iops = round_3_digit(iops)
        bw = round_3_digit(bw)
        iops_cum = round_3_digit(iops * vm_count)
        bw_cum = round_3_digit(bw * vm_count)
        med_lat = round_3_digit(med_lat)

        params = (descr, int(iops_cum), int(bw_cum),
                  int(iops), int(bw), dev_perc, med_lat)
        tab.add_row(params)

    tab.header(header)

    return tab.draw()
