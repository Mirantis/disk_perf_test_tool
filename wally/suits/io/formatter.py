import texttable

from wally.utils import ssize2b
from wally.statistic import round_3_digit
from wally.suits.io.agent import get_test_summary


def key_func(k_data):
    name, data = k_data
    return (data['rw'],
            data['sync_mode'],
            ssize2b(data['blocksize']),
            data['concurence'],
            name)


def format_results_for_console(test_set, dinfo):
    """
    create a table with io performance report
    for console
    """
    tab = texttable.Texttable(max_width=120)
    tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER)
    tab.set_cols_align(["l", "l", "r", "r", "r", "r", "r", "r"])

    items = sorted(test_set['res'].items(), key=key_func)

    prev_k = None
    vm_count = test_set['__test_meta__']['testnodes_count']
    header = ["Name", "Description", "iops\ncum", "KiBps\ncum",
              "Cnf\n95%", "iops\nper vm", "KiBps\nper vm", "lat\nms"]

    for test_name, data in items:

        curr_k = key_func((test_name, data))[:3]

        if prev_k is not None:
            if prev_k != curr_k:
                tab.add_row(
                    ["-------", "--------", "-----", "------",
                     "---", "------", "---", "-----"])

        prev_k = curr_k

        descr = get_test_summary(data)
        test_dinfo = dinfo[test_name]

        iops, _ = test_dinfo.iops.rounded_average_conf()
        bw, bw_conf = test_dinfo.bw.rounded_average_conf()
        conf_perc = int(round(bw_conf * 100 / bw))

        lat, _ = test_dinfo.lat.rounded_average_conf()
        lat = round_3_digit(int(lat) // 1000)

        iops_per_vm = round_3_digit(iops / float(vm_count))
        bw_per_vm = round_3_digit(bw / float(vm_count))

        iops = round_3_digit(iops)
        bw = round_3_digit(bw)

        params = (test_name.split('_', 1)[0],
                  descr, int(iops), int(bw), str(conf_perc),
                  int(iops_per_vm), int(bw_per_vm), lat)
        tab.add_row(params)

    tab.header(header)

    return tab.draw()
