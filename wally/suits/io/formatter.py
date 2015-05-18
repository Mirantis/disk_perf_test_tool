import texttable

from wally.utils import ssize2b
from wally.statistic import round_3_digit
from .fio_task_parser import get_test_summary, get_test_sync_mode


def key_func(data):
    p = data.params.vals

    th_count = data.params.vals.get('numjobs')

    if th_count is None:
        th_count = data.params.vals.get('concurence', 1)

    return (p['rw'],
            get_test_sync_mode(data.params),
            ssize2b(p['blocksize']),
            int(th_count) * data.testnodes_count,
            data.name)


def format_results_for_console(dinfo):
    """
    create a table with io performance report
    for console
    """
    tab = texttable.Texttable(max_width=120)
    tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER)
    tab.set_cols_align(["l", "l", "r", "r", "r", "r", "r", "r", "r"])

    items = sorted(dinfo.values(), key=key_func)

    prev_k = None
    header = ["Name", "Description", "iops\ncum", "KiBps\ncum",
              "Cnf\n95%", "Dev%", "iops\nper vm", "KiBps\nper vm", "lat\nms"]

    for data in items:

        curr_k = key_func(data)[:3]

        if prev_k is not None:
            if prev_k != curr_k:
                tab.add_row(
                    ["-------", "-----------", "-----", "------",
                     "---", "----", "------", "---", "-----"])

        prev_k = curr_k

        test_dinfo = dinfo[(data.name, data.summary)]

        iops, _ = test_dinfo.iops.rounded_average_conf()

        bw, bw_conf = test_dinfo.bw.rounded_average_conf()
        _, bw_dev = test_dinfo.bw.rounded_average_dev()
        conf_perc = int(round(bw_conf * 100 / bw))
        dev_perc = int(round(bw_dev * 100 / bw))

        lat, _ = test_dinfo.lat.rounded_average_conf()
        lat = round_3_digit(int(lat) // 1000)

        iops_per_vm = round_3_digit(iops / data.testnodes_count)
        bw_per_vm = round_3_digit(bw / data.testnodes_count)

        iops = round_3_digit(iops)
        bw = round_3_digit(bw)

        params = (data.name.rsplit('_', 1)[0],
                  data.summary, int(iops), int(bw), str(conf_perc),
                  str(dev_perc),
                  int(iops_per_vm), int(bw_per_vm), lat)
        tab.add_row(params)

    tab.header(header)

    return tab.draw()
