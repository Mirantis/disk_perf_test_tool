import re
import sys
import json

from disk_perf_test_tool.utils import kb_to_ssize

splitter_rr = "(?ms)=====+\n"

test_time_rr = r"""
(?ims)(?P<start_time>[:0-9]{8}) - DEBUG - io-perf-tool - Passing barrier, starting test
(?P<finish_time>[:0-9]{8}) - DEBUG - io-perf-tool - Done\. Closing connection
"""

test_time_rr = test_time_rr.strip().replace('\n', '\\s+')
test_time_rr = test_time_rr.strip().replace(' ', '\\s+')
test_time_re = re.compile(test_time_rr)


def to_sec(val):
    assert val.count(":") == 2
    h, m, s = val.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def to_min_sec(val):
    return "{0:2d}:{1:02d}".format(val / 60, val % 60)


def get_test_time(block):
    time_m = test_time_re.search(block)
    if time_m is None:
        raise ValueError("Can't found time")

    start_time = to_sec(time_m.group('start_time'))
    finish_time = to_sec(time_m.group('finish_time'))
    test_time = finish_time - start_time

    if test_time < 0:
        # ..... really need print UTC to logs
        test_time += 24 * 60 * 60
    return test_time


run_test_params_rr = r"(?ims)Run\s+test\s+with" + \
                     r"\s+'.*?--iosize\s+(?P<size>[^ ]*)"
run_test_params_re = re.compile(run_test_params_rr)


def get_orig_size(block):
    orig_size = run_test_params_re.search(block)
    if orig_size is None:
        print block
        raise ValueError("Can't find origin size")
    return orig_size.group(1)


def get_data_from_output(fname):
    results = {}
    results_meta = {}
    fc = open(fname).read()
    prev_block = None

    for block in re.split(splitter_rr, fc):
        block = block.strip()

        if block.startswith("[{u'__meta__':"):

            for val in eval(block):
                meta = val['__meta__']

                if meta['sync']:
                    meta['sync'] = 's'
                elif meta['direct_io']:
                    meta['sync'] = 'd'
                else:
                    meta['sync'] = 'a'

                meta['fsize'] = kb_to_ssize(meta['size'] * meta['concurence'])
                key = ("{action} {sync} {blocksize}k " +
                       "{concurence} {fsize}").format(**meta)
                results.setdefault(key, []).append(val['bw_mean'])

                cmeta = results_meta.setdefault(key, {})
                cmeta.setdefault('times', []).append(get_test_time(prev_block))
                cmeta['orig_size'] = get_orig_size(prev_block)

        prev_block = block

    processed_res = {}

    for k, v in results.items():
        v.sort()
        med = float(sum(v)) / len(v)
        ran = sum(abs(x - med) for x in v) / len(v)
        processed_res[k] = (int(med), int(ran))
        t = results_meta[k]['times']
        results_meta[k]['times'] = int(float(sum(t)) / len(t))

    return processed_res, results_meta


def ksort(x):
    op, sync, sz, conc, fsize = x.split(" ")
    return (op, sync, int(sz[:-1]), int(conc))


def create_json_results(meta, file_data):
    row = {"build_id": "",
           "type": "",
           "iso_md5": ""}
    row.update(file_data)
    return json.dumps(row)


LINES_PER_HEADER = 20


def show_data(*pathes):
    begin = "|  {:>10}  {:>6}  {:>5} {:>3} {:>5} {:>7}"
    first_file_templ = "  |  {:>6} ~ {:>5} {:>2}% {:>5} {:>6}"
    other_file_templ = first_file_templ + " ----  {:>6}%"

    line_templ = begin + first_file_templ + \
        other_file_templ * (len(pathes) - 1) + "  |"

    header_ln = line_templ.replace("<", "^").replace(">", "^")

    params = ["Oper", "Sync", "BSZ", "CC", "DSIZE", "OSIZE",
              "BW1", "DEV1", "%", "IOPS1", "TIME"]
    for pos in range(1, len(pathes)):
        params += "BW{0}+DEV{0}+%+IOPS{0}+DIFF %+TTIME".format(pos).split("+")

    header_ln = header_ln.format(*params)

    sep = '-' * len(header_ln)

    results = []
    metas = []

    for path in pathes:
        result, meta = get_data_from_output(path)
        results.append(result)
        metas.append(meta)

    print sep
    print header_ln
    print sep

    prev_tp = None

    common_keys = set(results[0].keys())
    for result in results[1:]:
        common_keys &= set(result.keys())

    lcount = 0
    for k in sorted(common_keys, key=ksort):
        tp = k.rsplit(" ", 3)[0]
        op, s, sz, conc, fsize = k.split(" ")

        s = {'a': 'async', "s": "sync", "d": "direct"}[s]

        if tp != prev_tp and prev_tp is not None:
            print sep

            if lcount > LINES_PER_HEADER:
                print header_ln
                print sep
                lcount = 0

        prev_tp = tp

        m0, d0 = results[0][k]
        iops0 = m0 / int(sz[:-1])
        perc0 = int(d0 * 100.0 / m0 + 0.5)

        data = [op, s, sz, conc, fsize,
                metas[0][k]['orig_size'],
                m0, d0, perc0, iops0,
                to_min_sec(metas[0][k]['times'])]

        for meta, result in zip(metas[1:], results[1:]):
            m, d = result[k]
            iops = m / int(sz[:-1])
            perc = int(d * 100.0 / m + 0.5)
            avg_diff = int(((m - m0) * 100.) / m + 0.5)

            dtime = to_min_sec(meta[k]['times'])
            data.extend([m, d, perc, iops, avg_diff, dtime])

        print line_templ.format(*data)
        lcount += 1

    print sep


def main(argv):
    path1 = argv[0]
    if path1 == '--json':
        print create_json_results(*get_data_from_output(argv[1]))
    else:
        show_data(*argv)
    return 0

if __name__ == "__main__":
    exit(main(sys.argv[1:]))
# print " ", results[k]
