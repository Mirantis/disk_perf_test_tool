import re
import sys
import json

splitter_rr = "(?ms)=====+\n"


def get_data_from_output(fname):
    results = {}
    fc = open(fname).read()

    for block in re.split(splitter_rr, fc):
        block = block.strip()

        if not block.startswith("[{u'__meta__':"):
            continue

        for val in eval(block):
            meta = val['__meta__']

            if meta['sync']:
                meta['sync'] = 's'
            elif meta['direct_io']:
                meta['sync'] = 'd'
            else:
                meta['sync'] = 'a'
            key = "{action} {sync} {blocksize}k {concurence}".format(**meta)
            results.setdefault(key, []).append(val['bw_mean'])

    processed_res = {}

    for k, v in results.items():
        v.sort()
        med = float(sum(v)) / len(v)
        ran = sum(abs(x - med) for x in v) / len(v)
        processed_res[k] = (int(med), int(ran))

    return meta, processed_res


def ksort(x):
    op, sync, sz, conc = x.split(" ")
    return (op, sync, int(sz[:-1]), int(conc))


def create_json_results(meta, file_data):
    row = {"build_id": "",
           "type": "",
           "iso_md5": ""}
    row.update(file_data)
    return json.dumps(row)


def show_data(*pathes):
    begin = "|  {:>10}  {:>6}  {:>5} {:>3}"
    first_file_templ = "  |  {:>6} ~ {:>5} {:>2}% {:>5}"
    other_file_templ = "  |  {:>6} ~ {:>5} {:>2}% {:>5} ----  {:>6}%"

    line_templ = begin + first_file_templ + \
        other_file_templ * (len(pathes) - 1) + "  |"

    header_ln = line_templ.replace("<", "^").replace(">", "^")

    params = ["Oper", "Sync", "BSZ", "CC", "BW1", "DEV1", "%", "IOPS1"]
    for pos in range(1, len(pathes)):
        params += "BW{0}+DEV{0}+%+IOPS{0}+DIFF %".format(pos).split("+")

    header_ln = header_ln.format(*params)

    sep = '-' * len(header_ln)

    results = [get_data_from_output(path)[1] for path in pathes]

    print sep
    print header_ln
    print sep

    prev_tp = None

    common_keys = set(results[0].keys())
    for result in results[1:]:
        common_keys &= set(result.keys())

    for k in sorted(common_keys, key=ksort):
        tp = k.rsplit(" ", 2)[0]
        op, s, sz, conc = k.split(" ")

        s = {'a': 'async', "s": "sync", "d": "direct"}[s]

        if tp != prev_tp and prev_tp is not None:
            print sep

        prev_tp = tp

        results0 = results[0]
        m0, d0 = results0[k]
        iops0 = m0 / int(sz[:-1])
        perc0 = int(d0 * 100.0 / m0 + 0.5)

        data = [op, s, sz, conc, m0, d0, perc0, iops0]

        for result in results[1:]:
            m, d = result[k]
            iops = m / int(sz[:-1])
            perc = int(d * 100.0 / m + 0.5)
            avg_diff = int(((m - m0) * 100.) / m + 0.5)
            data.extend([m, d, perc, iops, avg_diff])

        print line_templ.format(*data)

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
