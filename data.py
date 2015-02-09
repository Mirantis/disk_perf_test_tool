import re
import sys


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
            meta['sync'] = 's' if meta['sync'] else 'a'
            key = "{action} {sync} {blocksize}k".format(**meta)
            results.setdefault(key, []).append(val['bw_mean'])

    processed_res = {}

    for k, v in results.items():
        v.sort()
        med = float(sum(v)) / len(v)
        ran = sum(abs(x - med) for x in v) / len(v)
        processed_res[k] = (int(med), int(ran))

    return processed_res


def ksort(x):
    op, sync, sz = x.split(" ")
    return (op, sync, int(sz[:-1]))


def show_data(path1, path2=None):
    templ_1 = "  {:>10}  {:>5}  {:>5}     {:>6} ~ {:>5} {:>2}% {:>5}"
    templ_2 = templ_1 + "      {:>6} ~ {:>5} {:>2}% {:>5} ----  {:>6}%  "

    ln_1 = templ_1.replace("<", "^").replace(">", "^")
    ln_1 = ln_1.format("Oper", "Sync", "BSZ", "BW1", "DEV1", "%", "IOPS1")

    ln_2 = templ_2.replace("<", "^").replace(">", "^")
    ln_2 = ln_2.format("Oper", "Sync", "BSZ", "BW1", "DEV1", "%",
                       "IOPS1", "BW2", "DEV2", "%", "IOPS2", "DIFF %")

    sep_1 = '-' * len(ln_1)
    sep_2 = '-' * len(ln_2)

    res_1 = get_data_from_output(path1)

    if path2 is None:
        res_2 = None
        sep = sep_1
        ln = ln_1
        templ = templ_1
    else:
        res_2 = get_data_from_output(path2)
        sep = sep_2
        ln = ln_2
        templ = templ_2

    print sep
    print ln
    print sep

    prev_tp = None

    common_keys = set(res_1.keys())

    if res_2 is not None:
        common_keys &= set(res_2.keys())

    for k in sorted(common_keys, key=ksort):
        tp = k.rsplit(" ", 1)[0]
        op, s, sz = k.split(" ")
        s = 'sync' if s == 's' else 'async'

        if tp != prev_tp and prev_tp is not None:
            print sep

        prev_tp = tp

        m1, d1 = res_1[k]
        iops1 = m1 / int(sz[:-1])
        perc1 = int(d1 * 100.0 / m1 + 0.5)

        if res_2 is not None:
            m2, d2 = res_2[k]
            iops2 = m2 / int(sz[:-1])
            perc2 = int(d2 * 100.0 / m2 + 0.5)
            avg_diff = int(((m2 - m1) * 100.) / m2 + 0.5)

        if res_2 is not None:
            print templ.format(op, s, sz, m1, d1, perc1, iops1,
                               m2, d2, perc2, iops2, avg_diff)
        else:
            print templ.format(op, s, sz, m1, d1, perc1, iops1)

    print sep


def main(argv):
    path1 = argv[0]
    path2 = argv[1] if len(argv) > 1 else None
    show_data(path1, path2)
    return 0

if __name__ == "__main__":
    exit(main(sys.argv[1:]))
# print " ", results[k]
