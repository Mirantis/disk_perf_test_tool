import os
import sys
import time
import pprint
import threading


mapping = [
    "major number",
    "minor mumber",
    "device name",
    "reads completed successfully",
    "reads merged",
    "sectors read",
    "time spent reading (ms)",
    "writes complete",
    "writes merged",
    "sectors written",
    "time spent writing (ms)",
    "I/Os currently in progress",
    "time spent doing I/Os (ms)",
    "weighted time spent doing I/Os (ms)"
]


def read_dstats():
    res = {}
    for line in open("/proc/diskstats"):
        stats = dict(zip(mapping, line.split()))
        name = stats.pop('device name')
        res[name] = {k: int(v) for k, v in stats.items()}
    return res


def diff_stats(obj1, obj2):
    return {key: (val - obj2[key]) for key, val in obj1.items()}


def run_tool(cmd, suppress_console=False):
    s_cmd = " ".join(cmd)
    if suppress_console:
        s_cmd += " >/dev/null 2>&1 "
    os.system(s_cmd)

devices = sys.argv[1].split(',')
cmd = sys.argv[2:]

th = threading.Thread(None, run_tool, None, (cmd,))
th.daemon = True

rstats = read_dstats()
prev_stats = {device: rstats[device] for device in devices}
begin_stats = prev_stats

th.start()

wr_compl = "writes complete"

while True:
    time.sleep(1)

    rstats = read_dstats()
    new_stats = {device: rstats[device] for device in devices}

    # print "Delta writes complete =",
    for device in devices:
        delta = new_stats[device][wr_compl] - prev_stats[device][wr_compl]
        # print device, delta,
    # print

    prev_stats = new_stats

    if not th.is_alive():
        break

pprint.pprint(diff_stats(new_stats[device], begin_stats[device]))
