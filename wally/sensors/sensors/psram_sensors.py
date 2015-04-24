from ..discover import provides
from .utils import SensorInfo, get_pid_name, get_pid_list


# Based on ps_mem.py:
# Licence: LGPLv2
# Author:  P@draigBrady.com
# Source:  http://www.pixelbeat.org/scripts/ps_mem.py
#   http://github.com/pixelb/scripts/commits/master/scripts/ps_mem.py


# Note shared is always a subset of rss (trs is not always)
def get_mem_stats(pid):
    """ Return memory data of pid in format (private, shared) """

    fname = '/proc/{0}/{1}'.format(pid, "smaps")
    lines = open(fname).readlines()

    shared = 0
    private = 0
    pss = 0

    # add 0.5KiB as this avg error due to trunctation
    pss_adjust = 0.5

    for line in lines:
        if line.startswith("Shared"):
            shared += int(line.split()[1])

        if line.startswith("Private"):
            private += int(line.split()[1])

        if line.startswith("Pss"):
            pss += float(line.split()[1]) + pss_adjust

    # Note Shared + Private = Rss above
    # The Rss in smaps includes video card mem etc.

    if pss != 0:
        shared = int(pss - private)

    return (private, shared)


@provides("perprocess-ram")
def psram_stat(disallowed_prefixes=None, allowed_prefixes=None):
    results = {}
    pid_list = get_pid_list(disallowed_prefixes, allowed_prefixes)
    print pid_list
    for pid in pid_list:
        try:
            dev_name = get_pid_name(pid)

            private, shared = get_mem_stats(pid)
            total = private + shared
            sys_total = get_ram_size()
            usage = float(total) / float(sys_total)

            sensor_name = "{0}({1})".format(dev_name, pid)

            results[sensor_name + ".private_mem"] = SensorInfo(private, False)
            results[sensor_name + ".shared_mem"] = SensorInfo(shared, False)
            results[sensor_name + ".used_mem"] = SensorInfo(total, False)
            name = sensor_name + ".mem_usage_percent"
            results[name] = SensorInfo(usage * 100, False)
        except IOError:
            # permission denied or proc die
            continue
    return results


def get_ram_size():
    """ Return RAM size in Kb"""
    with open("/proc/meminfo") as proc:
        mem_total = proc.readline().split()
    return mem_total[1]
