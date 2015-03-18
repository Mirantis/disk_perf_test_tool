import os

from discover import provides
from utils import SensorInfo, get_pid_name, get_pid_list



@provides("perprocess-cpu")
def pscpu_stat(disallowed_prefixes=None, allowed_prefixes=None):
    results = {}
    pid_list = get_pid_list(disallowed_prefixes, allowed_prefixes)

    for pid in pid_list:
        try:
            dev_name = get_pid_name(pid)

            pid_stat1 = pid_stat(pid)

            sensor_name = "{0}.{1}".format(dev_name, pid)
            results[sensor_name] = SensorInfo(pid_stat1, True)
        except IOError:
            # may be, proc has already terminated, skip it
            continue
    return results


def pid_stat(pid):
    """ Return total cpu usage time from process"""
    # read /proc/pid/stat
    with open(os.path.join('/proc/', pid, 'stat'), 'r') as pidfile:
        proctimes = pidfile.readline().split()
    # get utime from /proc/<pid>/stat, 14 item
    utime = proctimes[13]
    # get stime from proc/<pid>/stat, 15 item
    stime = proctimes[14]
    # count total process used time
    proctotal = int(utime) + int(stime)
    return float(proctotal)
