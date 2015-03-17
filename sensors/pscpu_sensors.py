import os
import time

from discover import provides
from utils import SensorInfo, get_pid_name, get_pid_list



@provides("perprocess-cpu")
def pscpu_stat(disallowed_prefixes=None, allowed_prefixes=None):
    results = {}
    pid_stat0 = {}
    sys_stat0 = {}
    pid_list = get_pid_list(disallowed_prefixes, allowed_prefixes)

    for pid in pid_list:
        try:
            pid_stat0[pid] = pid_stat(pid)
            sys_stat0[pid] = sys_stat()
        except IOError:
            # may be, proc has already terminated
            continue

    time.sleep(1)

    for pid in pid_list:
        try:
            dev_name = get_pid_name(pid)

            pid_stat1 = pid_stat(pid)
            sys_stat1 = sys_stat()
            cpu = (pid_stat1 - pid_stat0[pid]) / (sys_stat1 - sys_stat0[pid])

            sensor_name = "{0}.{1}".format(dev_name, pid)
            results[sensor_name] = SensorInfo(cpu*100, False)
        except IOError:
            # may be, proc has already terminated
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


def sys_stat():
    """ Return total system cpu usage time"""
    with open('/proc/stat', 'r') as procfile:
        cputimes = procfile.readline().split()[1:]
        cputotal = 0
        # count from /proc/stat sum
        for i in cputimes:
            cputotal = cputotal + int(i)
        return float(cputotal)
