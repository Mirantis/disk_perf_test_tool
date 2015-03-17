from ps_mem import getMemStats

from discover import provides
from utils import SensorInfo, get_pid_name



@provides("perprocess-ram")
def psram_stat(disallowed_prefixes=None, allowed_prefixes=None):
    results = {}
    pid_list = get_pid_list(disallowed_prefixes, allowed_prefixes)
    print pid_list
    for pid in pid_list:
        try:
            dev_name = get_pid_name(pid)

            private, shared = getMemStats(pid)
            total = private + shared
            sys_total = get_ram_size()
            usage = float(total) / float(sys_total)

            sensor_name = "{0}.{1}".format(dev_name, pid)

            results[sensor_name + ".private_mem"] = SensorInfo(private, False)
            results[sensor_name + ".shared_mem"] = SensorInfo(shared, False)
            results[sensor_name + ".used_mem"] = SensorInfo(total, False)
            results[sensor_name + ".mem_usage_percent"] = SensorInfo(usage*100, False)
        except IOError:
            # permission denied or proc die
            continue
    return results


def get_ram_size():
    """ Return RAM size in Kb"""
    with open("/proc/meminfo") as proc:
        mem_total = proc.readline().split()
        return mem_total[1]
