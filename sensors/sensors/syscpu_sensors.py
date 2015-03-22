from discover import provides
from utils import SensorInfo, is_dev_accepted

# 0 - cpu name
# 1 - user: normal processes executing in user mode
# 2 - nice: niced processes executing in user mode
# 3 - system: processes executing in kernel mode
# 4 - idle: twiddling thumbs
# 5 - iowait: waiting for I/O to complete
# 6 - irq: servicing interrupts
# 7 - softirq: servicing softirqs

io_values_pos = [
    (1, 'user_processes', True),
    (2, 'nice_processes', True),
    (3, 'system_processes', True),
    (4, 'idle_time', True),
]


@provides("system-cpu")
def syscpu_stat(disallowed_prefixes=('intr', 'ctxt', 'btime', 'processes',
                                 'procs_running', 'procs_blocked', 'softirq'),
            allowed_prefixes=None):
    results = {}

    for line in open('/proc/stat'):
        vals = line.split()
        dev_name = vals[0]

        dev_ok = is_dev_accepted(dev_name,
                                 disallowed_prefixes,
                                 allowed_prefixes)

        if dev_ok:
            for pos, name, accum_val in io_values_pos:
                sensor_name = "{0}.{1}".format(dev_name, name)
                results[sensor_name] = SensorInfo(int(vals[pos]), accum_val)
    return results

