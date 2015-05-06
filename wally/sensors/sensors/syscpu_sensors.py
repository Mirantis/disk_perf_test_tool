from .utils import SensorInfo
from ..discover import provides

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
def syscpu_stat(disallowed_prefixes=None, allowed_prefixes=None):
    results = {}

    # calculate core count
    core_count = 0

    for line in open('/proc/stat'):
        vals = line.split()
        dev_name = vals[0]

        if dev_name == 'cpu':
            for pos, name, accum_val in io_values_pos:
                sensor_name = "{0}.{1}".format(dev_name, name)
                results[sensor_name] = SensorInfo(int(vals[pos]),
                                                  accum_val)
        elif dev_name == 'procs_blocked':
            val = int(vals[1]) // core_count
            results["cpu.procs_blocked"] = SensorInfo(val, False)
        elif dev_name.startswith('cpu'):
            core_count += 1

    return results
