from ..discover import provides
from .utils import SensorInfo, is_dev_accepted

#  1 - major number
#  2 - minor mumber
#  3 - device name
#  4 - reads completed successfully
#  5 - reads merged
#  6 - sectors read
#  7 - time spent reading (ms)
#  8 - writes completed
#  9 - writes merged
# 10 - sectors written
# 11 - time spent writing (ms)
# 12 - I/Os currently in progress
# 13 - time spent doing I/Os (ms)
# 14 - weighted time spent doing I/Os (ms)

net_values_pos = [
    (0, 'recv_bytes', True),
    (1, 'recv_packets', True),
    (8, 'send_bytes', True),
    (9, 'send_packets', True),
]


@provides("net-io")
def net_stat(disallowed_prefixes=('docker',), allowed_prefixes=None):
    results = {}

    for line in open('/proc/net/dev').readlines()[2:]:
        dev_name, stats = line.split(":", 1)
        dev_name = dev_name.strip()
        vals = stats.split()

        dev_ok = is_dev_accepted(dev_name,
                                 disallowed_prefixes,
                                 allowed_prefixes)
        if dev_ok:
            for pos, name, accum_val in net_values_pos:
                sensor_name = "{0}.{1}".format(dev_name, name)
                results[sensor_name] = SensorInfo(int(vals[pos]), accum_val)
    return results
