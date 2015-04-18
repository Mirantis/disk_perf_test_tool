from ..discover import provides
from .utils import SensorInfo, is_dev_accepted


# return this values or setted in allowed
ram_fields = [
    'MemTotal',
    'MemFree',
    'Buffers',
    'Cached',
    'SwapCached',
    'Dirty',
    'Writeback',
    'SwapTotal',
    'SwapFree'
]


@provides("system-ram")
def sysram_stat(disallowed_prefixes=None, allowed_prefixes=None):
    if allowed_prefixes is None:
        allowed_prefixes = ram_fields
    results = {}
    for line in open('/proc/meminfo'):
        vals = line.split()
        dev_name = vals[0]

        dev_ok = is_dev_accepted(dev_name,
                                 disallowed_prefixes,
                                 allowed_prefixes)

        if dev_ok:
            results[dev_name] = SensorInfo(int(vals[1]), False)
    return results
