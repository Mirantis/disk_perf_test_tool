import os
from collections import namedtuple

SensorInfo = namedtuple("SensorInfo", ['value', 'is_accumulated'])
# SensorInfo = NamedTuple("SensorInfo", [('value', int), ('is_accumulated', bool)])


def provides(name: str):
    def closure(func):
        return func
    return closure


def is_dev_accepted(name, disallowed_prefixes, allowed_prefixes):
    dev_ok = True

    if disallowed_prefixes is not None:
        dev_ok = all(not name.startswith(prefix)
                     for prefix in disallowed_prefixes)

    if dev_ok and allowed_prefixes is not None:
        dev_ok = any(name.startswith(prefix)
                     for prefix in allowed_prefixes)

    return dev_ok


def get_pid_list(disallowed_prefixes, allowed_prefixes):
    """Return pid list from list of pids and names"""
    # exceptions
    but = disallowed_prefixes if disallowed_prefixes is not None else []
    if allowed_prefixes is None:
        # if nothing setted - all ps will be returned except setted
        result = [pid
                  for pid in os.listdir('/proc')
                  if pid.isdigit() and pid not in but]
    else:
        result = []
        for pid in os.listdir('/proc'):
            if pid.isdigit() and pid not in but:
                name = get_pid_name(pid)
                if pid in allowed_prefixes or \
                   any(name.startswith(val) for val in allowed_prefixes):
                    # this is allowed pid?
                    result.append(pid)
    return result


def get_pid_name(pid):
    """Return name by pid"""
    try:
        with open(os.path.join('/proc/', pid, 'cmdline'), 'r') as pidfile:
            try:
                cmd = pidfile.readline().split()[0]
                return os.path.basename(cmd).rstrip('\x00')
            except IndexError:
                # no cmd returned
                return "<NO NAME>"
    except IOError:
        # upstream wait any string, no matter if we couldn't read proc
        return "no_such_process"


def delta(func, only_upd=True):
    prev = {}
    while True:
        for dev_name, vals in func():
            if dev_name not in prev:
                prev[dev_name] = {}
                for name, (val, _) in vals.items():
                    prev[dev_name][name] = val
            else:
                dev_prev = prev[dev_name]
                res = {}
                for stat_name, (val, accum_val) in vals.items():
                    if accum_val:
                        if stat_name in dev_prev:
                            delta = int(val) - int(dev_prev[stat_name])
                            if not only_upd or 0 != delta:
                                res[stat_name] = str(delta)
                        dev_prev[stat_name] = val
                    elif not only_upd or '0' != val:
                        res[stat_name] = val

                if only_upd and len(res) == 0:
                    continue
                yield dev_name, res
        yield None, None


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

io_values_pos = [
    (3, 'reads_completed', True),
    (5, 'sectors_read', True),
    (6, 'rtime', True),
    (7, 'writes_completed', True),
    (9, 'sectors_written', True),
    (10, 'wtime', True),
    (11, 'io_queue', False),
    (13, 'io_time', True)
]


@provides("block-io")
def io_stat(disallowed_prefixes=('ram', 'loop'), allowed_prefixes=None):
    results = {}
    for line in open('/proc/diskstats'):
        vals = line.split()
        dev_name = vals[2]

        dev_ok = is_dev_accepted(dev_name,
                                 disallowed_prefixes,
                                 allowed_prefixes)
        if dev_name[-1].isdigit():
            dev_ok = False

        if dev_ok:
            for pos, name, accum_val in io_values_pos:
                sensor_name = "{0}.{1}".format(dev_name, name)
                results[sensor_name] = SensorInfo(int(vals[pos]), accum_val)
    return results


def get_latency(stat1, stat2):
    disks = set(i.split('.', 1)[0] for i in stat1)
    results = {}

    for disk in disks:
        rdc = disk + '.reads_completed'
        wrc = disk + '.writes_completed'
        rdt = disk + '.rtime'
        wrt = disk + '.wtime'
        lat = 0.0

        io_ops1 = stat1[rdc].value + stat1[wrc].value
        io_ops2 = stat2[rdc].value + stat2[wrc].value

        diops = io_ops2 - io_ops1

        if diops != 0:
            io1 = stat1[rdt].value + stat1[wrt].value
            io2 = stat2[rdt].value + stat2[wrt].value
            lat = abs(float(io1 - io2)) / diops

        results[disk + '.latence'] = SensorInfo(lat, False)

    return results


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
def net_stat(disallowed_prefixes=('docker', 'lo'), allowed_prefixes=('eth',)):
    results = {}

    for line in open('/proc/net/dev').readlines()[2:]:
        dev_name, stats = line.split(":", 1)
        dev_name = dev_name.strip()
        vals = stats.split()

        dev_ok = is_dev_accepted(dev_name,
                                 disallowed_prefixes,
                                 allowed_prefixes)

        if '.' in dev_name and dev_name.split('.')[-1].isdigit():
            dev_ok = False

        if dev_ok:
            for pos, name, accum_val in net_values_pos:
                sensor_name = "{0}.{1}".format(dev_name, name)
                results[sensor_name] = SensorInfo(int(vals[pos]), accum_val)
    return results


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
    """Return total cpu usage time from process"""
    # read /proc/pid/stat
    with open(os.path.join('/proc/', pid, 'stat'), 'r') as pidfile:
        proctimes = pidfile.readline().split()
    # get utime from /proc/<pid>/stat, 14 item
    utime = proctimes[13]
    # get stime from proc/<pid>/stat, 15 item
    stime = proctimes[14]
    # count total process used time
    return float(int(utime) + int(stime))


# Based on ps_mem.py:
# Licence: LGPLv2
# Author:  P@draigBrady.com
# Source:  http://www.pixelbeat.org/scripts/ps_mem.py
#   http://github.com/pixelb/scripts/commits/master/scripts/ps_mem.py


# Note shared is always a subset of rss (trs is not always)
def get_mem_stats(pid):
    """Return memory data of pid in format (private, shared)"""

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
    """Return RAM size in Kb"""
    with open("/proc/meminfo") as proc:
        mem_total = proc.readline().split()
    return mem_total[1]


# 0 - cpu name
# 1 - user: normal processes executing in user mode
# 2 - nice: niced processes executing in user mode
# 3 - system: processes executing in kernel mode
# 4 - idle: twiddling thumbs
# 5 - iowait: waiting for I/O to complete
# 6 - irq: servicing interrupts
# 7 - softirq: servicing softirqs

cpu_values_pos = [
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
            for pos, name, accum_val in cpu_values_pos:
                sensor_name = "{0}.{1}".format(dev_name, name)
                results[sensor_name] = SensorInfo(int(vals[pos]),
                                                  accum_val)
        elif dev_name == 'procs_blocked':
            val = int(vals[1])
            results["cpu.procs_blocked"] = SensorInfo(val, False)
        elif dev_name.startswith('cpu'):
            core_count += 1

    # procs in queue
    TASKSPOS = 3
    vals = open('/proc/loadavg').read().split()
    ready_procs = vals[TASKSPOS].partition('/')[0]
    # dec on current proc
    procs_queue = (float(ready_procs) - 1) / core_count
    results["cpu.procs_queue"] = SensorInfo(procs_queue, False)

    return results


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
        dev_name = vals[0].rstrip(":")

        dev_ok = is_dev_accepted(dev_name,
                                 disallowed_prefixes,
                                 allowed_prefixes)

        title = "ram.{0}".format(dev_name)

        if dev_ok:
            results[title] = SensorInfo(int(vals[1]), False)

    if 'ram.MemFree' in results and 'ram.MemTotal' in results:
        used = results['ram.MemTotal'].value - results['ram.MemFree'].value
        usage = float(used) / results['ram.MemTotal'].value
        results["ram.usage_percent"] = SensorInfo(usage, False)
    return results
