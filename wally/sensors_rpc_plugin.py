import os
import time
import array
import threading


mod_name = "sensor"
__version__ = (0, 1)


SensorsMap = {}


def provides(name):
    def closure(func):
        SensorsMap[name] = func
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
    disallowed = disallowed_prefixes if disallowed_prefixes is not None else []
    if allowed_prefixes is None:
        # if nothing setted - all ps will be returned except setted
        result = [pid
                  for pid in os.listdir('/proc')
                  if pid.isdigit() and pid not in disallowed]
    else:
        result = []
        for pid in os.listdir('/proc'):
            if pid.isdigit() and pid not in disallowed:
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
        if not dev_ok or dev_name[-1].isdigit():
            continue

        for pos, name, _ in io_values_pos:
            results["{0}.{1}".format(dev_name, name)] = int(vals[pos])
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
            for pos, name, _ in net_values_pos:
                results["{0}.{1}".format(dev_name, name)] = int(vals[pos])
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


@provides("perprocess-cpu")
def pscpu_stat(disallowed_prefixes=None, allowed_prefixes=None):
    results = {}
    # TODO(koder): fixed list of PID's nust be given
    for pid in get_pid_list(disallowed_prefixes, allowed_prefixes):
        try:
            results["{0}.{1}".format(get_pid_name(pid), pid)] = pid_stat(pid)
        except IOError:
            # may be, proc has already terminated, skip it
            continue
    return results


def get_mem_stats(pid):
    """Return memory data of pid in format (private, shared)"""

    fname = '/proc/{0}/{1}'.format(pid, "smaps")
    lines = open(fname).readlines()

    shared = 0
    private = 0
    pss = 0

    # add 0.5KiB as this avg error due to truncation
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


def get_ram_size():
    """Return RAM size in Kb"""
    with open("/proc/meminfo") as proc:
        mem_total = proc.readline().split()
    return int(mem_total[1])


@provides("perprocess-ram")
def psram_stat(disallowed_prefixes=None, allowed_prefixes=None):
    results = {}
    # TODO(koder): fixed list of PID's nust be given
    for pid in get_pid_list(disallowed_prefixes, allowed_prefixes):
        try:
            dev_name = get_pid_name(pid)

            private, shared = get_mem_stats(pid)
            total = private + shared
            sys_total = get_ram_size()
            usage = float(total) / sys_total

            sensor_name = "{0}({1})".format(dev_name, pid)

            results.update([
                (sensor_name + ".private_mem", private),
                (sensor_name + ".shared_mem", shared),
                (sensor_name + ".used_mem", total),
                (sensor_name + ".mem_usage_percent", int(usage * 100))])
        except IOError:
            # permission denied or proc die
            continue
    return results

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
            for pos, name, _ in cpu_values_pos:
                sensor_name = "{0}.{1}".format(dev_name, name)
                results[sensor_name] = int(vals[pos])
        elif dev_name == 'procs_blocked':
            val = int(vals[1])
            results["cpu.procs_blocked"] = val
        elif dev_name.startswith('cpu'):
            core_count += 1

    # procs in queue
    TASKSPOS = 3
    vals = open('/proc/loadavg').read().split()
    ready_procs = vals[TASKSPOS].partition('/')[0]

    # dec on current proc
    procs_queue = (float(ready_procs) - 1) / core_count
    results["cpu.procs_queue"] = procs_queue

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
            results[title] = int(vals[1])

    if 'ram.MemFree' in results and 'ram.MemTotal' in results:
        used = results['ram.MemTotal'].value - results['ram.MemFree'].value
        results["ram.usage_percent"] = int(float(used) / results['ram.MemTotal'].value)

    return results


class SensorsData(object):
    def __init__(self):
        self.cond = threading.Condition()
        self.collected_at = array.array("f")
        self.stop = False
        self.data = {}  # map sensor_name to list of results
        self.data_fd = None


# TODO(koder): a lot code here can be optimized and cached, but nobody cares (c)
def sensors_bg_thread(sensors_config, sdata):
    next_collect_at = time.time()

    while not sdata.stop:
        dtime = next_collect_at - time.time()
        if dtime > 0:
            sdata.cond.wait(dtime)

        if sdata.stop:
            break

        ctm = time.time()
        curr = {}
        for name, config in sensors_config.items():
            params = {}

            if "allow" in config:
                params["allowed_prefixes"] = config["allow"]

            if "disallow" in config:
                params["disallowed_prefixes"] = config["disallow"]

            curr[name] = SensorsMap[name](**params)

        etm = time.time()

        if etm - ctm > 0.1:
            # TODO(koder): need to signal that something in not really ok with sensor collecting
            pass

        with sdata.cond:
            sdata.collected_at.append(ctm)
            for source_name, vals in curr.items():
                for sensor_name, val in vals.items():
                    key = (source_name, sensor_name)
                    if key not in sdata.data:
                        sdata.data[key] = array.array("I", [val])
                    else:
                        sdata.data[key].append(val)


sensors_thread = None
sdata = None  # type: SensorsData


def rpc_start(sensors_config):
    global sensors_thread
    global sdata

    if sensors_thread is not None:
        raise ValueError("Thread already running")

    sdata = SensorsData()
    sensors_thread = threading.Thread(target=sensors_bg_thread, args=(sensors_config, sdata))
    sensors_thread.daemon = True
    sensors_thread.start()


def rpc_get_updates():
    if sdata is None:
        raise ValueError("No sensor thread running")

    with sdata.cond:
        res = sdata.data
        collected_at = sdata.collected_at
        sdata.collected_at = array.array("f")
        sdata.data = {name: array.array("I") for name in sdata.data}

    return res, collected_at


def rpc_stop():
    global sensors_thread
    global sdata

    if sensors_thread is None:
        raise ValueError("No sensor thread running")

    sdata.stop = True
    with sdata.cond:
        sdata.cond.notify_all()

    sensors_thread.join()
    res = sdata.data
    collected_at = sdata.collected_at

    sensors_thread = None
    sdata = None

    return res, collected_at
