import os
import sys
import json
import time
import array
import logging
import threading
import traceback
import subprocess
import collections


mod_name = "sensors"
__version__ = (0, 1)


logger = logging.getLogger("agent.sensors")
SensorsMap = {}


class Sensor(object):
    def __init__(self, params, allowed=None, disallowed=None):
        self.params = params
        self.allowed = allowed
        self.disallowed = disallowed
        self.allowed_names = set()

    def add_data(self, device, name, value):
        pass

    def collect(self):
        pass

    def get_updates(self):
        pass

    @classmethod
    def unpack_results(cls, device, metrics, data):
        pass


class ArraysSensor(Sensor):
    typecode = 'L'

    def __init__(self, params, allowed=None, disallowed=None):
        Sensor.__init__(self, params, allowed, disallowed)
        self.data = collections.defaultdict(lambda: array.array(self.typecode))

    def add_data(self, device, name, value):
        self.data[(device, name)].append(value)

    def get_updates(self):
        res = self.data
        self.data = collections.defaultdict(lambda: array.array(self.typecode))
        packed = {
            name: arr.typecode + arr.tostring()
            for name, arr in res.items()
        }
        return packed

    @classmethod
    def unpack_results(cls, device, metrics, packed):
        arr = array.array(chr(packed[0]))
        if sys.version_info >= (3, 0, 0):
            arr.frombytes(packed[1:])
        else:
            arr.fromstring(packed[1:])
        return arr

    def is_dev_accepted(self, name):
        dev_ok = True

        if self.disallowed is not None:
            dev_ok = all(not name.startswith(prefix) for prefix in self.disallowed)

        if dev_ok and self.allowed is not None:
            dev_ok = any(name.startswith(prefix) for prefix in self.allowed)

        return dev_ok


def provides(name):
    def closure(cls):
        SensorsMap[name] = cls
        return cls
    return closure


def get_pid_list(disallowed_prefixes, allowed_prefixes):
    """Return pid list from list of pids and names"""
    # exceptions
    disallowed = disallowed_prefixes if disallowed_prefixes is not None else []
    if allowed_prefixes is None:
        # if nothing setted - all ps will be returned except setted
        result = [pid for pid in os.listdir('/proc')
                  if pid.isdigit() and pid not in disallowed]
    else:
        result = []
        for pid in os.listdir('/proc'):
            if pid.isdigit() and pid not in disallowed:
                name = get_pid_name(pid)
                if pid in allowed_prefixes or any(name.startswith(val) for val in allowed_prefixes):
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


@provides("block-io")
class BlockIOSensor(ArraysSensor):
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

    def __init__(self, *args, **kwargs):
        ArraysSensor.__init__(self, *args, **kwargs)
        if self.disallowed is None:
            self.disallowed = ('ram', 'loop')
        self.allowed_devs = set()

        for line in open('/proc/diskstats'):
            dev_name = line.split()[2]
            if self.is_dev_accepted(dev_name) and not dev_name[-1].isdigit():
                self.accepted_devs.add(dev_name)

    def collect(self):
        for line in open('/proc/diskstats'):
            vals = line.split()
            dev_name = vals[2]
            if dev_name not in self.allowed_devs:
                continue

            for pos, name, _ in self.io_values_pos:
                self.add_data(dev_name, name, int(vals[pos]))


@provides("net-io")
class NetIOSensor(ArraysSensor):
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

    def __init__(self, *args, **kwargs):
        ArraysSensor.__init__(self, *args, **kwargs)

        if self.disallowed is None:
            self.disallowed = ('docker', 'lo')

        if self.allowed is None:
            self.allowed = ('eth',)

        self.allowed_devs = set()
        for line in open('/proc/net/dev').readlines()[2:]:
            dev_name = line.split(":", 1)[0].strip()
            dev_ok = self.is_dev_accepted(dev_name)
            if dev_ok and ('.' not in dev_name or not dev_name.split('.')[-1].isdigit()):
                self.allowed_devs.add(dev_name)

    def collect(self):
        for line in open('/proc/net/dev').readlines()[2:]:
            dev_name, stats = line.split(":", 1)
            dev_name = dev_name.strip()
            if dev_name in self.allowed_devs:
                vals = stats.split()
                for pos, name, _ in self.net_values_pos:
                    self.add_data(dev_name, name, int(vals[pos]))


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
class ProcCpuSensor(ArraysSensor):
    def collect(self):
        # TODO(koder): fixed list of PID's must be given
        for pid in get_pid_list(self.disallowed, self.allowed):
            try:
                self.add_data(get_pid_name(pid), pid, pid_stat(pid))
            except IOError:
                # probably proc has already terminated, skip it
                continue


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
class ProcRamSensor(ArraysSensor):
    def collect(self):
        # TODO(koder): fixed list of PID's nust be given
        for pid in get_pid_list(self.disallowed, self.allowed):
            try:
                dev_name = get_pid_name(pid)

                private, shared = get_mem_stats(pid)
                total = private + shared
                sys_total = get_ram_size()
                usage = float(total) / sys_total

                sensor_name = "{0}({1})".format(dev_name, pid)

                self.add_data(sensor_name, "private_mem", private)
                self.add_data(sensor_name, "shared_mem", shared),
                self.add_data(sensor_name, "used_mem", total),
                self.add_data(sensor_name, "mem_usage_percent", int(usage * 100))
            except IOError:
                # permission denied or proc die
                continue


@provides("system-cpu")
class SystemCPUSensor(ArraysSensor):
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

    def collect(self):
        # calculate core count
        core_count = 0

        for line in open('/proc/stat'):
            vals = line.split()
            dev_name = vals[0]

            if dev_name == 'cpu':
                for pos, name, _ in self.cpu_values_pos:
                    self.add_data(dev_name, name, int(vals[pos]))
            elif dev_name == 'procs_blocked':
                self.add_data("cpu", "procs_blocked", int(vals[1]))
            elif dev_name.startswith('cpu'):
                core_count += 1

        # procs in queue
        TASKSPOS = 3
        vals = open('/proc/loadavg').read().split()
        ready_procs = vals[TASKSPOS].partition('/')[0]

        # dec on current proc
        procs_queue = (float(ready_procs) - 1) / core_count
        self.add_data("cpu", "procs_queue_x10", int(procs_queue * 10))


@provides("system-ram")
class SystemCPUSensor(ArraysSensor):
    # return this values or setted in allowed
    ram_fields = ['MemTotal', 'MemFree', 'Buffers', 'Cached', 'SwapCached',
                  'Dirty', 'Writeback', 'SwapTotal', 'SwapFree']

    def __init__(self, *args, **kwargs):
        ArraysSensor.__init__(self, *args, **kwargs)

        if self.allowed is None:
            self.allowed = self.ram_fields

        self.allowed_fields = set()
        for line in open('/proc/meminfo'):
            field_name = line.split()[0].rstrip(":")
            if self.is_dev_accepted(field_name):
                self.allowed_fields.add(field_name)

    def collect(self):
        for line in open('/proc/meminfo'):
            vals = line.split()
            field = vals[0].rstrip(":")
            if field in self.allowed_fields:
                self.add_data("ram", field, int(vals[1]))


@provides("ceph")
class CephSensor(ArraysSensor):
    def collect(self):
        def get_val(dct, path):
            if '/' in path:
                root, next = path.split('/', 1)
                return get_val(dct[root], next)
            return dct[path]

        for osd_id in self.params['osds']:
            asok = '/var/run/ceph/{}-osd.{}.asok'.format(self.params['cluster'], osd_id)
            out = subprocess.check_output('ceph daemon {} perf dump'.format(asok), shell=True)
            data = json.loads(out)
            for key_name in self.params['counters']:
                self.add_data("osd{}".format(osd_id), key_name.replace("/", "."), get_val(data, key_name))


class SensorsData(object):
    def __init__(self):
        self.cond = threading.Condition()
        self.collected_at = array.array("f")
        self.stop = False
        self.sensors = {}
        self.data_fd = None  # temporary file to store results
        self.exception = None


def collect(sensors_config):
    curr = {}
    for name, config in sensors_config.items():
        params = {'config': config}

        if "allow" in config:
            params["allowed_prefixes"] = config["allow"]

        if "disallow" in config:
            params["disallowed_prefixes"] = config["disallow"]

        curr[name] = SensorsMap[name](**params)
    return curr


# TODO(koder): a lot code here can be optimized and cached, but nobody cares (c)
def sensors_bg_thread(sensors_config, sdata):
    try:
        next_collect_at = time.time()

        # prepare sensor classes
        with sdata.cond:
            sdata.sensors = {}
            for name, config in sensors_config.items():
                params = {'params': config}

                if "allow" in config:
                    params["allowed_prefixes"] = config["allow"]

                if "disallow" in config:
                    params["disallowed_prefixes"] = config["disallow"]

                sdata.sensors[name] = SensorsMap[name](**params)

            import pprint
            logger.debug("sensors.config = %s", pprint.pformat(sensors_config))
            logger.debug("Sensors map keys %s", ", ".join(sdata.sensors.keys()))

        # TODO: handle exceptions here
        # main loop
        while not sdata.stop:
            dtime = next_collect_at - time.time()
            if dtime > 0:
                with sdata.cond:
                    sdata.cond.wait(dtime)

            next_collect_at += 1.0

            if sdata.stop:
                break

            ctm = time.time()
            with sdata.cond:
                sdata.collected_at.append(ctm)
                for sensor in sdata.sensors.values():
                    sensor.collect()
                etm = time.time()
                sdata.collected_at.append(etm)

            if etm - ctm > 0.1:
                # TODO(koder): need to signal that something in not really ok with sensor collecting
                pass

    except Exception:
        logger.exception("In sensor BG thread")
        sdata.exception = traceback.format_exc()


sensors_thread = None
sdata = None  # type: SensorsData


def rpc_start(sensors_config):
    global sensors_thread
    global sdata

    if array.array('L').itemsize != 8:
        message = "Python array.array('L') items should be 8 bytes in size, not {}." + \
                  " Can't provide sensors on this platform. Disable sensors in config and retry"
        raise ValueError(message.format(array.array('L').itemsize))

    if sensors_thread is not None:
        raise ValueError("Thread already running")

    sdata = SensorsData()
    sensors_thread = threading.Thread(target=sensors_bg_thread, args=(sensors_config, sdata))
    sensors_thread.daemon = True
    sensors_thread.start()


def unpack_rpc_updates(res_tuple):
    data, collected_at_b = res_tuple
    collected_at = array.array('f')
    collected_at.frombytes(collected_at_b)
    yield 'collected_at', collected_at

    # TODO: data is unpacked/repacked here with no reason
    for sensor_path, packed_data in data.items():
        sensor_path = sensor_path.decode("utf8")
        sensor_name, device, metric = sensor_path.split('.', 2)
        data = SensorsMap[sensor_name].unpack_results(device, metric, packed_data)
        yield sensor_path, data


def rpc_get_updates():
    if sdata is None:
        raise ValueError("No sensor thread running")

    with sdata.cond:
        if sdata.exception:
            raise Exception(sdata.exception)

        res = {}
        for sensor_name, sensor in sdata.sensors.items():
            for (device, metrics), val in sensor.get_updates().items():
                res["{}.{}.{}".format(sensor_name, device, metrics)] = val

        collected_at = sdata.collected_at
        sdata.collected_at = array.array(sdata.collected_at.typecode)

    return res, collected_at.tostring()


def rpc_stop():
    global sensors_thread
    global sdata

    if sensors_thread is None:
        raise ValueError("No sensor thread running")

    sdata.stop = True
    with sdata.cond:
        sdata.cond.notify_all()

    sensors_thread.join()

    if sdata.exception:
        raise Exception(sdata.exception)

    res = rpc_get_updates()

    sensors_thread = None
    sdata = None

    return res
