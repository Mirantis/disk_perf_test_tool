import os
import sys
import time
import json
import signal
import os.path
import argparse

import io_sensors
import net_sensors
import syscpu_sensors
import sysram_sensors
import pscpu_sensors

from utils import SensorInfo
from daemonize import Daemonize
from discover import all_sensors
from protocol import create_protocol


def get_values(required_sensors):
    result = {}
    for sensor_name, params in required_sensors:
        if sensor_name in all_sensors:
            result.update(all_sensors[sensor_name](**params))
        else:
            msg = "Sensor {0!r} isn't available".format(sensor_name)
            raise ValueError(msg)
    return time.time(), result


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--daemon',
                        choices=('start', 'stop', 'status'),
                        default=None)

    parser.add_argument('-u', '--url', default='stdout://')
    parser.add_argument('-t', '--timeout', type=float, default=1)
    parser.add_argument('sensors_config', type=argparse.FileType('r'),
                        default=None, nargs='?')
    return parser.parse_args(args[1:])


def daemon_main(required_sensors, opts):
    sender = create_protocol(opts.url)
    prev = {}

    while True:
        gtime, data = get_values(required_sensors.items())
        curr = {'time': SensorInfo(gtime, True)}
        for name, val in data.items():
            if val.is_accumulated:
                if name in prev:
                    curr[name] = SensorInfo(val.value - prev[name], True)
                prev[name] = val.value
            else:
                curr[name] = SensorInfo(val.value, False)
        sender.send(curr)
        time.sleep(opts.timeout)


def main(argv):
    opts = parse_args(argv)

    if opts.daemon is not None:
        pid_file = "/tmp/sensors.pid"
        if opts.daemon == 'start':
            required_sensors = json.loads(opts.sensors_config.read())

            def root_func():
                daemon_main(required_sensors, opts)

            daemon = Daemonize(app="perfcollect_app",
                               pid=pid_file,
                               action=root_func)
            daemon.start()
        elif opts.daemon == 'stop':
            if os.path.isfile(pid_file):
                pid = int(open(pid_file).read())
                if os.path.exists("/proc/" + str(pid)):
                    os.kill(pid, signal.SIGTERM)

                time.sleep(0.1)

                if os.path.exists("/proc/" + str(pid)):
                    os.kill(pid, signal.SIGKILL)

                if os.path.isfile(pid_file):
                    os.unlink(pid_file)
        elif opts.daemon == 'status':
            if os.path.isfile(pid_file):
                pid = int(open(pid_file).read())
                if os.path.exists("/proc/" + str(pid)):
                    print "running"
                    return
            print "stopped"
        else:
            raise ValueError("Unknown daemon operation {}".format(opts.daemon))
    else:
        required_sensors = json.loads(opts.sensors_config.read())
        daemon_main(required_sensors, opts)
    return 0

if __name__ == "__main__":
    exit(main(sys.argv))
