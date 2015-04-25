import os
import sys
import time
import json
import glob
import signal
import os.path
import argparse

from .sensors.utils import SensorInfo
from .daemonize import Daemonize
from .discover import all_sensors
from .protocol import create_protocol


# load all sensors
from . import sensors
sensors_dir = os.path.dirname(sensors.__file__)
for fname in glob.glob(os.path.join(sensors_dir, "*.py")):
    mod_name = os.path.basename(fname[:-3])
    __import__("sensors.sensors." + mod_name)


def get_values(required_sensors):
    result = {}
    for sensor_name, params in required_sensors:
        if sensor_name in all_sensors:
            result.update(all_sensors[sensor_name](**params))
        else:
            msg = "Sensor {0!r} isn't available".format(sensor_name)
            raise ValueError(msg)
    return result


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--daemon',
                        choices=('start', 'stop', 'status'),
                        default=None)

    parser.add_argument('-u', '--url', default='stdout://')
    parser.add_argument('-t', '--timeout', type=float, default=1)
    parser.add_argument('-l', '--list-sensors', action='store_true')
    parser.add_argument('sensors_config', type=argparse.FileType('r'),
                        default=None, nargs='?')
    return parser.parse_args(args[1:])


def daemon_main(required_sensors, opts):
    try:
        source_id = str(required_sensors.pop('source_id'))
    except KeyError:
        source_id = None

    sender = create_protocol(opts.url)
    prev = {}
    next_data_record_time = time.time()

    while True:
        real_time = int(time.time())

        if real_time < int(next_data_record_time):
            if int(next_data_record_time) - real_time > 2:
                print "Error: sleep too small portion!!"
            report_time = int(next_data_record_time)
        elif real_time > int(next_data_record_time):
            if real_time - int(next_data_record_time) > 2:
                report_time = real_time
            else:
                report_time = int(next_data_record_time)
        else:
            report_time = real_time

        data = get_values(required_sensors.items())
        curr = {'time': SensorInfo(report_time, True)}
        for name, val in data.items():
            if val.is_accumulated:
                if name in prev:
                    curr[name] = SensorInfo(val.value - prev[name], True)
                prev[name] = val.value
            else:
                curr[name] = SensorInfo(val.value, False)

        if source_id is not None:
            curr['source_id'] = source_id

        print report_time, int((report_time - time.time()) * 10) * 0.1
        sender.send(curr)

        next_data_record_time = report_time + opts.timeout + 0.5
        time.sleep(next_data_record_time - time.time())


def pid_running(pid):
    return os.path.exists("/proc/" + str(pid))


def main(argv):
    opts = parse_args(argv)

    if opts.list_sensors:
        print "\n".join(sorted(all_sensors))
        return 0

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
                if pid_running(pid):
                    os.kill(pid, signal.SIGTERM)

                time.sleep(0.5)

                if pid_running(pid):
                    os.kill(pid, signal.SIGKILL)

                time.sleep(0.5)

                if os.path.isfile(pid_file):
                    os.unlink(pid_file)
        elif opts.daemon == 'status':
            if os.path.isfile(pid_file):
                pid = int(open(pid_file).read())
                if pid_running(pid):
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
