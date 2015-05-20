import os
import time
import json
import logging
import contextlib

from concurrent.futures import ThreadPoolExecutor

from wally.ssh_utils import (copy_paths, run_over_ssh,
                             save_to_remote, read_from_remote)


logger = logging.getLogger("wally.sensors")


class SensorConfig(object):
    def __init__(self, conn, url, sensors, source_id,
                 monitor_url=None):
        self.conn = conn
        self.url = url
        self.sensors = sensors
        self.source_id = source_id
        self.monitor_url = monitor_url


@contextlib.contextmanager
def with_sensors(sensor_configs, remote_path):
    paths = {os.path.dirname(__file__):
             os.path.join(remote_path, "sensors")}
    config_remote_path = os.path.join(remote_path, "conf.json")

    def deploy_sensors(node_sensor_config):
        # check that path already exists
        copy_paths(node_sensor_config.conn, paths)
        with node_sensor_config.conn.open_sftp() as sftp:
            sensors_config = node_sensor_config.sensors.copy()
            sensors_config['source_id'] = node_sensor_config.source_id
            save_to_remote(sftp, config_remote_path,
                           json.dumps(sensors_config))

    def remove_sensors(node_sensor_config):
        run_over_ssh(node_sensor_config.conn,
                     "rm -rf {0}".format(remote_path),
                     node=node_sensor_config.url, timeout=10)

    logger.debug("Installing sensors on {0} nodes".format(len(sensor_configs)))
    with ThreadPoolExecutor(max_workers=32) as executor:
        list(executor.map(deploy_sensors, sensor_configs))
        try:
            yield
        finally:
            list(executor.map(remove_sensors, sensor_configs))


@contextlib.contextmanager
def sensors_info(sensor_configs, remote_path):
    config_remote_path = os.path.join(remote_path, "conf.json")

    def start_sensors(node_sensor_config):
        cmd_templ = 'env PYTHONPATH="{0}" python -m ' + \
                    "sensors.main -d start -u {1} {2}"

        cmd = cmd_templ.format(remote_path,
                               node_sensor_config.monitor_url,
                               config_remote_path)

        run_over_ssh(node_sensor_config.conn, cmd,
                     node=node_sensor_config.url)

    def stop_and_gather_data(node_sensor_config):
        cmd = 'env PYTHONPATH="{0}" python -m sensors.main -d stop'
        cmd = cmd.format(remote_path)
        run_over_ssh(node_sensor_config.conn, cmd,
                     node=node_sensor_config.url)
        # some magic
        time.sleep(1)

        assert node_sensor_config.monitor_url.startswith("csvfile://")

        res_path = node_sensor_config.monitor_url.split("//", 1)[1]
        with node_sensor_config.conn.open_sftp() as sftp:
            res = read_from_remote(sftp, res_path)

        return res

    results = []

    logger.debug("Starting sensors on {0} nodes".format(len(sensor_configs)))
    with ThreadPoolExecutor(max_workers=32) as executor:
        list(executor.map(start_sensors, sensor_configs))
        try:
            yield results
        finally:
            results.extend(executor.map(stop_and_gather_data, sensor_configs))
