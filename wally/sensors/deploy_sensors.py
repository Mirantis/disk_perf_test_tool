import time
import json
import os.path
import logging

from concurrent.futures import ThreadPoolExecutor, wait

from wally.ssh_utils import copy_paths, run_over_ssh

logger = logging.getLogger('wally.sensors')


def wait_all_ok(futures):
    return all(future.result() for future in futures)


def deploy_and_start_sensors(sensor_configs,
                             remote_path='/tmp/sensors/sensors'):

    paths = {os.path.dirname(__file__): remote_path}
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []

        for node_sensor_config in sensor_configs:
            futures.append(executor.submit(deploy_and_start_sensor,
                                           paths,
                                           node_sensor_config,
                                           remote_path))

        if not wait_all_ok(futures):
            raise RuntimeError("Sensor deployment fails on some nodes")


def deploy_and_start_sensor(paths, node_sensor_config, remote_path):
    try:
        copy_paths(node_sensor_config.conn, paths)
        sftp = node_sensor_config.conn.open_sftp()

        config_remote_path = os.path.join(remote_path, "conf.json")

        sensors_config = node_sensor_config.sensors.copy()
        sensors_config['source_id'] = node_sensor_config.source_id
        with sftp.open(config_remote_path, "w") as fd:
            fd.write(json.dumps(sensors_config))

        cmd_templ = 'env PYTHONPATH="{0}" python -m ' + \
                    "sensors.main -d start -u {1} {2}"

        cmd = cmd_templ.format(os.path.dirname(remote_path),
                               node_sensor_config.monitor_url,
                               config_remote_path)

        run_over_ssh(node_sensor_config.conn, cmd,
                     node=node_sensor_config.url)
        sftp.close()

    except:
        msg = "During deploing sensors on {0}".format(node_sensor_config.url)
        logger.exception(msg)
        return False
    return True


def stop_and_remove_sensor(conn, url, remote_path):
    try:
        cmd = 'env PYTHONPATH="{0}" python -m sensors.main -d stop'
        cmd = cmd.format(remote_path)
        run_over_ssh(conn, cmd, node=url)
        # some magic
        time.sleep(0.3)

        # logger.warning("Sensors don't removed")
        run_over_ssh(conn, "rm -rf {0}".format(remote_path),
                     node=url, timeout=10)
    except Exception as exc:
        msg = "Failed to remove sensors from node {0}: {1!s}"
        logger.error(msg.format(url, exc))


def stop_and_remove_sensors(configs, remote_path='/tmp/sensors'):
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []

        for node_sensor_config in configs:
            futures.append(executor.submit(stop_and_remove_sensor,
                                           node_sensor_config.conn,
                                           node_sensor_config.url,
                                           remote_path))

        wait(futures)
    logger.debug("Sensors stopped and removed")
