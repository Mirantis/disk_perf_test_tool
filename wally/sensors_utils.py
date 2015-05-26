import os.path
import logging
import contextlib

from concurrent.futures import ThreadPoolExecutor

from wally import ssh_utils
from wally.sensors.api import (with_sensors, sensors_info, SensorConfig)


logger = logging.getLogger("wally.sensors")


def get_sensors_config_for_nodes(cfg, nodes, remote_path):
    monitored_nodes = []
    sensors_configs = []
    source2roles_map = {}

    receiver_url = "csvfile://" + os.path.join(remote_path, "results.csv")

    for role, sensors_str in cfg["roles_mapping"].items():
        sensors = [sens.strip() for sens in sensors_str.split(",")]

        collect_cfg = dict((sensor, {}) for sensor in sensors)

        for node in nodes:
            if role in node.roles:
                source2roles_map[node.get_conn_id()] = node.roles
                monitored_nodes.append(node)
                sens_cfg = SensorConfig(node.connection,
                                        node.get_conn_id(),
                                        collect_cfg,
                                        source_id=node.get_conn_id(),
                                        monitor_url=receiver_url)
                sensors_configs.append(sens_cfg)

    return monitored_nodes, sensors_configs, source2roles_map


PID_FILE = "/tmp/sensors.pid"


def clear_old_sensors(sensors_configs):
    def stop_sensors(sens_cfg):
        with sens_cfg.conn.open_sftp() as sftp:
            try:
                pid = ssh_utils.read_from_remote(sftp, PID_FILE)
                pid = int(pid.strip())
                ssh_utils.run_over_ssh(sens_cfg.conn,
                                       "kill -9 " + str(pid))
                sftp.remove(PID_FILE)
            except:
                pass

    with ThreadPoolExecutor(32) as pool:
        list(pool.map(stop_sensors, sensors_configs))


@contextlib.contextmanager
def with_sensors_util(cfg, nodes):
    if 'sensors' not in cfg:
        yield
        return

    monitored_nodes, sensors_configs, source2roles_map = \
        get_sensors_config_for_nodes(cfg['sensors'], nodes,
                                     cfg['sensors_remote_path'])

    with with_sensors(sensors_configs, cfg['sensors_remote_path']):
        yield source2roles_map


@contextlib.contextmanager
def sensors_info_util(cfg, nodes):
    if 'sensors' not in cfg:
        yield None
        return

    _, sensors_configs, _ = \
        get_sensors_config_for_nodes(cfg['sensors'], nodes,
                                     cfg['sensors_remote_path'])

    clear_old_sensors(sensors_configs)
    ctx = sensors_info(sensors_configs, cfg['sensors_remote_path'])
    try:
        res = ctx.__enter__()
        yield res
    except:
        ctx.__exit__(None, None, None)
        raise
    finally:
        try:
            ctx.__exit__(None, None, None)
        except:
            logger.exception("During stop/collect sensors")
            del res[:]
