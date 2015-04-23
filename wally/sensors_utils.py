import time
import Queue
import logging
import threading

from wally import utils
from wally.config import cfg_dict
from wally.sensors.api import (start_listener_thread,
                               deploy_and_start_sensors,
                               SensorConfig,
                               stop_and_remove_sensors)


logger = logging.getLogger("wally")
DEFAULT_RECEIVER_URL = "udp://{ip}:5699"


def save_sensors_data(data_q, mon_q, fd):
    fd.write("\n")

    observed_nodes = set()

    try:
        while True:
            val = data_q.get()
            if val is None:
                break

            addr, data = val

            if addr not in observed_nodes:
                mon_q.put(addr + (data['source_id'],))
                observed_nodes.add(addr)

            fd.write("{0!s} : {1!r}\n".format(time.time(),
                     repr((addr, data))))
    except Exception:
        logger.exception("Error in sensors thread")
    logger.info("Sensors thread exits")


def get_sensors_config_for_nodes(cfg, nodes):
    monitored_nodes = []
    sensors_configs = []

    receiver_url = cfg.get("receiver_url", DEFAULT_RECEIVER_URL)
    assert '{ip}' in receiver_url

    for role, sensors_str in cfg["roles_mapping"].items():
        sensors = [sens.strip() for sens in sensors_str.split(",")]

        collect_cfg = dict((sensor, {}) for sensor in sensors)

        for node in nodes:
            if role in node.roles:

                if node.monitor_url is not None:
                    monitor_url = node.monitor_url
                else:
                    ip = node.get_ip()
                    if ip == '127.0.0.1':
                        ext_ip = '127.0.0.1'
                    else:
                        ext_ip = utils.get_ip_for_target(ip)
                    monitor_url = receiver_url.format(ip=ext_ip)

                monitored_nodes.append(node)
                sens_cfg = SensorConfig(node.connection,
                                        node.get_conn_id(),
                                        collect_cfg,
                                        source_id=node.get_conn_id(),
                                        monitor_url=monitor_url)
                sensors_configs.append(sens_cfg)

    return monitored_nodes, sensors_configs


def start_sensor_process_thread(ctx, cfg, sensors_configs):
    receiver_url = cfg.get('receiver_url', DEFAULT_RECEIVER_URL)
    sensors_data_q, stop_sensors_loop = \
        start_listener_thread(receiver_url.format(ip='0.0.0.0'))

    mon_q = Queue.Queue()
    fd = open(cfg_dict['sensor_storage'], "w")
    sensor_listen_th = threading.Thread(None, save_sensors_data, None,
                                        (sensors_data_q, mon_q, fd))
    sensor_listen_th.daemon = True
    sensor_listen_th.start()

    def stop_sensors_receiver(cfg, ctx):
        stop_sensors_loop()
        sensors_data_q.put(None)
        sensor_listen_th.join()

    ctx.clear_calls_stack.append(stop_sensors_receiver)
    return mon_q


def deploy_sensors_stage(cfg, ctx, nodes=None, undeploy=True):
    if 'sensors' not in cfg:
        return

    cfg = cfg.get('sensors')

    if nodes is None:
        nodes = ctx.nodes

    monitored_nodes, sensors_configs = get_sensors_config_for_nodes(cfg,
                                                                    nodes)

    if len(monitored_nodes) == 0:
        logger.info("Nothing to monitor, no sensors would be installed")
        return

    if ctx.sensors_mon_q is None:
        logger.info("Start sensors data receiving thread")
        ctx.sensors_mon_q = start_sensor_process_thread(ctx, cfg,
                                                        sensors_configs)

    if undeploy:
        def remove_sensors_stage(cfg, ctx):
            stop_and_remove_sensors(sensors_configs)
        ctx.clear_calls_stack.append(remove_sensors_stage)

    logger.info("Deploing new sensors on {0} node(s)".format(len(nodes)))
    deploy_and_start_sensors(sensors_configs)
    wait_for_new_sensors_data(ctx, monitored_nodes)


def wait_for_new_sensors_data(ctx, monitored_nodes):
    MAX_WAIT_FOR_SENSORS = 10
    etime = time.time() + MAX_WAIT_FOR_SENSORS

    msg = "Waiting at most {0}s till all {1} nodes starts report sensors data"
    nodes_ids = set(node.get_conn_id() for node in monitored_nodes)
    logger.debug(msg.format(MAX_WAIT_FOR_SENSORS, len(nodes_ids)))

    # wait till all nodes start sending data
    while len(nodes_ids) != 0:
        tleft = etime - time.time()
        try:
            source_id = ctx.sensors_mon_q.get(True, tleft)[2]
        except Queue.Empty:
            msg = "Node {0} not sending any sensor data in {1}s"
            msg = msg.format(", ".join(nodes_ids), MAX_WAIT_FOR_SENSORS)
            raise RuntimeError(msg)

        if source_id not in nodes_ids:
            msg = "Receive sensors from extra node: {0}".format(source_id)
            logger.warning(msg)

        nodes_ids.remove(source_id)
