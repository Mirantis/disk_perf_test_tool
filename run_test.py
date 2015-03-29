import sys
import Queue
import pprint
import logging
import argparse
import threading
import collections

from concurrent.futures import ThreadPoolExecutor

import utils
import ssh_utils
from nodes import discover
from nodes.node import Node
from config import cfg_dict
from itest import IOPerfTest, PgBenchTest

from sensors.api import start_monitoring


logger = logging.getLogger("io-perf-tool")


def setup_logger(logger, level=logging.DEBUG):
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)

    log_format = '%(asctime)s - %(levelname)-6s - %(name)s - %(message)s'
    formatter = logging.Formatter(log_format,
                                  "%H:%M:%S")
    ch.setFormatter(formatter)


def format_result(res, formatter):
    data = "\n{0}\n".format("=" * 80)
    data += pprint.pformat(res) + "\n"
    data += "{0}\n".format("=" * 80)
    templ = "{0}\n\n====> {1}\n\n{2}\n\n"
    return templ.format(data, formatter(res), "=" * 80)


def connect_one(node):
    try:
        ssh_pref = "ssh://"
        if node.conn_url.startswith(ssh_pref):
            url = node.conn_url[len(ssh_pref):]
            node.connection = ssh_utils.connect(url)
        else:
            raise ValueError("Unknown url type {0}".format(node.conn_url))
    except Exception:
        logger.exception("During connect to {0}".format(node))


def connect_all(nodes):
    logger.info("Connecting to nodes")
    with ThreadPoolExecutor(32) as pool:
        list(pool.map(connect_one, nodes))


def save_sensors_data(q):
    logger.info("Start receiving sensors data")
    while True:
        val = q.get()
        if val is None:
            break
        # logger.debug("Sensors -> {0!r}".format(val))
    logger.info("Sensors thread exits")


def test_thread(test, node, barrier):
    try:
        logger.debug("Run preparation for {0}".format(node.conn_url))
        test.pre_run(node.connection)
        logger.debug("Run test for {0}".format(node.conn_url))
        test.run(node.connection, barrier)
    except:
        logger.exception("In test {0} for node {1}".format(test, node))


def run_tests(config, nodes):
    tool_type_mapper = {
        "io": IOPerfTest,
        "pgbench": PgBenchTest,
    }

    test_nodes = [node for node in nodes
                  if 'testnode' in node.roles]

    res_q = Queue.Queue()

    for name, params in config['tests'].items():
        logger.info("Starting {0} tests".format(name))

        threads = []
        barrier = utils.Barrier(len(test_nodes))
        for node in test_nodes:
            msg = "Starting {0} test on {1} node"
            logger.debug(msg.format(name, node.conn_url))
            test = tool_type_mapper[name](params, res_q.put)
            th = threading.Thread(None, test_thread, None,
                                  (test, node, barrier))
            threads.append(th)
            th.daemon = True
            th.start()

        for th in threads:
            th.join()

        while not res_q.empty():
            logger.info("Get test result {0!r}".format(res_q.get()))


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run disk io performance test")

    parser.add_argument("-l", dest='extra_logs',
                        action='store_true', default=False,
                        help="print some extra log info")

    parser.add_argument('stages', nargs="+",
                        choices=["discover", "connect", "start_new_nodes",
                                 "deploy_sensors", "run_tests"])

    return parser.parse_args(argv[1:])


def log_nodes_statistic(nodes):
    logger.info("Found {0} nodes total".format(len(nodes)))
    per_role = collections.defaultdict(lambda: 0)
    for node in nodes:
        for role in node.roles:
            per_role[role] += 1

    for role, count in sorted(per_role.items()):
        logger.debug("Found {0} nodes with role {1}".format(count, role))


def log_sensors_config(cfg):
    pass


def main(argv):
    opts = parse_args(argv)

    level = logging.DEBUG if opts.extra_logs else logging.WARNING
    setup_logger(logger, level)

    nodes = []

    if 'discover' in opts.stages:
        logger.info("Start node discovery")
        nodes = discover.discover(cfg_dict.get('discover'))

    if 'explicit_nodes' in cfg_dict:
        for url, roles in cfg_dict['explicit_nodes'].items():
            nodes.append(Node(url, roles.split(",")))

    log_nodes_statistic(nodes)

    if 'connect' in opts.stages:
        connect_all(nodes)

    if 'deploy_sensors' in opts.stages:
        logger.info("Deploing sensors")
        cfg = cfg_dict.get('sensors')
        sens_cfg = []

        for role, sensors_str in cfg["roles_mapping"].items():
            sensors = [sens.strip() for sens in sensors_str.split(",")]

            collect_cfg = dict((sensor, {}) for sensor in sensors)

            for node in nodes:
                if role in node.roles:
                    sens_cfg.append((node.connection, collect_cfg))

        log_sensors_config(sens_cfg)

        sensor_cm = start_monitoring(cfg["receiver_uri"], None,
                                     connected_config=sens_cfg)

        with sensor_cm as sensors_control_queue:
            th = threading.Thread(None, save_sensors_data, None,
                                  (sensors_control_queue,))
            th.daemon = True
            th.start()

            # TODO: wait till all nodes start to send sensors data

            if 'run_tests' in opts.stages:
                run_tests(cfg_dict, nodes)

            sensors_control_queue.put(None)
            th.join()
    elif 'run_tests' in opts.stages:
        run_tests(cfg_dict, nodes)

    logger.info("Disconnecting")
    for node in nodes:
        node.connection.close()

    return 0


if __name__ == '__main__':
    exit(main(sys.argv))
