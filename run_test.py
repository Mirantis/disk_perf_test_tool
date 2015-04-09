import os
import pickle
import sys
import json
import Queue
import pprint
import logging
import argparse
import threading
import collections

from concurrent.futures import ThreadPoolExecutor

import report
# import formatters

import utils
import ssh_utils
import start_vms
from nodes import discover
from nodes.node import Node
from config import cfg_dict, parse_config
from tests.itest import IOPerfTest, PgBenchTest
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

    # logger.setLevel(logging.INFO)
    # logger.addHandler(logging.FileHandler('log.txt'))


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
    logger.info("All nodes connected successfully")


def save_sensors_data(q):
    logger.info("Start receiving sensors data")
    sensor_data = []
    while True:
        val = q.get()
        if val is None:
            print sensor_data
            q.put(sensor_data)
            break
        sensor_data.append(val)
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

    for test in config['tests']:
        for test in config['tests'][test]['internal_tests']:
            for name, params in test.items():
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

                results = []
                while not res_q.empty():
                    results.append(res_q.get())
                    # logger.info("Get test result {0!r}".format(results[-1]))
                yield name, results


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run disk io performance test")

    parser.add_argument("-l", dest='extra_logs',
                        action='store_true', default=False,
                        help="print some extra log info")

    parser.add_argument("-b", '--build_description',
                        type=str, default="Build info")
    parser.add_argument("-i", '--build_id', type=str, default="id")
    parser.add_argument("-t", '--build_type', type=str, default="GA")
    parser.add_argument("-u", '--username', type=str, default="admin")
    parser.add_argument("-o", '--output-dest', nargs="*")
    parser.add_argument("config_file", nargs="?", default="config.yaml")

    return parser.parse_args(argv[1:])


def log_nodes_statistic(_, ctx):
    nodes = ctx.nodes
    logger.info("Found {0} nodes total".format(len(nodes)))
    per_role = collections.defaultdict(lambda: 0)
    for node in nodes:
        for role in node.roles:
            per_role[role] += 1

    for role, count in sorted(per_role.items()):
        logger.debug("Found {0} nodes with role {1}".format(count, role))


def log_sensors_config(cfg):
    pass


def connect_stage(cfg, ctx):
    ctx.clear_calls_stack.append(disconnect_stage)
    connect_all(ctx.nodes)


def discover_stage(cfg, ctx):
    if 'discover' in cfg:
        discover_objs = [i.strip() for i in cfg['discover'].strip().split(",")]
        ctx.nodes.extend(discover.discover(discover_objs, cfg['clouds']))

    for url, roles in cfg.get('explicit_nodes', {}).items():
        ctx.nodes.append(Node(url, roles.split(",")))


def deploy_sensors_stage(cfg_dict, ctx):
    ctx.clear_calls_stack.append(remove_sensors_stage)
    if 'sensors' not in cfg_dict:
        return

    cfg = cfg_dict.get('sensors')
    sens_cfg = []

    for role, sensors_str in cfg["roles_mapping"].items():
        sensors = [sens.strip() for sens in sensors_str.split(",")]

        collect_cfg = dict((sensor, {}) for sensor in sensors)

        for node in ctx.nodes:
            if role in node.roles:
                sens_cfg.append((node.connection, collect_cfg))

    log_sensors_config(sens_cfg)

    ctx.sensor_cm = start_monitoring(cfg["receiver_uri"], None,
                                     connected_config=sens_cfg)

    ctx.sensors_control_queue = ctx.sensor_cm.__enter__()

    th = threading.Thread(None, save_sensors_data, None,
                          (ctx.sensors_control_queue,))
    th.daemon = True
    th.start()
    ctx.sensor_listen_thread = th


def remove_sensors_stage(cfg, ctx):
    ctx.sensors_control_queue.put(None)
    ctx.sensor_listen_thread.join()
    ctx.sensor_data = ctx.sensors_control_queue.get()


def run_all_test(cfg, ctx, store_nodes):
    ctx.results = []

    if 'start_test_nodes' in cfg['tests']:
        params = cfg['tests']['start_test_nodes']['openstack']

    for new_node in start_vms.launch_vms(params):
        new_node.roles.append('testnode')
        ctx.nodes.append(new_node)

    if 'tests' in cfg:
        store_nodes(ctx.nodes)
        ctx.results.extend(run_tests(cfg_dict, ctx.nodes))


def shut_down_vms(cfg, ctx):
    with open('vm_journal.log') as f:
        data = str(f.read())
        nodes = pickle.loads(data)

        for node in nodes:
            logger.info("Node " + str(node) + " has been loaded")

        logger.info("Removing nodes")
        start_vms.clear_nodes()
        logger.info("Nodes has been removed")


def store_nodes(nodes):
    with open('vm_journal.log', 'w+') as f:
        f.write(pickle.dumps([nodes]))
        for node in nodes:
            logger.info("Node " + str(node) + " has been stored")


def clear_enviroment(cfg, ctx):
    if os.path.exists('vm_journal.log'):
        shut_down_vms(cfg, ctx)
        os.remove('vm_journal.log')


def run_tests_stage(cfg, ctx):
    # clear nodes that possible were created on previous test running
    clear_enviroment(cfg, ctx)
    ctx.clear_calls_stack.append(shut_down_vms)
    run_all_test(cfg, ctx, store_nodes)


def disconnect_stage(cfg, ctx):
    for node in ctx.nodes:
        if node.connection is not None:
            node.connection.close()


def report_stage(cfg, ctx):
    output_dest = cfg.get('output_dest')
    if output_dest is not None:
        if output_dest.endswith(".html"):
            report.render_html_results(ctx, output_dest)
            logger.info("Results were stored in %s" % output_dest)
        else:
            with open(output_dest, "w") as fd:
                data = {"sensor_data": ctx.sensor_data,
                        "results": ctx.results}
                fd.write(json.dumps(data))
    else:
        print "=" * 20 + " RESULTS " + "=" * 20
        pprint.pprint(ctx.results)
        print "=" * 60


def complete_log_nodes_statistic(cfg, ctx):
    nodes = ctx.nodes
    for node in nodes:
        logger.debug(str(node))


class Context(object):
    def __init__(self):
        self.build_meta = {}
        self.nodes = []
        self.clear_calls_stack = []


def load_config(path):
    global cfg_dict
    cfg_dict = parse_config(path)


def main(argv):
    opts = parse_args(argv)

    level = logging.DEBUG if opts.extra_logs else logging.WARNING
    setup_logger(logger, level)

    stages = [
        discover_stage,
        log_nodes_statistic,
        complete_log_nodes_statistic,
        connect_stage,
        # complete_log_nodes_statistic,
        deploy_sensors_stage,
        run_tests_stage,
        # report_stage
    ]

    load_config(opts.config_file)

    ctx = Context()
    ctx.build_meta['build_id'] = opts.build_id
    ctx.build_meta['build_descrption'] = opts.build_description
    ctx.build_meta['build_type'] = opts.build_type
    ctx.build_meta['username'] = opts.username

    try:
        for stage in stages:
            logger.info("Start {0.__name__} stage".format(stage))
            print "Start {0.__name__} stage".format(stage)
            stage(cfg_dict, ctx)
    finally:
        exc, cls, tb = sys.exc_info()
        for stage in ctx.clear_calls_stack[::-1]:
            try:
                logger.info("Start {0.__name__} stage".format(stage))
                stage(cfg_dict, ctx)
            except:
                pass

        if exc is not None:
            raise exc, cls, tb

    return 0


if __name__ == '__main__':
    exit(main(sys.argv))
