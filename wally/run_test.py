import time
import json
import copy
import logging
from concurrent.futures import Future
from typing import List, Dict, Tuple, Optional, Union, cast

from cephlib.wally_storage import WallyDB
from cephlib.node import NodeInfo, IRPCNode, get_hw_info, get_sw_info
from cephlib.ssh import parse_ssh_uri
from cephlib.node_impl import setup_rpc, connect

from . import utils
from .config import ConfigBlock
from .stage import Stage, StepOrder
from .sensors import collect_sensors_data
from .suits.all_suits import all_suits
from .test_run_class import TestRun
from .result_classes import SuiteConfig


logger = logging.getLogger("wally")


class ConnectStage(Stage):
    """Connect to nodes stage"""

    priority = StepOrder.CONNECT

    def run(self, ctx: TestRun) -> None:
        with ctx.get_pool() as pool:
            logger.info("Connecting to %s nodes", len(ctx.nodes_info))

            def connect_ext(node_info: NodeInfo) -> Tuple[bool, Union[IRPCNode, NodeInfo]]:
                try:
                    ssh_node = connect(node_info, conn_timeout=ctx.config.connect_timeout)

                    return True, setup_rpc(ssh_node,
                                           ctx.rpc_code,
                                           ctx.default_rpc_plugins,
                                           log_level=ctx.config.rpc_log_level)
                except Exception as exc:
                    logger.exception("During connect to %s: %s", node_info, exc)
                    return False, node_info

            failed_testnodes = []  # type: List[NodeInfo]
            failed_nodes = []  # type: List[NodeInfo]
            ctx.nodes = []

            for ok, node in pool.map(connect_ext, ctx.nodes_info.values()):
                if not ok:
                    node = cast(NodeInfo, node)
                    if 'testnode' in node.roles:
                        failed_testnodes.append(node)
                    else:
                        failed_nodes.append(node)
                else:
                    ctx.nodes.append(cast(IRPCNode, node))

            if failed_nodes:
                msg = "Node(s) {} would be excluded - can't connect"
                logger.warning(msg.format(", ".join(map(str, failed_nodes))))

            if failed_testnodes:
                msg = "Can't start RPC on testnode(s) " + ",".join(map(str, failed_testnodes))
                logger.error(msg)
                raise utils.StopTestError(msg)

            if not failed_nodes:
                logger.info("All nodes connected successfully")

            def get_time(node):
                return node.conn.sys.time()

            t_start = time.time()
            tms = pool.map(get_time, ctx.nodes)
            t_end = time.time()

            for node, val in zip(ctx.nodes, tms):
                delta = 0
                if val > t_end:
                    delta = val - t_end
                elif t_start > val:
                    delta = t_start - val

                if delta > ctx.config.max_time_diff_ms:
                    msg = ("Too large time shift {}ms on node {}. Stopping test." +
                           " Fix time on cluster nodes and restart test, or change " +
                           "max_time_diff_ms(={}ms) setting in config").format(delta,
                                                                               str(node),
                                                                               ctx.config.max_time_diff_ms)
                    logger.error(msg)
                    raise utils.StopTestError(msg)
                if delta > 0:
                    logger.warning("Node %s has time shift at least %s ms", node, delta)


    def cleanup(self, ctx: TestRun) -> None:
        if ctx.config.get("download_rpc_logs", False):
            logger.info("Killing all outstanding processes")
            for node in ctx.nodes:
                node.conn.cli.killall()

            logger.info("Downloading RPC servers logs")
            for node in ctx.nodes:
                node.conn.cli.killall()
                if node.rpc_log_file is not None:
                    nid = node.node_id
                    path = WallyDB.rpc_logs.format(node_id=nid)
                    node.conn.server.flush_logs()
                    log = node.get_file_content(node.rpc_log_file)
                    if path in ctx.storage:
                        ctx.storage.append_raw(log, path)
                    else:
                        ctx.storage.put_raw(log, path)
                    logger.debug("RPC log from node {} stored into storage::{}".format(nid, path))

        logger.info("Disconnecting")
        with ctx.get_pool() as pool:
            list(pool.map(lambda node: node.disconnect(stop=True), ctx.nodes))


class CollectInfoStage(Stage):
    """Collect node info"""

    priority = StepOrder.START_SENSORS - 2
    config_block = 'collect_info'

    def run(self, ctx: TestRun) -> None:
        if not ctx.config.collect_info:
            return

        futures = {}  # type: Dict[Tuple[str, str], Future]

        with ctx.get_pool() as pool:
            # can't make next RPC request until finish with previous
            for node in ctx.nodes:
                nid = node.node_id
                hw_info_path = WallyDB.hw_info.format(node_id=nid)
                if hw_info_path not in ctx.storage:
                    futures[(hw_info_path, nid)] = pool.submit(get_hw_info, node)

            for (path, nid), future in futures.items():
                try:
                    ctx.storage.put(future.result(), path)
                except Exception:
                    logger.exception("During collecting hardware info from %s", nid)
                    raise utils.StopTestError()

            futures.clear()
            for node in ctx.nodes:
                nid = node.node_id
                sw_info_path = WallyDB.sw_info.format(node_id=nid)
                if sw_info_path not in ctx.storage:
                    futures[(sw_info_path, nid)] = pool.submit(get_sw_info, node)

            for (path, nid), future in futures.items():
                try:
                    ctx.storage.put(future.result(), path)
                except Exception:
                    logger.exception("During collecting software info from %s", nid)
                    raise utils.StopTestError()


class ExplicitNodesStage(Stage):
    """add explicit nodes"""

    priority = StepOrder.DISCOVER
    config_block = 'nodes'

    def run(self, ctx: TestRun) -> None:
        if WallyDB.all_nodes in ctx.storage:
            logger.info("Skip explicid nodes filling, as all_nodes all ready in storage")
            return

        for url, roles in ctx.config.get('nodes', {}).raw().items():
            ctx.merge_node(parse_ssh_uri(url), set(role.strip() for role in roles.split(",")))
            logger.debug("Add node %s with roles %s", url, roles)


class SleepStage(Stage):
    """Save nodes list to file"""

    priority = StepOrder.TEST
    config_block = 'sleep'

    def run(self, ctx: TestRun) -> None:
        logger.debug("Will sleep for %r seconds", ctx.config.sleep)
        stime = time.time()
        time.sleep(ctx.config.sleep)
        ctx.storage.put([int(stime), int(time.time())], 'idle')


class PrepareNodes(Stage):
    priority = StepOrder.START_SENSORS - 1

    def __init__(self):
        Stage.__init__(self)
        self.nodeepscrub_updated = False
        self.noscrub_updated = False

    def run(self, ctx: TestRun) -> None:
        ceph_sett = ctx.config.get('ceph_settings', "").split()
        if ceph_sett:
            for node in ctx.nodes:
                if "ceph-mon" in node.info.roles or "ceph-osd" in node.info.roles:
                    state = json.loads(node.run("ceph health --format json"))["summary"]["summary"]
                    if 'noscrub' in ceph_sett:
                        if 'noscrub' in state:
                            logger.debug("noscrub already set on cluster")
                        else:
                            logger.info("Applying noscrub settings to ceph cluster")
                            node.run("ceph osd set noscrub")
                            self.noscrub_updated = True

                    if 'nodeepscrub' in ceph_sett:
                        if 'nodeepscrub' in state:
                            logger.debug("noscrub already set on cluster")
                        else:
                            logger.info("Applying noscrub settings to ceph cluster")
                            node.run("ceph osd set noscrub")
                            self.nodeepscrub_updated = True
                    break

    def cleanup(self, ctx: TestRun) -> None:
        if self.nodeepscrub_updated or self.noscrub_updated:
            for node in ctx.nodes:
                if "ceph-mon" in node.info.roles or "ceph-osd" in node.info.roles :
                    if self.noscrub_updated:
                        logger.info("Reverting noscrub setting for ceph cluster")
                        node.run("ceph osd unset noscrub")
                        self.noscrub_updated = False

                    if self.nodeepscrub_updated:
                        logger.info("Reverting noscrub setting for ceph cluster")
                        node.run("ceph osd unset nodeepscrub")
                        self.nodeepscrub_updated = False


class RunTestsStage(Stage):

    priority = StepOrder.TEST
    config_block = 'tests'

    def run(self, ctx: TestRun) -> None:
        if ctx.config.no_tests:
            logger.info("Skiping tests, as 'no_tests' config settings is True")
            return

        for suite_idx, test_suite in enumerate(ctx.config.get('tests', [])):
            test_nodes = [node for node in ctx.nodes if 'testnode' in node.info.roles]

            if not test_nodes:
                logger.error("No test nodes found")
                raise utils.StopTestError()

            if len(test_suite) != 1:
                logger.error("Test suite %s contain more than one test. Put each test in separated group", suite_idx)
                raise utils.StopTestError()

            name, params = list(test_suite.items())[0]
            vm_count = params.get('node_limit', None)  # type: Optional[int]

            # select test nodes
            if vm_count is None:
                curr_test_nodes = test_nodes
            else:
                curr_test_nodes = test_nodes[:vm_count]

            if not curr_test_nodes:
                logger.error("No nodes found for test, skipping it.")
                continue

            if name not in all_suits:
                logger.error("Test suite %r not found. Only suits [%s] available", name, ", ".join(all_suits))
                raise utils.StopTestError()

            test_cls = all_suits[name]
            remote_dir = ctx.config.default_test_local_folder.format(name=name, uuid=ctx.config.run_uuid)
            suite = SuiteConfig(test_cls.name,
                                params=params,
                                run_uuid=ctx.config.run_uuid,
                                nodes=test_nodes,
                                remote_dir=remote_dir,
                                idx=suite_idx,
                                keep_raw_files=ctx.config.keep_raw_files)

            test_cls(storage=ctx.rstorage,
                     suite=suite,
                     on_idle=lambda: collect_sensors_data(ctx, False)).run()

    @classmethod
    def validate_config(cls, cfg: ConfigBlock) -> None:
        pass


class SaveNodesStage(Stage):
    """Save nodes list to file"""
    priority = StepOrder.UPDATE_NODES_INFO + 1

    def run(self, ctx: TestRun) -> None:
        infos = list(ctx.nodes_info.values())
        params = {node.node_id: node.params for node in infos}
        ninfos = [copy.copy(node) for node in infos]
        for node in ninfos:
            node.params = {"in file": WallyDB.nodes_params}
        ctx.storage.put_list(ninfos, WallyDB.all_nodes)
        ctx.storage.put_raw(json.dumps(params).encode('utf8'), WallyDB.nodes_params)


class LoadStoredNodesStage(Stage):
    priority = StepOrder.DISCOVER

    def run(self, ctx: TestRun) -> None:
        if WallyDB.all_nodes in ctx.storage:
            if ctx.nodes_info:
                logger.error("Internal error: Some nodes already stored in " +
                             "nodes_info before LoadStoredNodesStage stage")
                raise utils.StopTestError()

            ctx.nodes_info = {node.node_id: node for node in ctx.rstorage.load_nodes()}
            logger.info("%s nodes loaded from database", len(ctx.nodes_info))
