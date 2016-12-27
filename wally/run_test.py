import time
import json
import logging
from concurrent.futures import Future
from typing import List, Dict, Tuple, Optional, Union, cast

from . import utils, ssh_utils, hw_info
from .config import ConfigBlock
from .node import setup_rpc, connect
from .node_interfaces import NodeInfo, IRPCNode
from .stage import Stage, StepOrder
from .sensors import collect_sensors_data
from .suits.io.fio import IOPerfTest
from .suits.itest import TestInputConfig
from .suits.mysql import MysqlTest
from .suits.omgbench import OmgTest
from .suits.postgres import PgBenchTest
from .test_run_class import TestRun
from .utils import StopTestError


TOOL_TYPE_MAPPER = {
    "io": IOPerfTest,
    "pgbench": PgBenchTest,
    "mysql": MysqlTest,
    "omg": OmgTest,
}


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

    def cleanup(self, ctx: TestRun) -> None:
        if ctx.config.get("download_rpc_logs", False):
            for node in ctx.nodes:
                if node.rpc_log_file is not None:
                    nid = node.info.node_id()
                    path = "rpc_logs/" + nid
                    node.conn.server.flush_logs()
                    log = node.get_file_content(node.rpc_log_file)
                    if path in ctx.storage:
                        ctx.storage.append_raw(log, path)
                    else:
                        ctx.storage.put_raw(log, path)
                    logger.debug("RPC log from node {} stored into storage::{}".format(nid, path))

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
                nid = node.info.node_id()
                hw_info_path = "hw_info/{}".format(nid)
                if hw_info_path not in ctx.storage:
                    futures[(hw_info_path, nid)] = pool.submit(hw_info.get_hw_info, node)

            for (path, nid), future in futures.items():
                try:
                    ctx.storage.put(future.result(), path)
                except Exception:
                    logger.exception("During collecting hardware info from %s", nid)
                    raise utils.StopTestError()

            futures.clear()
            for node in ctx.nodes:
                nid = node.info.node_id()
                sw_info_path = "sw_info/{}".format(nid)
                if sw_info_path not in ctx.storage:
                    futures[(sw_info_path, nid)] = pool.submit(hw_info.get_sw_info, node)

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
        if 'all_nodes' in ctx.storage:
            logger.info("Skip explicid nodes filling, as all_nodes all ready in storage")
            return

        for url, roles in ctx.config.get('nodes', {}).raw().items():
            ctx.merge_node(ssh_utils.parse_ssh_uri(url), set(roles.split(",")))
            logger.debug("Add node %s with roles %s", url, roles)


class SaveNodesStage(Stage):
    """Save nodes list to file"""

    priority = StepOrder.CONNECT

    def run(self, ctx: TestRun) -> None:
        ctx.storage.put_list(ctx.nodes_info.values(), 'all_nodes')


class SleepStage(Stage):
    """Save nodes list to file"""

    priority = StepOrder.TEST
    config_block = 'sleep'

    def run(self, ctx: TestRun) -> None:
        logger.debug("Will sleep for %r seconds", ctx.config.sleep)
        time.sleep(ctx.config.sleep)


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
        for test_group in ctx.config.get('tests', []):
            if not ctx.config.no_tests:
                test_nodes = [node for node in ctx.nodes if 'testnode' in node.info.roles]

                if not test_nodes:
                    logger.error("No test nodes found")
                    raise StopTestError()

                for name, params in test_group.items():
                    vm_count = params.get('node_limit', None)  # type: Optional[int]

                    # select test nodes
                    if vm_count is None:
                        curr_test_nodes = test_nodes
                    else:
                        curr_test_nodes = test_nodes[:vm_count]

                    if not curr_test_nodes:
                        logger.error("No nodes found for test, skipping it.")
                        continue

                    test_cls = TOOL_TYPE_MAPPER[name]
                    remote_dir = ctx.config.default_test_local_folder.format(name=name, uuid=ctx.config.run_uuid)
                    test_cfg = TestInputConfig(test_cls.__name__,
                                               params=params,
                                               run_uuid=ctx.config.run_uuid,
                                               nodes=test_nodes,
                                               storage=ctx.storage,
                                               remote_dir=remote_dir)

                    test_cls(test_cfg,
                             on_idle=lambda: collect_sensors_data(ctx, False)).run()

    @classmethod
    def validate_config(cls, cfg: ConfigBlock) -> None:
        pass


class LoadStoredNodesStage(Stage):
    priority = StepOrder.DISCOVER

    def run(self, ctx: TestRun) -> None:
        if 'all_nodes' in ctx.storage:
            if ctx.nodes_info:
                logger.error("Internal error: Some nodes already stored in " +
                             "nodes_info before LoadStoredNodesStage stage")
                raise StopTestError()
            ctx.nodes_info = {node.node_id(): node
                              for node in ctx.storage.load_list(NodeInfo, "all_nodes")}
            logger.info("%s nodes loaded from database", len(ctx.nodes_info))
