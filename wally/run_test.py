import time
import logging
from concurrent.futures import Future
from typing import List, Dict, Tuple, Optional, Union, cast

from . import utils, ssh_utils, hw_info
from .config import ConfigBlock
from .node import setup_rpc, connect
from .node_interfaces import NodeInfo, IRPCNode
from .stage import Stage, StepOrder
from .suits.io.fio import IOPerfTest
from .suits.itest import TestInputConfig
from .suits.mysql import MysqlTest
from .suits.omgbench import OmgTest
from .suits.postgres import PgBenchTest
from .test_run_class import TestRun


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
                msg = "Can't connect to testnode(s) " + ",".join(map(str, failed_testnodes))
                logger.error(msg)
                raise utils.StopTestError(msg)

            if not failed_nodes:
                logger.info("All nodes connected successfully")

    def cleanup(self, ctx: TestRun) -> None:
        # TODO(koder): what next line was for?
        # ssh_utils.close_all_sessions()

        if ctx.config.get("download_rpc_logs", False):
            for node in ctx.nodes:
                if node.rpc_log_file is not None:
                    nid = node.info.node_id()
                    path = "rpc_logs/" + nid
                    node.conn.server.flush_logs()
                    log = node.get_file_content(node.rpc_log_file)
                    ctx.storage.store_raw(log, path)
                    logger.debug("RPC log from node {} stored into storage::{}".format(nid, path))

        with ctx.get_pool() as pool:
            list(pool.map(lambda node: node.disconnect(stop=True), ctx.nodes))


class CollectInfoStage(Stage):
    """Collect node info"""

    priority = StepOrder.START_SENSORS - 1
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
                    ctx.storage[path] = future.result()
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
                    ctx.storage[path] = future.result()
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
        ctx.storage['all_nodes'] = list(ctx.nodes_info.values())   # type: ignore


class SleepStage(Stage):
    """Save nodes list to file"""

    priority = StepOrder.TEST
    config_block = 'sleep'

    def run(self, ctx: TestRun) -> None:
        logger.debug("Will sleep for %r seconds", ctx.config.sleep)
        time.sleep(ctx.config.sleep)


class RunTestsStage(Stage):

    priority = StepOrder.TEST
    config_block = 'tests'

    def run(self, ctx: TestRun) -> None:
        for test_group in ctx.config.get('tests', []):
            if not ctx.config.no_tests:
                test_nodes = [node for node in ctx.nodes if 'testnode' in node.info.roles]

                if not test_nodes:
                    logger.error("No test nodes found")
                    return

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

                    test_cls(test_cfg).run()

    @classmethod
    def validate_config(cls, cfg: ConfigBlock) -> None:
        pass
