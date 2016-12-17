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
                    return True, setup_rpc(ssh_node, ctx.rpc_code, ctx.default_rpc_plugins)
                except Exception as exc:
                    logger.error("During connect to {}: {!s}".format(node, exc))
                    return False, node_info

            failed_testnodes = []  # type: List[NodeInfo]
            failed_nodes = []  # type: List[NodeInfo]
            ctx.nodes = []

            for ok, node in pool.map(connect_ext, ctx.nodes_info):
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
                logger.warning(msg.format(",".join(map(str, failed_nodes))))

            if failed_testnodes:
                msg = "Can't connect to testnode(s) " + \
                      ",".join(map(str, failed_testnodes))
                logger.error(msg)
                raise utils.StopTestError(msg)

            if not failed_nodes:
                logger.info("All nodes connected successfully")

    def cleanup(self, ctx: TestRun) -> None:
        # TODO(koder): what next line was for?
        # ssh_utils.close_all_sessions()

        for node in ctx.nodes:
            node.disconnect()


class CollectInfoStage(Stage):
    """Collect node info"""

    priority = StepOrder.START_SENSORS - 1
    config_block = 'collect_info'

    def run(self, ctx: TestRun) -> None:
        if not ctx.config.collect_info:
            return

        futures = {}  # type: Dict[str, Future]

        with ctx.get_pool() as pool:
            for node in ctx.nodes:
                hw_info_path = "hw_info/{}".format(node.info.node_id())
                if hw_info_path not in ctx.storage:
                    futures[hw_info_path] = pool.submit(hw_info.get_hw_info, node), node

                sw_info_path = "sw_info/{}".format(node.info.node_id())
                if sw_info_path not in ctx.storage:
                    futures[sw_info_path] = pool.submit(hw_info.get_sw_info, node)

            for path, future in futures.items():
                ctx.storage[path] = future.result()


class ExplicitNodesStage(Stage):
    """add explicit nodes"""

    priority = StepOrder.DISCOVER
    config_block = 'nodes'

    def run(self, ctx: TestRun) -> None:
        explicit_nodes = []
        for url, roles in ctx.config.get('explicit_nodes', {}).items():
            creds = ssh_utils.parse_ssh_uri(url)
            roles = set(roles.split(","))
            explicit_nodes.append(NodeInfo(creds, roles))

        ctx.nodes_info.extend(explicit_nodes)
        ctx.storage['explicit_nodes'] = explicit_nodes  # type: ignore


class SaveNodesStage(Stage):
    """Save nodes list to file"""

    priority = StepOrder.CONNECT

    def run(self, ctx: TestRun) -> None:
        ctx.storage['all_nodes'] = ctx.nodes_info   # type: ignore


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
