from typing import List, Callable, Any, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor

from cephlib.istorage import IStorage
from cephlib.node import NodeInfo, IRPCNode
from cephlib.ssh import ConnCreds
from cephlib.storage_selectors import DevRolesConfig

from .openstack_api import OSCreds, OSConnection
from .config import Config
from .result_classes import IWallyStorage


class TestRun:
    """Test run information"""
    def __init__(self, config: Config, storage: IStorage, rstorage: IWallyStorage) -> None:
        # NodesInfo list
        self.nodes_info: Dict[str, NodeInfo] = {}

        self.ceph_master_node: Optional[IRPCNode] = None
        self.ceph_extra_args: Optional[str] = None

        # Nodes list
        self.nodes: List[IRPCNode] = []

        self.build_meta: Dict[str,Any] = {}
        self.clear_calls_stack: List[Callable[['TestRun'], None]] = []

        # openstack credentials
        self.os_creds: Optional[OSCreds] = None  # type: ignore
        self.os_connection: Optional[OSConnection] = None  # type: ignore
        self.rpc_code: bytes = None  # type: ignore
        self.default_rpc_plugins: Dict[str, bytes] = None  # type: ignore

        self.storage = storage
        self.rstorage = rstorage
        self.config = config
        self.sensors_run_on: Set[str] = set()
        self.os_spawned_nodes_ids: List[int] = None  # type: ignore
        self.devs_locator: DevRolesConfig = []

    def get_pool(self):
        return ThreadPoolExecutor(self.config.get('worker_pool_sz', 32))

    def merge_node(self, creds: ConnCreds, roles: Set[str], **params) -> NodeInfo:
        info = NodeInfo(creds, roles, params)
        nid = info.node_id

        if nid in self.nodes_info:
            self.nodes_info[nid].roles.update(info.roles)
            self.nodes_info[nid].params.update(info.params)
            return self.nodes_info[nid]
        else:
            self.nodes_info[nid] = info
            return info
