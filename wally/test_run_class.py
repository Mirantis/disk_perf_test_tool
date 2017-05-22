import collections
from typing import List, Callable, Any, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor

from cephlib.istorage import IStorage
from cephlib.node import NodeInfo, IRPCNode
from cephlib.ssh import ConnCreds
from cephlib.storage_selectors import DevRolesConfig

from .openstack_api import OSCreds, OSConnection
from .config import Config
from .fuel_rest_api import Connection
from .result_classes import IWallyStorage


class TestRun:
    """Test run information"""
    def __init__(self, config: Config, storage: IStorage, rstorage: IWallyStorage) -> None:
        # NodesInfo list
        self.nodes_info = {}  # type: Dict[str, NodeInfo]

        # Nodes list
        self.nodes = []  # type: List[IRPCNode]

        self.build_meta = {}  # type: Dict[str,Any]
        self.clear_calls_stack = []  # type: List[Callable[['TestRun'], None]]

        # openstack credentials
        self.fuel_openstack_creds = None  # type: Optional[OSCreds]
        self.fuel_version = None  # type: Optional[List[int]]
        self.os_creds = None  # type: Optional[OSCreds]
        self.os_connection = None  # type: Optional[OSConnection]
        self.fuel_conn = None  # type: Optional[Connection]
        self.rpc_code = None  # type: bytes
        self.default_rpc_plugins = None  # type: Dict[str, bytes]

        self.storage = storage
        self.rstorage = rstorage
        self.config = config
        self.sensors_run_on = set()  # type: Set[str]
        self.os_spawned_nodes_ids = None  # type: List[int]
        self.devs_locator = []  # type: DevRolesConfig

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
