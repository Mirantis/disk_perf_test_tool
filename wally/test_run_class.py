from typing import List, Callable, Any, Dict, Optional, Set, Union
from concurrent.futures import ThreadPoolExecutor


from .timeseries import SensorDatastore
from .node_interfaces import NodeInfo, IRPCNode
from .openstack_api import OSCreds, OSConnection
from .storage import Storage
from .config import Config
from .fuel_rest_api import Connection
from .ssh_utils import ConnCreds


class TestRun:
    """Test run information"""
    def __init__(self, config: Config, storage: Storage) -> None:
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
        self.config = config
        self.sensors_data = SensorDatastore()
        self.sensors_run_on = set()  # type: Set[str]
        self.os_spawned_nodes_ids = None  # type: List[int]

    def get_pool(self):
        return ThreadPoolExecutor(self.config.get('worker_pool_sz', 32))

    def merge_node(self, creds: ConnCreds, roles: Set[str]) -> NodeInfo:
        info = NodeInfo(creds, roles)
        nid = info.node_id()

        if nid in self.nodes_info:
            self.nodes_info[nid].roles.update(info.roles)
            return self.nodes_info[nid]
        else:
            self.nodes_info[nid] = info
            return info
