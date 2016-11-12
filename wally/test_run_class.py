from typing import List, Callable, Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor


from .timeseries import SensorDatastore
from .node_interfaces import NodeInfo, IRPCNode, RPCBeforeConnCallback
from .start_vms import OSCreds, NovaClient, CinderClient
from .storage import Storage
from .config import Config


class TestRun:
    """Test run information"""
    def __init__(self, config: Config, storage: Storage):
        # NodesInfo list
        self.nodes_info = []  # type: List[NodeInfo]

        # Nodes list
        self.nodes = []  # type: List[IRPCNode]

        self.build_meta = {}  # type: Dict[str,Any]
        self.clear_calls_stack = []  # type: List[Callable[['TestRun'], None]]
        self.sensors_mon_q = None

        # openstack credentials
        self.fuel_openstack_creds = None  # type: Optional[OSCreds]
        self.os_creds = None  # type: Optional[OSCreds]
        self.nova_client = None  # type: Optional[NovaClient]
        self.cinder_client = None  # type: Optional[CinderClient]

        self.storage = storage
        self.config = config
        self.sensors_data = SensorDatastore()

        self.before_conn_callback = None  # type: RPCBeforeConnCallback

    def get_pool(self):
        return ThreadPoolExecutor(self.config.get('worker_pool_sz', 32))

