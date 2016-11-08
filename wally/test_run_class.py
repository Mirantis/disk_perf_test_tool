from typing import List, Callable, Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor


from .timeseries import SensorDatastore
from . import inode
from .start_vms import OSCreds
from .storage import IStorage
from .config import Config


class TestRun:
    """Test run information"""
    def __init__(self, config: Config, storage: IStorage):
        # NodesInfo list
        self.nodes_info = []  # type: List[inode.NodeInfo]

        # Nodes list
        self.nodes = []  # type: List[inode.INode]

        self.build_meta = {}  # type: Dict[str,Any]
        self.clear_calls_stack = []  # type: List[Callable[['TestRun'], None]]

        # created openstack nodes
        self.openstack_nodes_ids = []  # type: List[str]
        self.sensors_mon_q = None

        # openstack credentials
        self.fuel_openstack_creds = None  # type: Optional[OSCreds]

        self.storage = storage
        self.config = config
        self.sensors_data = SensorDatastore()

    def get_pool(self):
        return ThreadPoolExecutor(self.config.get('worker_pool_sz', 32))

