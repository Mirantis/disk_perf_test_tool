class TestRun:
    """Test run information"""
    def __init__(self):
        # NodesInfo list
        self.nodes_info = []

        # Nodes list
        self.nodes = []

        self.build_meta = {}
        self.clear_calls_stack = []

        # created openstack nodes
        self.openstack_nodes_ids = []
        self.sensors_mon_q = None

        # openstack credentials
        self.fuel_openstack_creds = None


