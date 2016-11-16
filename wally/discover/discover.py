import os.path
import logging
from typing import Dict, NamedTuple, List, Optional, cast

from paramiko.ssh_exception import AuthenticationException

from . import ceph
from . import fuel
from . import openstack
from ..utils import parse_creds, StopTestError
from ..config import ConfigBlock
from ..start_vms import OSCreds
from ..node_interfaces import NodeInfo
from ..node import connect, setup_rpc
from ..ssh_utils import parse_ssh_uri


logger = logging.getLogger("wally.discover")


openrc_templ = """#!/bin/sh
export LC_ALL=C
export OS_NO_CACHE='true'
export OS_TENANT_NAME='{tenant}'
export OS_USERNAME='{name}'
export OS_PASSWORD='{passwd}'
export OS_AUTH_URL='{auth_url}'
export OS_INSECURE={insecure}
export OS_AUTH_STRATEGY='keystone'
export OS_REGION_NAME='RegionOne'
export CINDER_ENDPOINT_TYPE='publicURL'
export GLANCE_ENDPOINT_TYPE='publicURL'
export KEYSTONE_ENDPOINT_TYPE='publicURL'
export NOVA_ENDPOINT_TYPE='publicURL'
export NEUTRON_ENDPOINT_TYPE='publicURL'
"""


DiscoveryResult = NamedTuple("DiscoveryResult", [("os_creds", Optional[OSCreds]), ("nodes", List[NodeInfo])])


def discover(discover_list: List[str], clusters_info: ConfigBlock, discover_nodes: bool = True) -> DiscoveryResult:
    """Discover nodes in clusters"""

    new_nodes = []  # type: List[NodeInfo]
    os_creds = None  # type: Optional[OSCreds]

    for cluster in discover_list:
        if cluster == "openstack":
            if not discover_nodes:
                logger.warning("Skip openstack cluster discovery")
                continue

            cluster_info = clusters_info["openstack"]  # type: ConfigBlock

            conn = cluster_info['connection']  # type: ConfigBlock
            if not conn:
                logger.error("No connection provided for %s. Skipping", cluster)
                continue

            user, passwd, tenant = parse_creds(conn['creds'])

            auth_data = dict(auth_url=conn['auth_url'],
                             username=user,
                             api_key=passwd,
                             project_id=tenant)  # type: Dict[str, str]

            logger.debug("Discovering openstack nodes with connection details: %r", conn)
            new_nodes.extend(openstack.discover_openstack_nodes(auth_data, cluster_info))

        elif cluster == "fuel" or cluster == "fuel_openrc_only":
            if cluster == "fuel_openrc_only":
                discover_nodes = False

            fuel_node_info = NodeInfo(parse_ssh_uri(clusters_info['fuel']['ssh_creds']), {'fuel_master'})
            try:
                fuel_rpc_conn = setup_rpc(connect(fuel_node_info))
            except AuthenticationException:
                raise StopTestError("Wrong fuel credentials")
            except Exception:
                logger.exception("While connection to FUEL")
                raise StopTestError("Failed to connect to FUEL")

            with fuel_rpc_conn:
                nodes, fuel_info = fuel.discover_fuel_nodes(fuel_rpc_conn, clusters_info['fuel'], discover_nodes)
                new_nodes.extend(nodes)

                if fuel_info.openrc:
                    auth_url = cast(str, fuel_info.openrc['os_auth_url'])
                    if fuel_info.version >= [8, 0] and auth_url.startswith("https://"):
                            logger.warning("Fixing FUEL 8.0 AUTH url - replace https://->http://")
                            auth_url = auth_url.replace("https", "http", 1)

                    os_creds = OSCreds(name=cast(str, fuel_info.openrc['username']),
                                       passwd=cast(str, fuel_info.openrc['password']),
                                       tenant=cast(str, fuel_info.openrc['tenant_name']),
                                       auth_url=cast(str, auth_url),
                                       insecure=cast(bool, fuel_info.openrc['insecure']))

        elif cluster == "ceph":
            if discover_nodes:
                cluster_info = clusters_info["ceph"]
                root_node_uri = cast(str, cluster_info["root_node"])
                cluster = clusters_info["ceph"].get("cluster", "ceph")
                conf = clusters_info["ceph"].get("conf")
                key = clusters_info["ceph"].get("key")
                info = NodeInfo(parse_ssh_uri(root_node_uri), set())
                with setup_rpc(connect(info)) as ceph_root_conn:
                    new_nodes.extend(ceph.discover_ceph_nodes(ceph_root_conn, cluster=cluster, conf=conf, key=key))
            else:
                logger.warning("Skip ceph cluster discovery")
        else:
            msg_templ = "Unknown cluster type in 'discover' parameter: {!r}"
            raise ValueError(msg_templ.format(cluster))

    return DiscoveryResult(os_creds, new_nodes)
