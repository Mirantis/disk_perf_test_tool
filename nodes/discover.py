import logging

import ceph
import fuel
import openstack

from disk_perf_test_tool.utils import parse_creds

logger = logging.getLogger("io-perf-tool")


def discover(cluster_conf):
    if not cluster_conf:
        logger.error("No nodes configured")

    nodes_to_run = []
    for cluster, cluster_info in cluster_conf.items():
        if cluster == "openstack":

            conn = cluster_info['connection']
            user, passwd, tenant = parse_creds(conn['creds'])

            auth_data = dict(
                auth_url=conn['auth_url'],
                username=user,
                api_key=passwd,
                project_id=tenant)

            if not conn:
                logger.error("No connection provided for %s. Skipping"
                             % cluster)
                continue

            logger.debug("Discovering openstack nodes "
                         "with connection details: %r" %
                         conn)

            os_nodes = openstack.discover_openstack_nodes(auth_data,
                                                          cluster_info)
            nodes_to_run.extend(os_nodes)

        if cluster == "fuel":
            url = cluster_info['connection'].pop('url')
            creads = cluster_info['connection']
            roles = cluster_info['discover']

            if isinstance(roles, basestring):
                roles = [roles]

            nodes_to_run.extend(fuel.discover_fuel_nodes(url, creads, roles))

        if cluster == "ceph":
            nodes_to_run.extend(ceph.discover_ceph_nodes(cluster_info))

    return nodes_to_run
