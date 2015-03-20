import logging

import openstack
import ceph
import fuel


logger = logging.getLogger("io-perf-tool")


def discover(cluster_conf):
    if not cluster_conf:
        logger.error("No nodes configured")

    nodes_to_run = []
    for cluster, cluster_info in cluster_conf.items():
        if cluster == "openstack":
            conn = cluster_info.get('connection')
            if not conn:
                logger.error("No connection provided for %s. Skipping"
                             % cluster)
                continue
            logger.debug("Discovering openstack nodes "
                         "with connection details: %r" %
                         conn)

            nodes_to_run.extend(openstack.discover_openstack_nodes(
                conn, cluster_info))
        if cluster == "fuel":
            url = cluster_info['connection'].pop('url')
            creads = cluster_info['connection']
            roles = cluster_info['discover']
            if isinstance(roles, basestring):
                roles = [roles]
            nodes_to_run.extend(fuel.discover_fuel_nodes(url, creads, roles))

        if cluster == "ceph":
            nodes_to_run.extend(ceph.discover_ceph_node())
    return nodes_to_run
