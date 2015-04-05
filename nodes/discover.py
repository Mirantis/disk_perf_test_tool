import logging
import urlparse

import ceph
import fuel
import openstack

from disk_perf_test_tool.utils import parse_creds

logger = logging.getLogger("io-perf-tool")


def discover(discover, clusters_info):
    nodes_to_run = []
    for cluster in discover:
        if cluster == "openstack":
            cluster_info = clusters_info["openstack"]
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

        elif cluster == "fuel" or cluster == "fuel+openstack":
            cluster_info = clusters_info['fuel']
            url = cluster_info['url']
            creds = cluster_info['creds']
            ssh_creds = cluster_info['ssh_creds']
            # if user:password format us used
            if not ssh_creds.startswith("ssh://"):
                ip_port = urlparse.urlparse(url).netloc

                if ':' in ip_port:
                    ip = ip_port.split(":")[0]
                else:
                    ip = ip_port

                ssh_creds = "ssh://{0}@{1}".format(ssh_creds, ip)

            env = cluster_info['openstack_env']

            nodes_to_run.extend(fuel.discover_fuel_nodes(url, creds, env))

        elif cluster == "ceph":
            cluster_info = clusters_info["ceph"]
            nodes_to_run.extend(ceph.discover_ceph_nodes(cluster_info))
        else:
            msg_templ = "Unknown cluster type in 'discover' parameter: {0!r}"
            raise ValueError(msg_templ.format(cluster))

    return nodes_to_run
