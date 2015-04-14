import logging
import urlparse

import ceph
import openstack
from utils import parse_creds
from scripts import connector

logger = logging.getLogger("io-perf-tool")


def discover(ctx, discover, clusters_info):
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

        elif cluster == "fuel":
            cluster_info = clusters_info['fuel']
            cluster_name = cluster_info['openstack_env']
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

            dfunc = connector.discover_fuel_nodes
            nodes, clean_data, openrc_dict = dfunc(url, creds, cluster_name)

            ctx.fuel_openstack_creds = {'name': openrc_dict['username'],
                                        'passwd': openrc_dict['password'],
                                        'tenant': openrc_dict['tenant_name'],
                                        'auth_url': openrc_dict['os_auth_url']}

            nodes_to_run.extend(nodes)

        elif cluster == "ceph":
            cluster_info = clusters_info["ceph"]
            nodes_to_run.extend(ceph.discover_ceph_nodes(cluster_info))
        else:
            msg_templ = "Unknown cluster type in 'discover' parameter: {0!r}"
            raise ValueError(msg_templ.format(cluster))

    return nodes_to_run
