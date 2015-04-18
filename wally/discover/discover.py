import logging

from . import ceph
from . import fuel
from . import openstack
from wally.utils import parse_creds


logger = logging.getLogger("wally.discover")


def discover(ctx, discover, clusters_info, var_dir):
    nodes_to_run = []
    clean_data = None
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

            res = fuel.discover_fuel_nodes(clusters_info['fuel'], var_dir)
            nodes, clean_data, openrc_dict = res

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

    return nodes_to_run, clean_data


def undiscover(clean_data):
    if clean_data is not None:
        fuel.clean_fuel_port_forwarding(clean_data)
