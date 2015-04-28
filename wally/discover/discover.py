import os.path
import logging

from . import ceph
from . import fuel
from . import openstack
from wally.utils import parse_creds


logger = logging.getLogger("wally.discover")


openrc_templ = """#!/bin/sh
export LC_ALL=C
export OS_NO_CACHE='true'
export OS_TENANT_NAME='{tenant}'
export OS_USERNAME='{name}'
export OS_PASSWORD='{passwd}'
export OS_AUTH_URL='{auth_url}'
export OS_AUTH_STRATEGY='keystone'
export OS_REGION_NAME='RegionOne'
export CINDER_ENDPOINT_TYPE='publicURL'
export GLANCE_ENDPOINT_TYPE='publicURL'
export KEYSTONE_ENDPOINT_TYPE='publicURL'
export NOVA_ENDPOINT_TYPE='publicURL'
export NEUTRON_ENDPOINT_TYPE='publicURL'
"""


def discover(ctx, discover, clusters_info, var_dir, discover_nodes=True):
    nodes_to_run = []
    clean_data = None
    for cluster in discover:
        if cluster == "openstack" and not discover_nodes:
            logger.warning("Skip openstack cluster discovery")
        elif cluster == "openstack" and discover_nodes:
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

        elif cluster == "fuel" or cluster == "fuel_openrc_only":
            if cluster == "fuel_openrc_only":
                discover_nodes = False

            res = fuel.discover_fuel_nodes(clusters_info['fuel'],
                                           var_dir,
                                           discover_nodes)
            nodes, clean_data, openrc_dict = res

            ctx.fuel_openstack_creds = {'name': openrc_dict['username'],
                                        'passwd': openrc_dict['password'],
                                        'tenant': openrc_dict['tenant_name'],
                                        'auth_url': openrc_dict['os_auth_url']}

            env_name = clusters_info['fuel']['openstack_env']
            env_f_name = env_name
            for char in "-+ {}()[]":
                env_f_name = env_f_name.replace(char, '_')

            fuel_openrc_fname = os.path.join(var_dir,
                                             env_f_name + "_openrc")

            with open(fuel_openrc_fname, "w") as fd:
                fd.write(openrc_templ.format(**ctx.fuel_openstack_creds))

            msg = "Openrc for cluster {0} saves into {1}"
            logger.debug(msg.format(env_name, fuel_openrc_fname))
            nodes_to_run.extend(nodes)

        elif cluster == "ceph":
            if discover_nodes:
                cluster_info = clusters_info["ceph"]
                nodes_to_run.extend(ceph.discover_ceph_nodes(cluster_info))
            else:
                logger.warning("Skip ceph cluster discovery")
        else:
            msg_templ = "Unknown cluster type in 'discover' parameter: {0!r}"
            raise ValueError(msg_templ.format(cluster))

    return nodes_to_run, clean_data


def undiscover(clean_data):
    if clean_data is not None:
        fuel.clean_fuel_port_forwarding(clean_data)
