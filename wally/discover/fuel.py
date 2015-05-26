import re
import socket
import logging
from urlparse import urlparse

import sshtunnel
from paramiko import AuthenticationException


from wally.fuel_rest_api import (KeystoneAuth, get_cluster_id,
                                 reflect_cluster, FuelInfo)
from wally.utils import (parse_creds, check_input_param, StopTestError,
                         clean_resource, get_ip_for_target)
from wally.ssh_utils import (run_over_ssh, connect, set_key_for_node,
                             read_from_remote)

from .node import Node


logger = logging.getLogger("wally.discover")
BASE_PF_PORT = 44006


def discover_fuel_nodes(fuel_data, var_dir, discover_nodes=True):
    username, tenant_name, password = parse_creds(fuel_data['creds'])
    creds = {"username": username,
             "tenant_name": tenant_name,
             "password": password}

    conn = KeystoneAuth(fuel_data['url'], creds, headers=None)

    msg = "openstack_env should be provided in fuel config"
    check_input_param('openstack_env' in fuel_data, msg)

    cluster_id = get_cluster_id(conn, fuel_data['openstack_env'])
    cluster = reflect_cluster(conn, cluster_id)

    if not discover_nodes:
        logger.warning("Skip fuel cluster discovery")
        return ([], None, cluster.get_openrc())

    version = FuelInfo(conn).get_version()

    fuel_nodes = list(cluster.get_nodes())

    logger.info("Found FUEL {0}".format(".".join(map(str, version))))

    network = 'fuelweb_admin' if version >= [6, 0] else 'admin'

    ssh_creds = fuel_data['ssh_creds']

    fuel_host = urlparse(fuel_data['url']).hostname
    fuel_ip = socket.gethostbyname(fuel_host)

    try:
        ssh_conn = connect("{0}@{1}".format(ssh_creds, fuel_host))
    except AuthenticationException:
        raise StopTestError("Wrong fuel credentials")
    except Exception:
        logger.exception("While connection to FUEL")
        raise StopTestError("Failed to connect to FUEL")

    fuel_ext_iface = get_external_interface(ssh_conn, fuel_ip)

    logger.debug("Downloading fuel master key")
    fuel_key = download_master_key(ssh_conn)

    nodes = []
    ips_ports = []

    logger.info("Forwarding ssh ports from FUEL nodes to localhost")
    fuel_usr, fuel_passwd = ssh_creds.split(":", 1)
    ips = [str(fuel_node.get_ip(network)) for fuel_node in fuel_nodes]
    port_fw = forward_ssh_ports(fuel_host, fuel_usr, fuel_passwd, ips)
    listen_ip = get_ip_for_target(fuel_host)

    for port, fuel_node, ip in zip(port_fw, fuel_nodes, ips):
        logger.debug(
            "SSH port forwarding {0} => localhost:{1}".format(ip, port))

        conn_url = "ssh://root@127.0.0.1:{0}".format(port)
        set_key_for_node(('127.0.0.1', port), fuel_key)

        node = Node(conn_url, fuel_node['roles'])
        node.monitor_ip = listen_ip
        nodes.append(node)
        ips_ports.append((ip, port))

    logger.debug("Found %s fuel nodes for env %r" %
                 (len(nodes), fuel_data['openstack_env']))

    if version > [6, 0]:
        openrc = cluster.get_openrc()
    else:
        logger.warning("Getting openrc on fuel 6.0 is broken, skip")
        openrc = None

    return (nodes,
            (ssh_conn, fuel_ext_iface, ips_ports),
            openrc)


def download_master_key(conn):
    # download master key
    with conn.open_sftp() as sftp:
        return read_from_remote(sftp, '/root/.ssh/id_rsa')


def get_external_interface(conn, ip):
    data = run_over_ssh(conn, "ip a", node='fuel-master', nolog=True)
    curr_iface = None
    for line in data.split("\n"):

        match1 = re.match(r"\d+:\s+(?P<name>.*?):\s\<", line)
        if match1 is not None:
            curr_iface = match1.group('name')

        match2 = re.match(r"\s+inet\s+(?P<ip>[0-9.]+)/", line)
        if match2 is not None:
            if match2.group('ip') == ip:
                assert curr_iface is not None
                return curr_iface
    raise KeyError("Can't found interface for ip {0}".format(ip))


def forward_ssh_ports(proxy_ip, proxy_user, proxy_passwd, ips):
    for ip in ips:
        tunnel = sshtunnel.open(
                    (proxy_ip, 22),
                    ssh_username=proxy_user,
                    ssh_password=proxy_passwd,
                    threaded=True,
                    remote_bind_address=(ip, 22))
        tunnel.__enter__()
        clean_resource(tunnel.__exit__, None, None, None)
        yield tunnel.local_bind_port
