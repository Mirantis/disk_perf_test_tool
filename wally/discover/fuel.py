import os
import re
import sys
import socket
import logging
from urlparse import urlparse

import yaml
from wally.fuel_rest_api import (KeystoneAuth, get_cluster_id,
                                 reflect_cluster, FuelInfo)
from wally.utils import parse_creds
from wally.ssh_utils import run_over_ssh, connect

from .node import Node


logger = logging.getLogger("wally.discover")
BASE_PF_PORT = 33467


def discover_fuel_nodes(fuel_data, var_dir):
    username, tenant_name, password = parse_creds(fuel_data['creds'])
    creds = {"username": username,
             "tenant_name": tenant_name,
             "password": password}

    conn = KeystoneAuth(fuel_data['url'], creds, headers=None)

    cluster_id = get_cluster_id(conn, fuel_data['openstack_env'])
    cluster = reflect_cluster(conn, cluster_id)
    version = FuelInfo(conn).get_version()

    fuel_nodes = list(cluster.get_nodes())

    logger.debug("Found FUEL {0}".format("".join(map(str, version))))

    network = 'fuelweb_admin' if version >= [6, 0] else 'admin'

    ssh_creds = fuel_data['ssh_creds']

    fuel_host = urlparse(fuel_data['url']).hostname
    fuel_ip = socket.gethostbyname(fuel_host)
    ssh_conn = connect("{0}@@{1}".format(ssh_creds, fuel_host))

    fuel_ext_iface = get_external_interface(ssh_conn, fuel_ip)

    # TODO: keep ssh key in memory
    # http://stackoverflow.com/questions/11994139/how-to-include-the-private-key-in-paramiko-after-fetching-from-string
    fuel_key_file = os.path.join(var_dir, "fuel_master_node_id_rsa")
    download_master_key(ssh_conn, fuel_key_file)

    nodes = []
    ports = range(BASE_PF_PORT, BASE_PF_PORT + len(fuel_nodes))
    ips_ports = []

    for fuel_node, port in zip(fuel_nodes, ports):
        ip = fuel_node.get_ip(network)
        forward_ssh_port(ssh_conn, fuel_ext_iface, port, ip)

        conn_url = "ssh://root@{0}:{1}:{2}".format(fuel_host,
                                                   port,
                                                   fuel_key_file)
        nodes.append(Node(conn_url, fuel_node['roles']))
        ips_ports.append((ip, port))

    logger.debug("Found %s fuel nodes for env %r" %
                 (len(nodes), fuel_data['openstack_env']))

    return ([],
            (ssh_conn, fuel_ext_iface, ips_ports),
            cluster.get_openrc())

    return (nodes,
            (ssh_conn, fuel_ext_iface, ips_ports),
            cluster.get_openrc())


def download_master_key(conn, dest):
    # download master key
    sftp = conn.open_sftp()
    sftp.get('/root/.ssh/id_rsa', dest)
    os.chmod(dest, 0o400)
    sftp.close()

    logger.debug("Fuel master key stored in {0}".format(dest))


def get_external_interface(conn, ip):
    data = run_over_ssh(conn, "ip a", node='fuel-master')
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


def forward_ssh_port(conn, iface, new_port, ip, clean=False):
    mode = "-D" if clean is True else "-A"
    cmd = "iptables -t nat {mode} PREROUTING -p tcp " + \
          "-i {iface} --dport {port} -j DNAT --to {ip}:22"
    run_over_ssh(conn,
                 cmd.format(iface=iface, port=new_port, ip=ip, mode=mode),
                 node='fuel-master')


def clean_fuel_port_forwarding(clean_data):
    conn, iface, ips_ports = clean_data
    for ip, port in ips_ports:
        forward_ssh_port(conn, iface, port, ip, clean=True)


def main(argv):
    fuel_data = yaml.load(open(sys.argv[1]).read())['clouds']['fuel']
    nodes, to_clean, openrc = discover_fuel_nodes(fuel_data, '/tmp')

    print nodes
    print openrc
    print "Ready to test"

    sys.stdin.readline()

    clean_fuel_port_forwarding(to_clean)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
