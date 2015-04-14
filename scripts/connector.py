import os
import sys
import logging
import argparse
import tempfile
import paramiko

import fuel_rest_api
from nodes.node import Node
from utils import parse_creds
from urlparse import urlparse


tmp_file = tempfile.NamedTemporaryFile().name
openrc_path = tempfile.NamedTemporaryFile().name
logger = logging.getLogger("io-perf-tool")


def discover_fuel_nodes(fuel_url, creds, cluster_name):
    username, tenant_name, password = parse_creds(creds)
    creds = {"username": username,
             "tenant_name": tenant_name,
             "password": password}

    conn = fuel_rest_api.KeystoneAuth(fuel_url, creds, headers=None)
    cluster_id = fuel_rest_api.get_cluster_id(conn, cluster_name)
    cluster = fuel_rest_api.reflect_cluster(conn, cluster_id)

    nodes = list(cluster.get_nodes())
    ips = [node.get_ip('admin') for node in nodes]
    roles = [node["roles"] for node in nodes]

    host = urlparse(fuel_url).hostname

    nodes, to_clean = run_agent(ips, roles, host, tmp_file)
    nodes = [Node(node[0], node[1]) for node in nodes]

    openrc_dict = cluster.get_openrc()

    logger.debug("Found %s fuel nodes for env %r" % (len(nodes), cluster_name))
    return nodes, to_clean, openrc_dict


def discover_fuel_nodes_clean(fuel_url, ssh_creds, nodes, base_port=12345):
    admin_ip = urlparse(fuel_url).hostname
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=admin_ip, port=ssh_creds["port"],
                password=ssh_creds["password"], username=ssh_creds["username"])

    command = "python /tmp/agent.py --clean=True --ext_ip=" + \
              admin_ip + " --base_port=" \
              + str(base_port) + " --ports"

    for node in nodes:
        ip = urlparse(node[0]).hostname
        command += " " + ip

    (stdin, stdout, stderr) = ssh.exec_command(command)
    for line in stdout.readlines():
        print line


def run_agent(ip_addresses, roles, host, tmp_name, password="test37", port=22,
              base_port=12345):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=host, port=port, password=password, username="root")
    sftp = ssh.open_sftp()
    sftp.put(os.path.join(os.path.dirname(__file__), 'agent.py'),
             "/tmp/agent.py")
    fuel_id_rsa_path = tmp_name
    sftp.get('/root/.ssh/id_rsa', fuel_id_rsa_path)
    os.chmod(fuel_id_rsa_path, 0o700)
    command = "python /tmp/agent.py --base_port=" + \
              str(base_port) + " --ext_ip=" \
              + host + " --ports"

    for address in ip_addresses:
        command += " " + address

    (stdin, stdout, stderr) = ssh.exec_command(command)
    node_port_mapping = {}

    for line in stdout.readlines():
        results = line.split(' ')

        if len(results) != 2:
            continue

        node, port = results
        node_port_mapping[node] = port

    nodes = []
    nodes_to_clean = []

    for i in range(len(ip_addresses)):
        ip = ip_addresses[i]
        role = roles[i]
        port = node_port_mapping[ip]

        nodes_to_clean.append(("ssh://root@" + ip + ":" +
                               port.rstrip('\n')
                               + ":" + fuel_id_rsa_path, role))

        nodes.append(("ssh://root@" + host + ":" + port.rstrip('\n')
                      + ":" + fuel_id_rsa_path, role))

    ssh.close()
    logger.info('Files has been transferred successfully to Fuel node, ' +
                'agent has been launched')

    return nodes, nodes_to_clean


def parse_command_line(argv):
    parser = argparse.ArgumentParser(
        description="Connect to fuel master and setup ssh agent")
    parser.add_argument(
        "--fuel_url", required=True)
    parser.add_argument(
        "--cluster_name", required=True)
    parser.add_argument(
        "--iface", default="eth1")
    parser.add_argument(
        "--creds", default="admin:admin@admin")

    return parser.parse_args(argv)


def main(argv):
    args = parse_command_line(argv)

    nodes, to_clean, _ = discover_fuel_nodes(args.fuel_url,
                                             args.creds,
                                             args.cluster_name)
    discover_fuel_nodes_clean(args.fuel_url, {"username": "root",
                                              "password": "test37",
                                              "port": 22}, to_clean)


if __name__ == "__main__":
    main(sys.argv[1:])
