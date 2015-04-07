import argparse
import sys
import tempfile
import os
import paramiko

from urlparse import urlparse
from nodes.node import Node
from utils import parse_creds
from keystone import KeystoneAuth


from keystone import KeystoneAuth

tmp_file = tempfile.NamedTemporaryFile().name


def discover_fuel_nodes(fuel_url, creds, cluster_id):
    username, tenant_name, password = parse_creds(creds)
    creds = {"username": username,
             "tenant_name": tenant_name,
             "password": password}

    admin_ip = urlparse(fuel_url).hostname
    fuel = KeystoneAuth(fuel_url, creds, headers=None, echo=None,
                        admin_node_ip=admin_ip)
    nodes = fuel.do("get", path="/api/nodes?cluster_id=" + str(cluster_id))
    ips = [node["ip"] for node in nodes]
    roles = [node["roles"] for node in nodes]

    host = urlparse(fuel_url).hostname

    nodes, to_clean = run_agent(ips, roles, host, tmp_file)
    nodes = [Node(node[0], node[1]) for node in nodes]

    return nodes, to_clean


def discover_fuel_nodes_clean(fuel_url, ssh_creds, nodes, base_port=12345):
    admin_ip = urlparse(fuel_url).hostname
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=admin_ip, port=ssh_creds["port"],
                password=ssh_creds["password"], username=ssh_creds["username"])

    command = "python /tmp/agent.py --clean=True --ext_ip=" + admin_ip + " --base_port=" \
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
    command = "python /tmp/agent.py --base_port=" + str(base_port) + " --ext_ip=" \
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

        nodes_to_clean.append(("ssh://root@" + ip + ":" + port.rstrip('\n')
                                + ":" + fuel_id_rsa_path, role))

        nodes.append(("ssh://root@" + host + ":" + port.rstrip('\n')
                      + ":" + fuel_id_rsa_path, role))

    ssh.close()
    print 'Files has been transfered successefully to Fuel node, ' \
          'agent has been launched'
    print nodes

    return nodes, nodes_to_clean


def parse_command_line(argv):
    parser = argparse.ArgumentParser(
        description="Connect to fuel master and setup ssh agent")
    parser.add_argument(
        "--fuel_url", required=True)
    parser.add_argument(
        "--cluster_id", required=True)
    parser.add_argument(
        "--iface", default="eth1")
    parser.add_argument(
        "--creds", default="admin:admin:admin")


    return parser.parse_args(argv)


def main(argv):
    args = parse_command_line(argv)

    nodes, to_clean = discover_fuel_nodes(args.fuel_url, args.creds, args.cluster_id)
    discover_fuel_nodes_clean(args.fuel_url, {"username": "root",
                                              "password": "test37",
                                              "port": 22}, to_clean)


if __name__ == "__main__":
    main(sys.argv[1:])
