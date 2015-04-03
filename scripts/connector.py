import argparse
import sys
import os
import paramiko

from urlparse import urlparse


from keystone import KeystoneAuth


def discover_fuel_nodes(fuel_url, creds, cluster_id):
    admin_ip = urlparse(fuel_url).hostname
    fuel = KeystoneAuth(fuel_url, creds, headers=None, echo=None,
                        admin_node_ip=admin_ip)
    nodes = fuel.do("get", path="/api/nodes?cluster_id=" + str(cluster_id))
    ips = [node["ip"] for node in nodes]
    roles = [node["roles"] for node in nodes]

    host = urlparse(fuel_url).hostname

    return run_agent(ips, roles, host)


def discover_fuel_nodes_clean(fuel_url, ssh_creds, nodes, base_port=12345):
    admin_ip = urlparse(fuel_url).hostname
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=admin_ip, port=ssh_creds["port"],
                password=ssh_creds["password"], username=ssh_creds["username"])

    command = "python /tmp/agent.py --clean=True --base_port=" \
              + str(base_port) + " --ports"

    for node in nodes:
        ip = urlparse(node[0]).hostname
        command += " " + ip

    (stdin, stdout, stderr) = ssh.exec_command(command)
    for line in stdout.readlines():
        print line

    os.remove('/tmp/fuel_id_rsa')


def run_agent(ip_addresses, roles, host, password="test37", port=22,
              base_port=12345):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=host, port=port, password=password, username="root")
    sftp = ssh.open_sftp()
    sftp.put(os.path.join(os.path.dirname(__file__), 'agent.py'),
             "/tmp/agent.py")
    fuel_id_rsa_path = '/tmp/fuel_id_rsa'
    sftp.get('/root/.ssh/id_rsa', fuel_id_rsa_path)
    os.chmod(fuel_id_rsa_path, 700)
    command = "python /tmp/agent.py --base_port=" + str(base_port) + " --ports"

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

    for i in range(len(ip_addresses)):
        ip = ip_addresses[i]
        role = roles[i]
        port = node_port_mapping[ip]

        nodes.append(("ssh://root@" + ip + ":" + port +
                      ":/tmp/fuel_id_rsa", role))

    ssh.close()
    print 'Files has been transfered successefully to Fuel node, ' \
          'agent has been launched'
    print nodes

    return nodes


def parse_command_line(argv):
    parser = argparse.ArgumentParser(
        description="Connect to fuel master and setup ssh agent")
    parser.add_argument(
        "--fuel_url", required=True)
    parser.add_argument(
        "--cluster_id", required=True)
    parser.add_argument(
        "--username", default="admin")
    parser.add_argument(
        "--tenantname", default="admin")
    parser.add_argument(
        "--password", default="admin")

    return parser.parse_args(argv)


def main(argv):
    args = parse_command_line(argv)
    creds = {"username": args.username,
             "tenant_name": args.tenantname,
             "password": args.password}

    nodes = discover_fuel_nodes(args.fuel_url, creds, args.cluster_id)
    discover_fuel_nodes_clean(args.fuel_url, {"username": "root",
                                              "password": "test37",
                                              "port": 22}, nodes)


if __name__ == "__main__":
    main(sys.argv[1:])
