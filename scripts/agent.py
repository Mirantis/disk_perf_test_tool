import argparse
import subprocess
import sys


def make_tunnels(ips, base_port=12345, delete=False):
    node_port = {}

    if delete is True:
        mode = "-D"
    else:
        mode = "-A"

    for ip in ips:
        p = subprocess.Popen(["iptables -t nat " + mode + " PREROUTING " +
                              "-p tcp -i eth1 --dport " + str(base_port) +
                              " -j DNAT --to " + str(ip) + ":22"],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             shell=True)

        out, err = p.communicate()

        if out is not None:
            print out

        if err is not None:
            print err

        node_port[ip] = base_port
        base_port += 1

    return node_port


def parse_command_line(argv):
    parser = argparse.ArgumentParser(description="Connect to fuel master " +
                                     "and setup ssh agent")
    parser.add_argument("--base_port", type=int, required=True)
    # To do: fix clean to be False when string is False
    parser.add_argument("--clean", type=bool, default=False)
    parser.add_argument("--ports", type=str, nargs='+')

    return parser.parse_args(argv)


def main(argv):
    arg_object = parse_command_line(argv)
    mapping = make_tunnels(arg_object.ports,
                           base_port=arg_object.base_port,
                           delete=arg_object.clean)

    if arg_object.clean is False:
        for k in mapping:
            print k + " " + str(mapping[k])


if __name__ == "__main__":
    main(sys.argv[1:])
