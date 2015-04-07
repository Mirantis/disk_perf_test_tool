import argparse
import subprocess
import sys
import socket
import fcntl
import struct
import array


def all_interfaces():
    max_possible = 128  # arbitrary. raise if needed.
    bytes = max_possible * 32
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    names = array.array('B', '\0' * bytes)
    outbytes = struct.unpack('iL', fcntl.ioctl(
        s.fileno(),
        0x8912,  # SIOCGIFCONF
        struct.pack('iL', bytes, names.buffer_info()[0])
    ))[0]
    namestr = names.tostring()
    lst = []
    for i in range(0, outbytes, 40):
        name = namestr[i:i+16].split('\0', 1)[0]
        ip = namestr[i+20:i+24]
        lst.append((name, ip))
    return lst


def format_ip(addr):
    return str(ord(addr[0])) + '.' + \
           str(ord(addr[1])) + '.' + \
           str(ord(addr[2])) + '.' + \
           str(ord(addr[3]))


def find_interface_by_ip(ext_ip):
    ifs = all_interfaces()
    for i in ifs:
        ip = format_ip(i[1])

        if ip == ext_ip:
            return str(i[0])

    print "External ip doesnt corresponds to any of available interfaces"
    return None


def make_tunnels(ips, ext_ip, base_port=12345, delete=False):
    node_port = {}

    if delete is True:
        mode = "-D"
    else:
        mode = "-A"

    iface = find_interface_by_ip(ext_ip)

    for ip in ips:
        p = subprocess.Popen(["iptables -t nat " + mode + " PREROUTING " +
                              "-p tcp -i " + iface + "  --dport " + str(base_port) +
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
    parser = argparse.ArgumentParser(description=
                                     "Connect to fuel master "
                                     "and setup ssh agent")
    parser.add_argument(
        "--base_port", type=int, required=True)

    parser.add_argument(
        "--ext_ip", type=str, required=True)

    parser.add_argument(
        "--clean", type=bool, default=False)

    parser.add_argument(
        "--ports", type=str, nargs='+')

    return parser.parse_args(argv)


def main(argv):
    arg_object = parse_command_line(argv)
    mapping = make_tunnels(arg_object.ports,
                           ext_ip=arg_object.ext_ip,
                           base_port=arg_object.base_port,
                           delete=arg_object.clean)

    if arg_object.clean is False:
        for k in mapping:
            print k + " " + str(mapping[k])


if __name__ == "__main__":
    main(sys.argv[1:])
