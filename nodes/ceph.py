""" Collect data about ceph nodes"""
import json
import logging


from node import Node
from disk_perf_test_tool.ssh_utils import connect


logger = logging.getLogger("io-perf-tool")


def discover_ceph_node(ip):
    """ Return list of ceph's nodes ips """
    ips = {}
    ssh = connect(ip)

    osd_ips = get_osds_ips(ssh, get_osds_list(ssh))
    mon_ips = get_mons_or_mds_ips(ssh, "mon")
    mds_ips = get_mons_or_mds_ips(ssh, "mds")

    for ip in osd_ips:
        url = "ssh://%s" % ip
        ips.setdefault(url, []).append("ceph-osd")

    for ip in mon_ips:
        url = "ssh://%s" % ip
        ips.setdefault(url, []).append("ceph-mon")

    for ip in mds_ips:
        url = "ssh://%s" % ip
        ips.setdefault(url, []).append("ceph-mds")

    return [Node(ip=url, roles=list(roles)) for url, roles in ips.items()]


def get_osds_list(ssh):
    """ Get list of osds id"""
    _, chan, _ = ssh.exec_command("ceph osd ls")
    return filter(None, chan.read().split("\n"))


def get_mons_or_mds_ips(ssh, who):
    """ Return mon ip list
        :param who - "mon" or "mds" """
    if who == "mon":
        _, chan, _ = ssh.exec_command("ceph mon dump")
    elif who == "mds":
        _, chan, _ = ssh.exec_command("ceph mds dump")
    else:
        raise ValueError(("'%s' in get_mons_or_mds_ips instead" +
                          "of mon/mds") % who)

    line_res = chan.read().split("\n")
    ips = set()

    for line in line_res:
        fields = line.split()

        # what does fields[1], fields[2] means?
        # make this code looks like:
        # SOME_MENINGFULL_VAR1, SOME_MENINGFULL_VAR2 = line.split()[1:3]

        if len(fields) > 2 and who in fields[2]:
            ips.add(fields[1].split(":")[0])

    return ips


def get_osds_ips(ssh, osd_list):
    """ Get osd's ips
        :param osd_list - list of osd names from osd ls command"""
    ips = set()
    for osd_id in osd_list:
        _, chan, _ = ssh.exec_command("ceph osd find {0}".format(osd_id))
        ip = json.loads(chan.read())["ip"]
        ips.add(ip.split(":")[0])
    return ips
