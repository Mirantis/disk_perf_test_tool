""" Collect data about ceph nodes"""
import json

from node import Node
from disk_perf_test_tool.ext_libs import sh


def discover_ceph_node():
    """ Return list of ceph's nodes ips """
    ips = {}

    osd_ips = get_osds_ips(get_osds_list())
    mon_ips = get_mons_or_mds_ips("mon")
    mds_ips = get_mons_or_mds_ips("mds")

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


def get_osds_list():
    """ Get list of osds id"""
    return filter(None, sh.ceph.osd.ls().split("\n"))


def get_mons_or_mds_ips(who):
    """ Return mon ip list
        :param who - "mon" or "mds" """
    if who == "mon":
        res = sh.ceph.mon.dump()
    elif who == "mds":
        res = sh.ceph.mds.dump()
    else:
        raise ValueError(("'%s' in get_mons_or_mds_ips instead" +
                          "of mon/mds") % who)

    line_res = res.split("\n")
    ips = set()

    for line in line_res:
        fields = line.split()

        # what does fields[1], fields[2] means?
        # make this code looks like:
        # SOME_MENINGFULL_VAR1, SOME_MENINGFULL_VAR2 = line.split()[1:3]

        if len(fields) > 2 and who in fields[2]:
            ips.add(fields[1].split(":")[0])

    return ips


def get_osds_ips(osd_list):
    """ Get osd's ips
        :param osd_list - list of osd names from osd ls command"""
    ips = set()
    for osd_id in osd_list:
        res = sh.ceph.osd.find(osd_id)
        ip = json.loads(str(res))["ip"]
        ips.add(ip.split(":")[0])
    return ips
