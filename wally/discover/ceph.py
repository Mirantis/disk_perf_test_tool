""" Collect data about ceph nodes"""
import json
import logging
from typing import Iterable


from ..node import NodeInfo, Node


logger = logging.getLogger("wally.discover")


def discover_ceph_nodes(node: Node) -> Iterable[NodeInfo]:
    """Return list of ceph's nodes NodeInfo"""
    ips = {}

    osd_ips = get_osds_ips(node, get_osds_list(node))
    mon_ips = get_mons_or_mds_ips(node, "mon")
    mds_ips = get_mons_or_mds_ips(node, "mds")

    for ip in osd_ips:
        url = "ssh://%s" % ip
        ips.setdefault(url, []).append("ceph-osd")

    for ip in mon_ips:
        url = "ssh://%s" % ip
        ips.setdefault(url, []).append("ceph-mon")

    for ip in mds_ips:
        url = "ssh://%s" % ip
        ips.setdefault(url, []).append("ceph-mds")

    return [NodeInfo(url, set(roles)) for url, roles in ips.items()]


def get_osds_list(node: Node) -> Iterable[str]:
    """Get list of osd's id"""
    return filter(None, node.run("ceph osd ls").split("\n"))


def get_mons_or_mds_ips(node: Node, who: str) -> Iterable[str]:
    """Return mon ip list. who - mon/mds"""
    assert who in ("mon", "mds"), \
        "{!r} in get_mons_or_mds_ips instead of mon/mds".format(who)

    line_res = node.run("ceph {0} dump".format(who)).split("\n")

    ips = set()
    for line in line_res:
        fields = line.split()
        if len(fields) > 2 and who in fields[2]:
            ips.add(fields[1].split(":")[0])

    return ips


def get_osds_ips(node: Node, osd_list: Iterable[str]) -> Iterable[str]:
    """Get osd's ips. osd_list - list of osd names from osd ls command"""
    ips = set()
    for osd_id in osd_list:
        out = node.run("ceph osd find {0}".format(osd_id))
        ip = json.loads(out)["ip"]
        ips.add(str(ip.split(":")[0]))
    return ips
