""" Collect data about ceph nodes"""
import json
import logging
from typing import List, Set, Dict


from ..node_interfaces import NodeInfo, IRPCNode
from ..ssh_utils import ConnCreds
from ..common_types import IP

logger = logging.getLogger("wally.discover")


def discover_ceph_nodes(node: IRPCNode,
                        cluster: str = "ceph",
                        conf: str = None,
                        key: str = None) -> List[NodeInfo]:
    """Return list of ceph's nodes NodeInfo"""

    if conf is None:
        conf = "/etc/ceph/{}.conf".format(cluster)

    if key is None:
        key = "/etc/ceph/{}.client.admin.keyring".format(cluster)

    try:
        osd_ips = get_osds_ips(node, conf, key)
    except Exception as exc:
        logger.error("OSD discovery failed: %s", exc)
        osd_ips = set()

    try:
        mon_ips = get_mons_ips(node, conf, key)
    except Exception as exc:
        logger.error("MON discovery failed: %s", exc)
        mon_ips = set()

    ips = {}  # type: Dict[str, List[str]]
    for ip in osd_ips:
        ips.setdefault(ip, []).append("ceph-osd")

    for ip in mon_ips:
        ips.setdefault(ip, []).append("ceph-mon")

    ssh_key = node.get_file_content("~/.ssh/id_rsa")
    return [NodeInfo(ConnCreds(host=ip, user="root", key=ssh_key), set(roles)) for ip, roles in ips.items()]


def get_osds_ips(node: IRPCNode, conf: str, key: str) -> Set[IP]:
    """Get set of osd's ip"""

    data = node.run("ceph -c {} -k {} --format json osd dump".format(conf, key))
    jdata = json.loads(data)
    ips = set()  # type: Set[IP]
    first_error = True
    for osd_data in jdata["osds"]:
        if "public_addr" not in osd_data:
            if first_error:
                osd_id = osd_data.get("osd", "<OSD_ID_MISSED>")
                logger.warning("No 'public_addr' field in 'ceph osd dump' output for osd %s" +
                               "(all subsequent errors omitted)", osd_id)
                first_error = False
        else:
            ip_port = osd_data["public_addr"]
            if '/' in ip_port:
                ip_port = ip_port.split("/", 1)[0]
            ips.add(IP(ip_port.split(":")[0]))
    return ips


def get_mons_ips(node: IRPCNode, conf: str, key: str) -> Set[IP]:
    """Return mon ip set"""

    data = node.run("ceph -c {} -k {} --format json mon_status".format(conf, key))
    jdata = json.loads(data)
    ips = set()  # type: Set[IP]
    first_error = True
    for mon_data in jdata["monmap"]["mons"]:
        if "addr" not in mon_data:
            if first_error:
                mon_name = mon_data.get("name", "<MON_NAME_MISSED>")
                logger.warning("No 'addr' field in 'ceph mon_status' output for mon %s" +
                               "(all subsequent errors omitted)", mon_name)
                first_error = False
        else:
            ip_port = mon_data["addr"]
            if '/' in ip_port:
                ip_port = ip_port.split("/", 1)[0]
            ips.add(IP(ip_port.split(":")[0]))

    return ips
