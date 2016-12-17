""" Collect data about ceph nodes"""
import json
import logging
from typing import Set, Dict, cast


from .node_interfaces import NodeInfo, IRPCNode
from .ssh_utils import ConnCreds
from .common_types import IP
from .stage import Stage, StepOrder
from .test_run_class import TestRun
from .ssh_utils import parse_ssh_uri
from .node import connect, setup_rpc


logger = logging.getLogger("wally.discover")


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


class DiscoverCephStage(Stage):
    config_block = 'ceph'
    priority = StepOrder.DISCOVER

    def run(self, ctx: TestRun) -> None:
        """Return list of ceph's nodes NodeInfo"""

        if 'ceph_nodes' in ctx.storage:
            ctx.nodes_info.extend(ctx.storage.load_list(NodeInfo, 'ceph_nodes'))
        else:
            ceph = ctx.config.ceph
            root_node_uri = cast(str, ceph.root_node)
            cluster = ceph.get("cluster", "ceph")
            conf = ceph.get("conf")
            key = ceph.get("key")
            info = NodeInfo(parse_ssh_uri(root_node_uri), set())
            ceph_nodes = {}  # type: Dict[IP, NodeInfo]

            if conf is None:
                conf = "/etc/ceph/{}.conf".format(cluster)

            if key is None:
                key = "/etc/ceph/{}.client.admin.keyring".format(cluster)

            with setup_rpc(connect(info), ctx.rpc_code, ctx.default_rpc_plugins) as node:

                # new_nodes.extend(ceph.discover_ceph_nodes(ceph_root_conn, cluster=cluster, conf=conf, key=key))
                ssh_key = node.get_file_content("~/.ssh/id_rsa")

                try:
                    for ip in get_osds_ips(node, conf, key):
                        if ip in ceph_nodes:
                            ceph_nodes[ip].roles.add("ceph-osd")
                        else:
                            ceph_nodes[ip] = NodeInfo(ConnCreds(cast(str, ip), user="root", key=ssh_key), {"ceph-osd"})
                except Exception as exc:
                    logger.error("OSD discovery failed: %s", exc)

                try:
                    for ip in get_mons_ips(node, conf, key):
                        if ip in ceph_nodes:
                            ceph_nodes[ip].roles.add("ceph-mon")
                        else:
                            ceph_nodes[ip] = NodeInfo(ConnCreds(cast(str, ip), user="root", key=ssh_key), {"ceph-mon"})
                except Exception as exc:
                    logger.error("MON discovery failed: %s", exc)

            ctx.nodes_info.extend(ceph_nodes.values())
            ctx.storage['ceph-nodes'] = list(ceph_nodes.values())
