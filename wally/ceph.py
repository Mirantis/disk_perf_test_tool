""" Collect data about ceph nodes"""
import json
import logging
from typing import Dict, cast, List, Set, Optional


from .node_interfaces import NodeInfo, IRPCNode
from .ssh_utils import ConnCreds
from .common_types import IP
from .stage import Stage, StepOrder
from .test_run_class import TestRun
from .ssh_utils import parse_ssh_uri
from .node import connect, setup_rpc
from .utils import StopTestError


logger = logging.getLogger("wally")


class OSDInfo:
    def __init__(self, id: int, journal: str = None, storage: str = None) -> None:
        self.id = id
        self.journal = journal
        self.storage = storage


def get_osds_info(node: IRPCNode, conf: str, key: str) -> Dict[IP, List[OSDInfo]]:
    """Get set of osd's ip"""

    data = node.run("ceph -c {} -k {} --format json osd dump".format(conf, key))
    jdata = json.loads(data)
    ips = {}  # type: Dict[IP, List[OSDInfo]]
    first_error = True
    for osd_data in jdata["osds"]:
        osd_id = int(osd_data["osd"])
        if "public_addr" not in osd_data:
            if first_error:
                logger.warning("No 'public_addr' field in 'ceph osd dump' output for osd %s" +
                               "(all subsequent errors omitted)", osd_id)
                first_error = False
        else:
            ip_port = osd_data["public_addr"]
            if '/' in ip_port:
                ip_port = ip_port.split("/", 1)[0]
            ip = IP(ip_port.split(":")[0])

            osd_journal_path = None  # type: Optional[str]
            osd_data_path = None  # type: Optional[str]

            # TODO: parallelize this!
            osd_cfg = node.run("ceph -n osd.{} --show-config".format(osd_id))
            for line in osd_cfg.split("\n"):
                if line.startswith("osd_journal ="):
                    osd_journal_path = line.split("=")[1].strip()
                elif line.startswith("osd_data ="):
                    osd_data_path = line.split("=")[1].strip()

            if osd_data_path is None or osd_journal_path is None:
                logger.error("Can't detect osd %s journal or storage path", osd_id)
                raise StopTestError()

            ips.setdefault(ip, []).append(OSDInfo(osd_id,
                                                  journal=osd_journal_path,
                                                  storage=osd_data_path))
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

        discovery = ctx.config.get("discovery")
        if discovery == 'disable' or discovery == 'metadata':
            logger.info("Skip ceph discovery due to config setting")
            return

        if 'all_nodes' in ctx.storage:
            logger.debug("Skip ceph discovery, use previously discovered nodes")
            return

        ceph = ctx.config.ceph
        root_node_uri = cast(str, ceph.root_node)
        cluster = ceph.get("cluster", "ceph")
        conf = ceph.get("conf")
        key = ceph.get("key")

        logger.debug("Start discovering ceph nodes from root %s", root_node_uri)
        logger.debug("cluster=%s key=%s conf=%s", cluster, conf, key)

        info = NodeInfo(parse_ssh_uri(root_node_uri), set())

        if conf is None:
            conf = "/etc/ceph/{}.conf".format(cluster)

        if key is None:
            key = "/etc/ceph/{}.client.admin.keyring".format(cluster)

        ceph_params = {"cluster": cluster, "conf": conf, "key": key}

        with setup_rpc(connect(info),
                       ctx.rpc_code,
                       ctx.default_rpc_plugins,
                       log_level=ctx.config.rpc_log_level) as node:

            ssh_key = node.get_file_content("~/.ssh/id_rsa")


            try:
                ips = set()
                for ip, osds_info in get_osds_info(node, conf, key).items():
                    ips.add(ip)
                    creds = ConnCreds(cast(str, ip), user="root", key=ssh_key)
                    info = ctx.merge_node(creds, {'ceph-osd'})
                    info.params.setdefault('ceph-osds', []).extend(osds_info)
                    assert 'ceph' not in info.params or info.params['ceph'] == ceph_params
                    info.params['ceph'] = ceph_params

                logger.debug("Found %s nodes with ceph-osd role", len(ips))
            except Exception as exc:
                if discovery != 'ignore_errors':
                    logger.exception("OSD discovery failed")
                    raise StopTestError()
                else:
                    logger.warning("OSD discovery failed %s", exc)

            try:
                counter = 0
                for counter, ip in enumerate(get_mons_ips(node, conf, key)):
                    creds = ConnCreds(cast(str, ip), user="root", key=ssh_key)
                    info = ctx.merge_node(creds, {'ceph-mon'})
                    assert 'ceph' not in info.params or info.params['ceph'] == ceph_params
                    info.params['ceph'] = ceph_params
                logger.debug("Found %s nodes with ceph-mon role", counter + 1)
            except Exception as exc:
                if discovery != 'ignore_errors':
                    logger.exception("MON discovery failed")
                    raise StopTestError()
                else:
                    logger.warning("MON discovery failed %s", exc)
