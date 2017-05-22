""" Collect data about ceph nodes"""
import logging
from typing import Dict, cast, List, Set

from cephlib import discover
from cephlib.discover import OSDInfo
from cephlib.common import to_ip
from cephlib.node import NodeInfo, IRPCNode
from cephlib.ssh import ConnCreds, IP, parse_ssh_uri
from cephlib.node_impl import connect, setup_rpc

from .stage import Stage, StepOrder
from .test_run_class import TestRun
from .utils import StopTestError


logger = logging.getLogger("wally")


def get_osds_info(node: IRPCNode, ceph_extra_args: str = "", thcount: int = 8) -> Dict[IP, List[OSDInfo]]:
    """Get set of osd's ip"""
    res = {}  # type: Dict[IP, List[OSDInfo]]
    return {IP(ip): osd_info_list
            for ip, osd_info_list in discover.get_osds_nodes(node.run, ceph_extra_args, thcount=thcount).items()}


def get_mons_ips(node: IRPCNode, ceph_extra_args: str = "") -> Set[IP]:
    """Return mon ip set"""
    return {IP(ip) for ip, _ in discover.get_mons_nodes(node.run, ceph_extra_args).values()}


class DiscoverCephStage(Stage):
    config_block = 'ceph'
    priority = StepOrder.DISCOVER

    def run(self, ctx: TestRun) -> None:
        """Return list of ceph's nodes NodeInfo"""
        if 'all_nodes' in ctx.storage:
            logger.debug("Skip ceph discovery, use previously discovered nodes")
            return

        if 'metadata' in ctx.config.discover:
            logger.exception("Ceph metadata discovery is not implemented")
            raise StopTestError()

        ignore_errors = 'ignore_errors' in ctx.config.discover
        ceph = ctx.config.ceph
        root_node_uri = cast(str, ceph.root_node)
        cluster = ceph.get("cluster", "ceph")
        ip_remap = ctx.config.ceph.get('ip_remap', {})

        conf = ceph.get("conf")
        key = ceph.get("key")

        if conf is None:
            conf = "/etc/ceph/{}.conf".format(cluster)

        if key is None:
            key = "/etc/ceph/{}.client.admin.keyring".format(cluster)

        ceph_extra_args = ""

        if conf:
            ceph_extra_args += " -c '{}'".format(conf)

        if key:
            ceph_extra_args += " -k '{}'".format(key)

        logger.debug("Start discovering ceph nodes from root %s", root_node_uri)
        logger.debug("cluster=%s key=%s conf=%s", cluster, conf, key)

        info = NodeInfo(parse_ssh_uri(root_node_uri), set())

        ceph_params = {"cluster": cluster, "conf": conf, "key": key}

        with setup_rpc(connect(info), ctx.rpc_code, ctx.default_rpc_plugins,
                       log_level=ctx.config.rpc_log_level) as node:

            try:
                ips = set()
                for ip, osds_info in get_osds_info(node, ceph_extra_args, thcount=16).items():
                    ip = ip_remap.get(ip, ip)
                    ips.add(ip)
                    creds = ConnCreds(to_ip(cast(str, ip)), user="root")
                    info = ctx.merge_node(creds, {'ceph-osd'})
                    info.params.setdefault('ceph-osds', []).extend(info.__dict__.copy() for info in osds_info)
                    assert 'ceph' not in info.params or info.params['ceph'] == ceph_params
                    info.params['ceph'] = ceph_params
                logger.debug("Found %s nodes with ceph-osd role", len(ips))
            except Exception as exc:
                if not ignore_errors:
                    logger.exception("OSD discovery failed")
                    raise StopTestError()
                else:
                    logger.warning("OSD discovery failed %s", exc)

            try:
                counter = 0
                for counter, ip in enumerate(get_mons_ips(node, ceph_extra_args)):
                    ip = ip_remap.get(ip, ip)
                    creds = ConnCreds(to_ip(cast(str, ip)), user="root")
                    info = ctx.merge_node(creds, {'ceph-mon'})
                    assert 'ceph' not in info.params or info.params['ceph'] == ceph_params
                    info.params['ceph'] = ceph_params
                logger.debug("Found %s nodes with ceph-mon role", counter + 1)
            except Exception as exc:
                if not ignore_errors:
                    logger.exception("MON discovery failed")
                    raise StopTestError()
                else:
                    logger.warning("MON discovery failed %s", exc)


def raw_dev_name(path: str) -> str:
    if path.startswith("/dev/"):
        path = path[5:]
    while path and path[-1].isdigit():
        path = path[:-1]
    return path


class CollectCephInfoStage(Stage):
    config_block = 'ceph'
    priority = StepOrder.UPDATE_NODES_INFO

    def run(self, ctx: TestRun) -> None:
        for node in ctx.nodes:
            if 'ceph_storage_devs' not in node.info.params:
                if 'ceph-osd' in node.info.roles:
                    jdevs = set()  # type: Set[str]
                    sdevs = set()  # type: Set[str]
                    for osd_info in node.info.params['ceph-osds']:
                        for key, sset in [('journal', jdevs), ('storage', sdevs)]:
                            path = osd_info.get(key)
                            if path:
                                dpath = node.conn.fs.get_dev_for_file(path)
                                if isinstance(dpath, bytes):
                                    dpath = dpath.decode('utf8')
                                sset.add(raw_dev_name(dpath))
                    node.info.params['ceph_storage_devs'] = list(sdevs)
                    node.info.params['ceph_journal_devs'] = list(jdevs)
