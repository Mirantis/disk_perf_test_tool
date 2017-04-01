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
from .utils import StopTestError, to_ip


from cephlib import discover
from cephlib.discover import OSDInfo


logger = logging.getLogger("wally")


def get_osds_info(node: IRPCNode, ceph_extra_args: str = "") -> Dict[IP, List[OSDInfo]]:
    """Get set of osd's ip"""
    res = {}  # type: Dict[IP, List[OSDInfo]]
    return {IP(ip): osd_info_list
            for ip, osd_info_list in discover.get_osds_nodes(node.run, ceph_extra_args)}


def get_mons_ips(node: IRPCNode, ceph_extra_args: str = "") -> Set[IP]:
    """Return mon ip set"""
    return {IP(ip) for ip in discover.get_mons_nodes(node.run, ceph_extra_args).values()}


class DiscoverCephStage(Stage):
    config_block = 'ceph'
    priority = StepOrder.DISCOVER

    def run(self, ctx: TestRun) -> None:
        """Return list of ceph's nodes NodeInfo"""

        if 'ceph' not in ctx.config.discovery:
            logger.debug("Skip ceph discovery due to config setting")
            return

        if 'all_nodes' in ctx.storage:
            logger.debug("Skip ceph discovery, use previously discovered nodes")
            return

        if 'metadata' in ctx.config.discovery:
            logger.exception("Ceph metadata discovery is not implemented")
            raise StopTestError()

        ignore_errors = 'ignore_errors' in ctx.config.discovery
        ceph = ctx.config.ceph
        root_node_uri = cast(str, ceph.root_node)
        cluster = ceph.get("cluster", "ceph")

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

            ssh_key = node.get_file_content("~/.ssh/id_rsa")

            try:
                ips = set()
                for ip, osds_info in get_osds_info(node, ceph_extra_args).items():
                    ips.add(ip)
                    creds = ConnCreds(to_ip(cast(str, ip)), user="root", key=ssh_key)
                    info = ctx.merge_node(creds, {'ceph-osd'})
                    info.params.setdefault('ceph-osds', []).extend(osds_info)
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
                    creds = ConnCreds(to_ip(cast(str, ip)), user="root", key=ssh_key)
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
