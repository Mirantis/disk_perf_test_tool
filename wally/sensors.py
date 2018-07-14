import bz2
import time
import array
import logging
from typing import Dict, Tuple, Optional, Any

import numpy

from cephlib import sensors_rpc_plugin
from cephlib.units import b2ssize
from cephlib.wally_storage import WallyDB

from . import utils
from .test_run_class import TestRun
from .result_classes import DataSource
from .stage import Stage, StepOrder


plugin_fname = sensors_rpc_plugin.__file__.rsplit(".", 1)[0] + ".py"
SENSORS_PLUGIN_CODE = open(plugin_fname, "rb").read()  # type: bytes


logger = logging.getLogger("wally")


# TODO(koder): in case if node has more than one role sensor settings might be incorrect
class StartSensorsStage(Stage):
    priority = StepOrder.START_SENSORS
    config_block = 'sensors'

    def run(self, ctx: TestRun) -> None:
        if  array.array('L').itemsize != 8:
            message = "Python array.array('L') items should be 8 bytes in size, not {}." + \
                " Can't provide sensors on this platform. Disable sensors in config and retry"
            logger.critical(message.format(array.array('L').itemsize))
            raise utils.StopTestError()

        # TODO: need carefully fix this
        # sensors config is:
        #   role:
        #     sensor: [str]
        # or
        #  role:
        #     sensor:
        #        allowed: [str]
        #        dissallowed: [str]
        #        params: Any
        per_role_config = {}  # type: Dict[str, Dict[str, str]]

        for name, val in ctx.config.sensors.roles_mapping.raw().items():
            if isinstance(val, str):
                val = {vl.strip(): (".*" if vl.strip() != 'ceph' else {}) for vl in val.split(",")}
            elif isinstance(val, list):
                val = {vl: (".*" if vl != 'ceph' else {}) for vl in val}
            per_role_config[name] = val

        if 'all' in per_role_config:
            all_vl = per_role_config.pop('all')
            all_roles = set(per_role_config)

            for node in ctx.nodes:
                all_roles.update(node.info.roles)  # type: ignore

            for name, vals in list(per_role_config.items()):
                new_vals = all_vl.copy()
                new_vals.update(vals)
                per_role_config[name] = new_vals

        for node in ctx.nodes:
            node_cfg = {}  # type: Dict[str, Dict[str, str]]
            for role in node.info.roles:
                node_cfg.update(per_role_config.get(role, {}))  # type: ignore

            nid = node.node_id
            if node_cfg:
                # ceph requires additional settings
                if 'ceph' in node_cfg:
                    node_cfg['ceph'].update(node.info.params['ceph'])
                    node_cfg['ceph']['osds'] = [osd['id'] for osd in node.info.params['ceph-osds']]  # type: ignore

                logger.debug("Setting up sensors RPC plugin for node %s", nid)
                node.upload_plugin("sensors", SENSORS_PLUGIN_CODE)
                ctx.sensors_run_on.add(nid)
                logger.debug("Start monitoring node %s", nid)
                node.conn.sensors.start(node_cfg)
            else:
                logger.debug("Skip monitoring node %s, as no sensors selected", nid)


def collect_sensors_data(ctx: TestRun,
                         stop: bool = False,
                         before_test: bool = False):
    total_sz = 0

    # ceph pg and pool data collected separatelly
    cluster_metrics = getattr(ctx.config.sensors, 'cluster', [])

    pgs_io = 'ceph-pgs-io' in cluster_metrics
    pools_io = 'ceph-pools-io' in cluster_metrics

    if pgs_io or pools_io:
        assert ctx.ceph_master_node is not None

        def collect() -> Tuple[Optional[Any], Optional[Any]]:
            pg_dump = ctx.ceph_master_node.run(f"ceph {ctx.ceph_extra_args} pg dump --format json") if pgs_io else None
            pools_dump = ctx.ceph_master_node.run(f"rados {ctx.ceph_extra_args} df --format json") if pools_io else None
            return pg_dump, pools_dump
        future = ctx.get_pool().submit(collect)
    else:
        future = None

    ctime = int(time.time())

    if not before_test:
        logger.info("Start loading sensors")
        for node in ctx.nodes:
            node_id = node.node_id
            if node_id in ctx.sensors_run_on:
                func = node.conn.sensors.stop if stop else node.conn.sensors.get_updates

                # hack to calculate total transferred size
                offset_map, compressed_blob, compressed_collected_at_b = func()
                data_tpl = (offset_map, compressed_blob, compressed_collected_at_b)

                total_sz += len(compressed_blob) + len(compressed_collected_at_b) + sum(map(len, offset_map)) + \
                    16 * len(offset_map)

                for path, value, is_array, units in sensors_rpc_plugin.unpack_rpc_updates(data_tpl):
                    if path == 'collected_at':
                        ds = DataSource(node_id=node_id, metric='collected_at', tag='csv')
                        ctx.rstorage.append_sensor(numpy.array(value), ds, units)
                    else:
                        sensor, dev, metric = path.split(".")
                        ds = DataSource(node_id=node_id, metric=metric, dev=dev, sensor=sensor, tag='csv')
                        if is_array:
                            ctx.rstorage.append_sensor(numpy.array(value), ds, units)
                        else:
                            if metric == 'historic':
                                value = bz2.compress(value)
                                tag = 'bz2'
                            else:
                                assert metric == 'perf_dump'
                                tag = 'txt'
                            ctx.storage.put_raw(value, WallyDB.ceph_metric(node_id=node_id,
                                                                           metric=metric,
                                                                           time=ctime,
                                                                           tag=tag))

    if future:
        pgs_info, pools_info = future.result()
        if pgs_info:
            total_sz += len(pgs_info)
            ctx.storage.put_raw(bz2.compress(pgs_info.encode('utf8')), WallyDB.pgs_io.format(time=ctime))

        if pools_info:
            total_sz += len(pools_info)
            ctx.storage.put_raw(bz2.compress(pools_info.encode('utf8')), WallyDB.pools_io.format(time=ctime))

    logger.info("Download %sB of sensors data", b2ssize(total_sz))



class CollectSensorsStage(Stage):
    priority = StepOrder.COLLECT_SENSORS
    config_block = 'sensors'

    def run(self, ctx: TestRun) -> None:
        collect_sensors_data(ctx, True, False)

