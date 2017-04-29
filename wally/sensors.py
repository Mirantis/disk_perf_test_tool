import array
import logging
from typing import List, Dict, Tuple

import numpy

from cephlib import sensors_rpc_plugin

from . import utils
from .test_run_class import TestRun
from .result_classes import DataSource
from .stage import Stage, StepOrder
from .hlstorage import ResultStorage


plugin_fname = sensors_rpc_plugin.__file__.rsplit(".", 1)[0] + ".py"
SENSORS_PLUGIN_CODE = open(plugin_fname, "rb").read()  # type: bytes


logger = logging.getLogger("wally")


sensor_units = {
    "system-cpu.idle": "",
    "system-cpu.nice": "",
    "system-cpu.user": "",
    "system-cpu.sys": "",
    "system-cpu.iowait": "",
    "system-cpu.irq": "",
    "system-cpu.sirq": "",
    "system-cpu.steal": "",
    "system-cpu.guest": "",

    "system-cpu.procs_blocked": "",
    "system-cpu.procs_queue_x10": "",

    "net-io.recv_bytes": "B",
    "net-io.recv_packets": "",
    "net-io.send_bytes": "B",
    "net-io.send_packets": "",

    "block-io.io_queue": "",
    "block-io.io_time": "ms",
    "block-io.reads_completed": "",
    "block-io.rtime": "ms",
    "block-io.sectors_read": "B",
    "block-io.sectors_written": "B",
    "block-io.writes_completed": "",
    "block-io.wtime": "ms",
    "block-io.weighted_io_time": "ms"
}


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


def collect_sensors_data(ctx: TestRun, stop: bool = False):
    rstorage = ResultStorage(ctx.storage)
    raw_skipped = False
    for node in ctx.nodes:
        node_id = node.node_id
        if node_id in ctx.sensors_run_on:

            if stop:
                func = node.conn.sensors.stop
            else:
                func = node.conn.sensors.get_updates

            # TODO: units should came along with data
            # TODO: process raw sensors data

            for path, value, is_parsed in sensors_rpc_plugin.unpack_rpc_updates(func()):
                if not is_parsed:
                    if not raw_skipped:
                        logger.warning("Raw sensors data at path %r and, maybe, others are skipped", path)
                    raw_skipped = True
                    continue

                if path == 'collected_at':
                    ds = DataSource(node_id=node_id, metric='collected_at')
                    units = 'us'
                else:
                    sensor, dev, metric = path.split(".")
                    ds = DataSource(node_id=node_id, metric=metric, dev=dev, sensor=sensor)
                    units = sensor_units["{}.{}".format(sensor, metric)]

                rstorage.append_sensor(numpy.array(value), ds, units)


class CollectSensorsStage(Stage):
    priority = StepOrder.COLLECT_SENSORS
    config_block = 'sensors'

    def run(self, ctx: TestRun) -> None:
        collect_sensors_data(ctx, True)


# def delta(func, only_upd=True):
#     prev = {}
#     while True:
#         for dev_name, vals in func():
#             if dev_name not in prev:
#                 prev[dev_name] = {}
#                 for name, (val, _) in vals.items():
#                     prev[dev_name][name] = val
#             else:
#                 dev_prev = prev[dev_name]
#                 res = {}
#                 for stat_name, (val, accum_val) in vals.items():
#                     if accum_val:
#                         if stat_name in dev_prev:
#                             delta = int(val) - int(dev_prev[stat_name])
#                             if not only_upd or 0 != delta:
#                                 res[stat_name] = str(delta)
#                         dev_prev[stat_name] = val
#                     elif not only_upd or '0' != val:
#                         res[stat_name] = val
#
#                 if only_upd and len(res) == 0:
#                     continue
#                 yield dev_name, res
#         yield None, None
#
#


