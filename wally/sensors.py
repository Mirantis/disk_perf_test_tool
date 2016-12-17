from typing import List, Dict, Tuple, Any

from .test_run_class import TestRun
from . import sensors_rpc_plugin
from .stage import Stage, StepOrder

plugin_fname = sensors_rpc_plugin.__file__.rsplit(".", 1)[0] + ".py"
SENSORS_PLUGIN_CODE = open(plugin_fname).read()


# TODO(koder): in case if node has more than one role sensor settigns might be incorrect

class StartSensorsStage(Stage):
    priority = StepOrder.START_SENSORS
    config_block = 'sensors'

    def run(self, ctx: TestRun) -> None:
        if 'sensors' not in ctx.config:
            return

        per_role_config = {}  # type: Dict[str, Dict[str, str]]
        for name, val in ctx.config['sensors'].copy():
            if isinstance(val, str):
                val = {vl.strip(): ".*" for vl in val.split(",")}
            elif isinstance(val, list):
                val = {vl: ".*" for vl in val}
            per_role_config[name] = val

        if 'all' in per_role_config:
            all_vl = per_role_config.pop('all')
            all_roles = set(per_role_config)

            for node in ctx.nodes:
                all_roles.update(node.info.roles)

            for name, vals in list(per_role_config.items()):
                new_vals = all_vl.copy()
                new_vals.update(vals)
                per_role_config[name] = new_vals

        for node in ctx.nodes:
            node_cfg = {}  # type: Dict[str, str]
            for role in node.info.roles:
                node_cfg.update(per_role_config.get(role, {}))

            if node_cfg:
                node.conn.upload_plugin(SENSORS_PLUGIN_CODE)
                ctx.sensors_run_on.add(node.info.node_id())
            node.conn.sensors.start()


class CollectSensorsStage(Stage):
    priority = StepOrder.COLLECT_SENSORS
    config_block = 'sensors'

    def run(self, ctx: TestRun) -> None:
        for node in ctx.nodes:
            node_id = node.info.node_id()
            if node_id in ctx.sensors_run_on:

                data, collected_at = node.conn.sensors.stop()  # type: Dict[Tuple[str, str], List[int]], List[float]

                mstore = ctx.storage.sub_storage("metric", node_id)
                for (source_name, sensor_name), values in data.items():
                    mstore[source_name, sensor_name] = values
                    mstore["collected_at"] = collected_at


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


