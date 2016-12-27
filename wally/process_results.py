# put all result preprocessing here
# selection, aggregation

from .stage import Stage, StepOrder
from .test_run_class import TestRun
from .statistic import calc_norm_stat_props, NormStatProps
from .result_classes import NormStatProps

class CalcStatisticStage(Stage):
    priority = StepOrder.TEST + 1

    def run(self, ctx: TestRun) -> None:
        results = {}

        for is_file, name in ctx.storage.list("result"):
            if is_file:
                continue

            path = "result/{}".format(name)
            info = ctx.storage.get("result/{}/info".format(name))

            if info['test'] == 'fio':
                for node in info['nodes']:
                    data_path = "{}/measurement/{}".format(path, node)

                    iops = ctx.storage.get_array('Q', data_path, 'iops_data')
                    iops_stat_path = "{}/iops_stat".format(data_path)
                    if iops_stat_path in ctx.storage:
                        iops_stat= ctx.storage.load(NormStatProps, iops_stat_path)
                    else:
                        iops_stat = calc_norm_stat_props(iops)
                        ctx.storage.put(iops_stat, iops_stat_path)

                    bw = ctx.storage.get_array('Q', data_path, 'bw_data')
                    bw_stat_path = "{}/bw_stat".format(data_path)
                    if bw_stat_path in ctx.storage:
                        bw_stat = ctx.storage.load(NormStatProps, bw_stat_path)
                    else:
                        bw_stat = calc_norm_stat_props(bw)
                        ctx.storage.put(bw_stat, bw_stat_path)

                    lat = ctx.storage.get_array('L', data_path, 'lat_data')
                    lat_stat = None

                    results[name] = (iops, iops_stat, bw, bw_stat, lat, lat_stat)

        for name, (iops, iops_stat, bw, bw_stat, lat, lat_stat) in results.items():
            print(" -------------------  IOPS -------------------")
            print(iops_stat)  # type: ignore
            print(" -------------------  BW -------------------")
            print(bw_stat)  # type: ignore
            # print(" -------------------  LAT -------------------")
            # print(calc_stat_props(lat))
