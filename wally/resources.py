import logging
from typing import Tuple, Dict, cast, List

import numpy


from cephlib.units import b2ssize_10, b2ssize, unit_conversion_coef_f
from cephlib.statistic import NormStatProps, HistoStatProps, calc_norm_stat_props, calc_histo_stat_props
from cephlib.numeric_types import TimeSeries
from cephlib.wally_storage import find_nodes_by_roles, WallyDB
from cephlib.storage_selectors import sum_sensors

from .result_classes import IWallyStorage, SuiteConfig
from .utils import STORAGE_ROLES
from .suits.io.fio import FioJobConfig
from .suits.job import JobConfig
from .data_selectors import get_aggregated


logger = logging.getLogger('wally')


class IOSummary:
    def __init__(self, qd: int, block_size: int, nodes_count:int, bw: NormStatProps, lat: HistoStatProps) -> None:
        self.qd = qd
        self.nodes_count = nodes_count
        self.block_size = block_size
        self.bw = bw
        self.lat = lat


class ResourceNames:
    io_made = "Client IOP made"
    data_tr = "Client data transfered"

    test_send = "Test nodes net send"
    test_recv = "Test nodes net recv"
    test_net = "Test nodes net total"
    test_send_pkt = "Test nodes send pkt"
    test_recv_pkt = "Test nodes recv pkt"
    test_net_pkt = "Test nodes total pkt"

    test_write = "Test nodes disk write"
    test_read = "Test nodes disk read"
    test_write_iop = "Test nodes write IOP"
    test_read_iop = "Test nodes read IOP"
    test_iop = "Test nodes IOP"
    test_rw = "Test nodes disk IO"

    storage_send = "Storage nodes net send"
    storage_recv = "Storage nodes net recv"
    storage_send_pkt = "Storage nodes send pkt"
    storage_recv_pkt = "Storage nodes recv pkt"
    storage_net = "Storage nodes net total"
    storage_net_pkt = "Storage nodes total pkt"

    storage_write = "Storage nodes disk write"
    storage_read = "Storage nodes disk read"
    storage_write_iop = "Storage nodes write IOP"
    storage_read_iop = "Storage nodes read IOP"
    storage_iop = "Storage nodes IOP"
    storage_rw = "Storage nodes disk IO"

    storage_cpu = "Storage nodes CPU"
    storage_cpu_s = "Storage nodes CPU s/IOP"
    storage_cpu_s_b = "Storage nodes CPU s/B"


def avg_dev_div(vec: numpy.ndarray, denom: numpy.ndarray, avg_ranges: int = 10) -> Tuple[float, float]:
    step = min(vec.size, denom.size) // avg_ranges
    assert step >= 1
    vals = []

    whole_sum = denom.sum() / denom.size * step * 0.5
    for i in range(0, avg_ranges):
        s1 = denom[i * step: (i + 1) * step].sum()
        if s1 > 1e-5 and s1 >= whole_sum:
            vals.append(vec[i * step: (i + 1) * step].sum() / s1)

    assert len(vals) > 1
    return vec.sum() / denom.sum(), numpy.std(vals, ddof=1)


iosum_cache = {}  # type: Dict[Tuple[str, str], IOSummary]


def make_iosum(rstorage: IWallyStorage, suite: SuiteConfig, job: FioJobConfig, hist_boxes: int,
               nc: bool = False) -> IOSummary:

    key = (suite.storage_id, job.storage_id)
    if not nc and key in iosum_cache:
        return iosum_cache[key]

    lat = get_aggregated(rstorage, suite.storage_id, job.storage_id, "lat", job.reliable_info_range_s)
    io = get_aggregated(rstorage, suite.storage_id, job.storage_id, "bw", job.reliable_info_range_s)

    res = IOSummary(job.qd,
                    nodes_count=len(suite.nodes_ids),
                    block_size=job.bsize,
                    lat=calc_histo_stat_props(lat, rebins_count=hist_boxes),
                    bw=calc_norm_stat_props(io, hist_boxes))

    if not nc:
        iosum_cache[key] = res

    return res


cpu_load_cache = {}  # type: Dict[Tuple[int, Tuple[str, ...], Tuple[int, int]], Dict[str, TimeSeries]]


def get_cluster_cpu_load(rstorage: IWallyStorage, roles: List[str],
                         time_range: Tuple[int, int], nc: bool = False) -> Dict[str, TimeSeries]:

    key = (id(rstorage), tuple(roles), time_range)
    if not nc and key in cpu_load_cache:
        return cpu_load_cache[key]

    cpu_ts = {}
    cpu_metrics = "idle guest iowait sirq nice irq steal sys user".split()
    nodes = find_nodes_by_roles(rstorage.storage, roles)

    cores_per_node = {}
    for node in rstorage.load_nodes():
        cores_per_node[node.node_id] = sum(cores for _, cores in node.hw_info.cpus)

    for name in cpu_metrics:
        cpu_ts[name] = sum_sensors(rstorage, time_range, node_id=nodes, sensor='system-cpu', metric=name)

    it = iter(cpu_ts.values())
    total_over_time = next(it).data.copy()  # type: numpy.ndarray
    for ts in it:
        if ts is not None:
            total_over_time += ts.data

    total = cpu_ts['idle'].copy(no_data=True)
    total.data = total_over_time
    cpu_ts['total'] = total

    if not nc:
        cpu_load_cache[key] = cpu_ts

    return cpu_ts


def get_resources_usage(suite: SuiteConfig,
                        job: JobConfig,
                        rstorage: IWallyStorage,
                        large_block: int = 256,
                        hist_boxes: int = 10,
                        nc: bool = False) -> Tuple[Dict[str, Tuple[str, float, float]], bool]:

    records = {}  # type: Dict[str, Tuple[str, float, float]]
    if not nc:
        records = rstorage.get_job_info(suite, job, WallyDB.resource_usage_rel)
        if records is not None:
            records = records.copy()
            iops_ok = records.pop('iops_ok')
            return records, iops_ok

    fjob = cast(FioJobConfig, job)
    iops_ok = fjob.bsize < large_block

    io_sum = make_iosum(rstorage, suite, fjob, hist_boxes)

    tot_io_coef = unit_conversion_coef_f(io_sum.bw.units, "Bps")
    io_transfered = io_sum.bw.data * tot_io_coef

    records = {
        ResourceNames.data_tr: (b2ssize(io_transfered.sum()) + "B", None, None)
    }

    if iops_ok:
        ops_done = io_transfered / (fjob.bsize * unit_conversion_coef_f("KiBps", "Bps"))
        records[ResourceNames.io_made] = (b2ssize_10(ops_done.sum()) + "OP", None, None)
    else:
        ops_done = None

    all_metrics = [
        (ResourceNames.test_send, 'net-io', 'send_bytes', b2ssize, ['testnode'], "B", io_transfered),
        (ResourceNames.test_recv, 'net-io', 'recv_bytes', b2ssize, ['testnode'], "B", io_transfered),
        (ResourceNames.test_send_pkt, 'net-io', 'send_packets', b2ssize_10, ['testnode'], "pkt", ops_done),
        (ResourceNames.test_recv_pkt, 'net-io', 'recv_packets', b2ssize_10, ['testnode'], "pkt", ops_done),

        (ResourceNames.test_write, 'block-io', 'sectors_written', b2ssize, ['testnode'], "B", io_transfered),
        (ResourceNames.test_read, 'block-io', 'sectors_read', b2ssize, ['testnode'], "B", io_transfered),
        (ResourceNames.test_write_iop, 'block-io', 'writes_completed', b2ssize_10, ['testnode'], "OP", ops_done),
        (ResourceNames.test_read_iop, 'block-io', 'reads_completed', b2ssize_10, ['testnode'], "OP", ops_done),

        (ResourceNames.storage_send, 'net-io', 'send_bytes', b2ssize, STORAGE_ROLES, "B", io_transfered),
        (ResourceNames.storage_recv, 'net-io', 'recv_bytes', b2ssize, STORAGE_ROLES, "B", io_transfered),
        (ResourceNames.storage_send_pkt, 'net-io', 'send_packets', b2ssize_10, STORAGE_ROLES, "OP", ops_done),
        (ResourceNames.storage_recv_pkt, 'net-io', 'recv_packets', b2ssize_10, STORAGE_ROLES, "OP", ops_done),

        (ResourceNames.storage_write, 'block-io', 'sectors_written', b2ssize, STORAGE_ROLES, "B", io_transfered),
        (ResourceNames.storage_read, 'block-io', 'sectors_read', b2ssize, STORAGE_ROLES, "B", io_transfered),
        (ResourceNames.storage_write_iop, 'block-io', 'writes_completed', b2ssize_10, STORAGE_ROLES, "OP", ops_done),
        (ResourceNames.storage_read_iop, 'block-io', 'reads_completed', b2ssize_10, STORAGE_ROLES, "OP", ops_done),
    ]

    all_agg = {}

    for vname, sensor, metric, ffunc, roles, units, service_provided_count in all_metrics:
        if service_provided_count is None:
            continue

        nodes = find_nodes_by_roles(rstorage.storage, roles)
        res_ts = sum_sensors(rstorage, job.reliable_info_range_s, node_id=nodes, sensor=sensor, metric=metric)
        if res_ts is None:
            continue

        data = res_ts.data
        if units == "B":
            data = data * unit_conversion_coef_f(res_ts.units, "B")

        avg, dev = avg_dev_div(data, service_provided_count)
        if avg < 0.1:
            dev = None
        records[vname] = (ffunc(data.sum()) + units, avg, dev)
        all_agg[vname] = data

    # cpu usage
    stor_cores_count = None
    for node in rstorage.load_nodes():
        if node.roles.intersection(STORAGE_ROLES):
            if stor_cores_count is None:
                stor_cores_count = sum(cores for _, cores in node.hw_info.cpus)
            else:
                assert stor_cores_count == sum(cores for _, cores in node.hw_info.cpus)

    assert stor_cores_count != 0

    cpu_ts = get_cluster_cpu_load(rstorage, STORAGE_ROLES, job.reliable_info_range_s)
    cpus_used_sec = (1.0 - (cpu_ts['idle'].data + cpu_ts['iowait'].data) / cpu_ts['total'].data) * stor_cores_count
    used_s = b2ssize_10(cpus_used_sec.sum()) + 's'

    all_agg[ResourceNames.storage_cpu] = cpus_used_sec

    if ops_done is not None:
        records[ResourceNames.storage_cpu_s] = (used_s, *avg_dev_div(cpus_used_sec, ops_done))

    records[ResourceNames.storage_cpu_s_b] = (used_s, *avg_dev_div(cpus_used_sec, io_transfered))

    cums = [
        (ResourceNames.test_iop, ResourceNames.test_read_iop, ResourceNames.test_write_iop,
         b2ssize_10, "OP", ops_done),
        (ResourceNames.test_rw, ResourceNames.test_read, ResourceNames.test_write, b2ssize, "B", io_transfered),
        (ResourceNames.test_net, ResourceNames.test_send, ResourceNames.test_recv, b2ssize, "B", io_transfered),
        (ResourceNames.test_net_pkt, ResourceNames.test_send_pkt, ResourceNames.test_recv_pkt, b2ssize_10,
         "pkt", ops_done),

        (ResourceNames.storage_iop, ResourceNames.storage_read_iop, ResourceNames.storage_write_iop, b2ssize_10,
         "OP", ops_done),
        (ResourceNames.storage_rw, ResourceNames.storage_read, ResourceNames.storage_write, b2ssize, "B",
         io_transfered),
        (ResourceNames.storage_net, ResourceNames.storage_send, ResourceNames.storage_recv, b2ssize, "B",
         io_transfered),
        (ResourceNames.storage_net_pkt, ResourceNames.storage_send_pkt, ResourceNames.storage_recv_pkt, b2ssize_10,
         "pkt", ops_done),
    ]

    for vname, name1, name2, ffunc, units, service_provided_masked in cums:
        if service_provided_masked is None:
            continue
        if name1 in all_agg and name2 in all_agg:
            agg = all_agg[name1] + all_agg[name2]
            avg, dev = avg_dev_div(agg, service_provided_masked)
            if avg < 0.1:
                dev = None
            records[vname] = (ffunc(agg.sum()) + units, avg, dev)

    if not nc:
        toflt = lambda x: float(x) if x is not None else None

        for name, (v1, v2, v3) in list(records.items()):
            records[name] = v1, toflt(v2), toflt(v3)

        srecords = records.copy()
        srecords['iops_ok'] = iops_ok
        rstorage.put_job_info(suite, job, WallyDB.resource_usage_rel, srecords)

    return records, iops_ok
