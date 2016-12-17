import os.path
import unittest


from oktest import ok, main, test


from wally.suits.io import fio_task_parser

code_test_defaults = """
[defaults]
wait_for_previous
buffered=0
iodepth=2
RUNTIME=20

[sec1]
group_reporting
time_based
softrandommap=1
filename=/tmp/xxx
size=5G
ramp_time=20
runtime={RUNTIME}
blocksize=1m
rw=read
direct=1
numjobs=1
some_extra=1.2314

[sec2]
group_reporting
time_based
iodepth=1
softrandommap=1
filename=/tmp/xxx
size=5G
ramp_time=20
runtime={RUNTIME}
blocksize=1m
rw=read
direct=1
numjobs=1
some_extra=1.2314
"""

defaults = """
[defaults]
wait_for_previous
group_reporting
time_based
buffered=0
iodepth=1
softrandommap=1
filename=/tmp/xxx
size=5G
ramp_time=20
runtime=20
blocksize=1m
rw=read
direct=1
numjobs=1
"""

code_test_auto_params_1 = defaults + """
[defaults]
RUNTIME=30

[sec1_{TEST_SUMM}]
ramp_time={% 20, 40 %}
runtime={RUNTIME}
blocksize={% 4k, 4m %}
"""


code_test_uniq = defaults + """
[defaults]
REPCOUNT=2
RUNTIME=30

[sec1_{TEST_SUMM}_{UNIQ} * 3]

[sec2_{TEST_SUMM}_{UNIQ} * {REPCOUNT}]
"""

code_test_cycles_default = defaults + """
[defaults]
REPCOUNT=2
RUNTIME={% 30, 60 %}

[sec1_{TEST_SUMM}_{UNIQ} * 3]
runtime={RUNTIME}
blocksize={% 4k, 4m %}
"""


class AgentTest(unittest.TestCase):
    @test("test_parse_value")
    def test_parse_value(self):
        x = "asdfasd adsd d"
        ok(fio_task_parser.parse_value(x)) == x
        ok(fio_task_parser.parse_value("10 2")) == "10 2"
        ok(fio_task_parser.parse_value("None")).is_(None)
        ok(fio_task_parser.parse_value("10")) == 10
        ok(fio_task_parser.parse_value("20")) == 20
        ok(fio_task_parser.parse_value("10.1") - 10.1) < 1E-7
        ok(fio_task_parser.parse_value("{% 10, 20 %}")) == [10, 20]
        ok(fio_task_parser.parse_value("{% 10,20 %}")) == [10, 20]

    code_test_compile_simplest = defaults + """
[sec1]
some_extra=1.2314
"""

    @test("test_compile_simplest")
    def test_compile_simplest(self):
        sections = fio_task_parser.parse_all_in_1(self.code_test_compile_simplest, {})
        sections = list(sections)

        ok(len(sections)) == 1
        sec1 = sections[0]
        ok(sec1.name) == "sec1"
        vals = sec1.vals
        ok(vals['wait_for_previous']).is_(None)
        ok(vals['iodepth']) == 1
        ok(vals['some_extra'] - 1.2314) < 1E-7

    code_test_params_in_defaults = defaults + """
[defaults]
RUNTIME=20

[sec1]
runtime={RUNTIME}
"""

    @test("test_compile_defaults")
    def test_compile_defaults(self):
        sections = fio_task_parser.parse_all_in_1(self.code_test_params_in_defaults, {})
        sections = list(sections)

        ok(len(sections)) == 1
        sec1 = sections[0]
        ok(sec1.name) == "sec1"
        vals = sec1.vals
        ok(vals['wait_for_previous']).is_(None)
        ok(vals['iodepth']) == 1
        ok(vals['runtime']) == 20

    @test("test_defaults")
    def test_defaults(self):
        sections = fio_task_parser.parse_all_in_1(code_test_defaults, {})
        sections = list(sections)

        ok(len(sections)) == 2
        sec1, sec2 = sections

        ok(sec1.name) == "sec1"
        ok(sec2.name) == "sec2"

        ok(sec1.vals['wait_for_previous']).is_(None)
        ok(sec2.vals['wait_for_previous']).is_(None)

        ok(sec1.vals['iodepth']) == 2
        ok(sec2.vals['iodepth']) == 1

        ok(sec1.vals['buffered']) == 0
        ok(sec2.vals['buffered']) == 0

    code_test_ext_params = defaults + """
[sec1]
runtime={RUNTIME}
"""

    @test("test_external_params")
    def test_external_params(self):
        with self.assertRaises(KeyError):
            sections = fio_task_parser.parse_all_in_1(self.code_test_ext_params, {})
            list(sections)

        sections = fio_task_parser.parse_all_in_1(self.code_test_ext_params, {'RUNTIME': 20})
        sections = list(sections)

    code_test_cycle = defaults + """
[sec1]
runtime={RUNTIME}
ramp_time={% 20, 40 %}
"""

    @test("test_cycle")
    def test_cycle(self):
        sections = fio_task_parser.parse_all_in_1(self.code_test_cycle, {'RUNTIME': 20})
        sections = list(sections)
        ok(len(sections)) == 2
        ok(sections[0].vals['ramp_time']) == 20
        ok(sections[1].vals['ramp_time']) == 40

    code_test_cycles = defaults + """
[sec1]
ramp_time={% 20, 40 %}
runtime={RUNTIME}
blocksize={% 4k, 4m %}
"""

    @test("test_cycles")
    def test_cycles(self):
        sections = fio_task_parser.parse_all_in_1(self.code_test_cycles, {'RUNTIME': 20})
        sections = list(sections)
        ok(len(sections)) == 4

        combinations = [
            (section.vals['ramp_time'], section.vals['blocksize'])
            for section in sections
        ]

        combinations.sort()

        ok(combinations) == [(20, '4k'), (20, '4m'), (40, '4k'), (40, '4m')]

    @test("test_time_estimate")
    def test_time_estimate(self):
        sections = fio_task_parser.parse_all_in_1(self.code_test_cycles, {'RUNTIME': 20})
        sections = list(sections)
        etime = fio_task_parser.calculate_execution_time(sections)

        ok(etime) == 20 * 4 + 20 * 2 + 40 * 2
        ok(fio_task_parser.sec_to_str(etime)) == "0:03:20"

    code_test_cycles2 = defaults + """
[sec1 * 7]
ramp_time={% 20, 40 %}
runtime={RUNTIME}
blocksize={% 4k, 4m %}
"""

    @test("test_time_estimate")
    def test_time_estimate_large(self):
        sections = fio_task_parser.parse_all_in_1(self.code_test_cycles2, {'RUNTIME': 30})
        sections = list(sections)

        ok(sections[0].name) == 'sec1'
        ok(len(sections)) == 7 * 4

        etime = fio_task_parser.calculate_execution_time(sections)
        # ramptime optimization
        expected_time = (20 + 30 + 30 * 6) * 2
        expected_time += (40 + 30 + 30 * 6) * 2
        ok(etime) == expected_time

    code_test_cycles3 = defaults + """
[sec1 * 7]
ramp_time={% 20, 40 %}
runtime={RUNTIME}
blocksize={% 4k, 4m %}

[sec2 * 7]
ramp_time={% 20, 40 %}
runtime={RUNTIME}
blocksize={% 4k, 4m %}
"""

    @test("test_time_estimate2")
    def test_time_estimate_large2(self):
        sections = fio_task_parser.parse_all_in_1(self.code_test_cycles3, {'RUNTIME': 30})
        sections = list(sections)

        ok(sections[0].name) == 'sec1'
        ok(sections[1].name) == 'sec1'
        ok(len(sections)) == 7 * 4 * 2

        etime = fio_task_parser.calculate_execution_time(sections)
        # ramptime optimization
        expected_time = (20 + 30 + 30 * 6) * 2
        expected_time += (40 + 30 + 30 * 6) * 2
        ok(etime) == expected_time * 2

    code_test_repeats = defaults + """
[defaults]
REPCOUNT=2
[sec1 * 3]
[sec2 * {REPCOUNT}]
"""

    @test("test_repeat")
    def test_repeat(self):
        sections = fio_task_parser.parse_all_in_1(self.code_test_repeats, {})
        sections = list(sections)
        ok(len(sections)) == 2 + 3
        ok(sections[0].name) == 'sec1'
        ok(sections[1].name) == 'sec1'
        ok(sections[2].name) == 'sec1'
        ok(sections[3].name) == 'sec2'
        ok(sections[4].name) == 'sec2'

    @test("test_real_tasks")
    def test_real_tasks(self):
        tasks_dir = os.path.dirname(fio_task_parser.__file__)
        fname = os.path.join(tasks_dir, 'io_scenario_ceph.cfg')
        fc = open(fname).read()

        sections = P(fc, {'FILENAME': '/dev/null'})
        sections = list(sections)

        ok(len(sections)) == 7 * 9 * 4 + 7

        etime = fio_task_parser.calculate_execution_time(sections)
        # ramptime optimization
        expected_time = (60 * 7 + 30) * 9 * 4 + (60 * 7 + 30)
        ok(etime) == expected_time

if __name__ == '__main__':
    main()

# def do_run_fio_fake(bconf):
#     def estimate_iops(sz, bw, lat):
#         return 1 / (lat + float(sz) / bw)
#     global count
#     count += 1
#     parsed_out = []

#     BW = 120.0 * (1024 ** 2)
#     LAT = 0.003

#     for name, cfg in bconf:
#         sz = to_bytes(cfg['blocksize'])
#         curr_lat = LAT * ((random.random() - 0.5) * 0.1 + 1)
#         curr_ulat = curr_lat * 1000000
#         curr_bw = BW * ((random.random() - 0.5) * 0.1 + 1)
#         iops = estimate_iops(sz, curr_bw, curr_lat)
#         bw = iops * sz

#         res = {'ctx': 10683,
#                'error': 0,
#                'groupid': 0,
#                'jobname': name,
#                'majf': 0,
#                'minf': 30,
#                'read': {'bw': 0,
#                         'bw_agg': 0.0,
#                         'bw_dev': 0.0,
#                         'bw_max': 0,
#                         'bw_mean': 0.0,
#                         'bw_min': 0,
#                         'clat': {'max': 0,
#                                  'mean': 0.0,
#                                  'min': 0,
#                                  'stddev': 0.0},
#                         'io_bytes': 0,
#                         'iops': 0,
#                         'lat': {'max': 0, 'mean': 0.0,
#                                 'min': 0, 'stddev': 0.0},
#                         'runtime': 0,
#                         'slat': {'max': 0, 'mean': 0.0,
#                                  'min': 0, 'stddev': 0.0}
#                         },
#                'sys_cpu': 0.64,
#                'trim': {'bw': 0,
#                         'bw_agg': 0.0,
#                         'bw_dev': 0.0,
#                         'bw_max': 0,
#                         'bw_mean': 0.0,
#                         'bw_min': 0,
#                         'clat': {'max': 0,
#                                  'mean': 0.0,
#                                  'min': 0,
#                                  'stddev': 0.0},
#                         'io_bytes': 0,
#                         'iops': 0,
#                         'lat': {'max': 0, 'mean': 0.0,
#                                 'min': 0, 'stddev': 0.0},
#                         'runtime': 0,
#                         'slat': {'max': 0, 'mean': 0.0,
#                                  'min': 0, 'stddev': 0.0}
#                         },
#                'usr_cpu': 0.23,
#                'write': {'bw': 0,
#                          'bw_agg': 0,
#                          'bw_dev': 0,
#                          'bw_max': 0,
#                          'bw_mean': 0,
#                          'bw_min': 0,
#                          'clat': {'max': 0, 'mean': 0,
#                                   'min': 0, 'stddev': 0},
#                          'io_bytes': 0,
#                          'iops': 0,
#                          'lat': {'max': 0, 'mean': 0,
#                                  'min': 0, 'stddev': 0},
#                          'runtime': 0,
#                          'slat': {'max': 0, 'mean': 0.0,
#                                   'min': 0, 'stddev': 0.0}
#                          }
#                }

#         if cfg['rw'] in ('read', 'randread'):
#             key = 'read'
#         elif cfg['rw'] in ('write', 'randwrite'):
#             key = 'write'
#         else:
#             raise ValueError("Uknown op type {0}".format(key))

#         res[key]['bw'] = bw
#         res[key]['iops'] = iops
#         res[key]['runtime'] = 30
#         res[key]['io_bytes'] = res[key]['runtime'] * bw
#         res[key]['bw_agg'] = bw
#         res[key]['bw_dev'] = bw / 30
#         res[key]['bw_max'] = bw * 1.5
#         res[key]['bw_min'] = bw / 1.5
#         res[key]['bw_mean'] = bw
#         res[key]['clat'] = {'max': curr_ulat * 10, 'mean': curr_ulat,
#                             'min': curr_ulat / 2, 'stddev': curr_ulat}
#         res[key]['lat'] = res[key]['clat'].copy()
#         res[key]['slat'] = res[key]['clat'].copy()

#         parsed_out.append(res)

#     return zip(parsed_out, bconf)
