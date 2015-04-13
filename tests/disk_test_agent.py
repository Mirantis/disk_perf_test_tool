import re
import sys
import time
import json
import random
import select
import pprint
import argparse
import traceback
import subprocess
import itertools
from collections import OrderedDict


SECTION = 0
SETTING = 1


def get_test_summary(params):
    rw = {"randread": "rr",
          "randwrite": "rw",
          "read": "sr",
          "write": "sw"}[params["rw"]]

    if params.get("direct") == '1':
        sync_mode = 'd'
    elif params.get("sync") == '1':
        sync_mode = 's'
    else:
        sync_mode = 'a'

    th_count = int(params.get('numjobs', '1'))

    return "{0}{1}{2}th{3}".format(rw, sync_mode,
                                   params['blocksize'], th_count)


counter = [0]


def process_section(name, vals, defaults, format_params):
    vals = vals.copy()
    params = format_params.copy()

    if '*' in name:
        name, repeat = name.split('*')
        name = name.strip()
        repeat = int(repeat.format(**params))
    else:
        repeat = 1

    # this code can be optimized
    iterable_names = []
    iterable_values = []
    processed_vals = {}

    for val_name, val in vals.items():
        if val is None:
            processed_vals[val_name] = val
        # remove hardcode
        elif val.startswith('{%'):
            assert val.endswith("%}")
            content = val[2:-2].format(**params)
            iterable_names.append(val_name)
            iterable_values.append(list(i.strip() for i in content.split(',')))
        else:
            processed_vals[val_name] = val.format(**params)

    group_report_err_msg = "Group reporting should be set if numjobs != 1"

    if iterable_values == []:
        params['UNIQ'] = 'UN{0}'.format(counter[0])
        counter[0] += 1
        params['TEST_SUMM'] = get_test_summary(processed_vals)

        if processed_vals.get('numjobs', '1') != '1':
            assert 'group_reporting' in processed_vals, group_report_err_msg

        ramp_time = processed_vals.get('ramp_time')
        for i in range(repeat):
            yield name.format(**params), processed_vals.copy()

            if 'ramp_time' in processed_vals:
                del processed_vals['ramp_time']

        if ramp_time is not None:
            processed_vals['ramp_time'] = ramp_time
    else:
        for it_vals in itertools.product(*iterable_values):
            processed_vals.update(dict(zip(iterable_names, it_vals)))
            params['UNIQ'] = 'UN{0}'.format(counter[0])
            counter[0] += 1
            params['TEST_SUMM'] = get_test_summary(processed_vals)

            if processed_vals.get('numjobs', '1') != '1':
                assert 'group_reporting' in processed_vals,\
                    group_report_err_msg

            ramp_time = processed_vals.get('ramp_time')

            for i in range(repeat):
                yield name.format(**params), processed_vals.copy()
                if 'ramp_time' in processed_vals:
                    del processed_vals['ramp_time']

            if ramp_time is not None:
                processed_vals['ramp_time'] = ramp_time


def calculate_execution_time(combinations):
    time = 0
    for _, params in combinations:
        time += int(params.get('ramp_time', 0))
        time += int(params.get('runtime', 0))
    return time


def parse_fio_config_full(fio_cfg, params=None):
    defaults = {}
    format_params = {}

    if params is None:
        ext_params = {}
    else:
        ext_params = params.copy()

    curr_section = None
    curr_section_name = None

    for tp, name, val in parse_fio_config_iter(fio_cfg):
        if tp == SECTION:
            non_def = curr_section_name != 'defaults'
            if curr_section_name is not None and non_def:
                format_params.update(ext_params)
                for sec in process_section(curr_section_name,
                                           curr_section,
                                           defaults,
                                           format_params):
                    yield sec

            if name == 'defaults':
                curr_section = defaults
            else:
                curr_section = OrderedDict()
                curr_section.update(defaults)
            curr_section_name = name

        else:
            assert tp == SETTING
            assert curr_section_name is not None, "no section name"
            if name == name.upper():
                assert curr_section_name == 'defaults'
                format_params[name] = val
            else:
                curr_section[name] = val

    if curr_section_name is not None and curr_section_name != 'defaults':
        format_params.update(ext_params)
        for sec in process_section(curr_section_name,
                                   curr_section,
                                   defaults,
                                   format_params):
            yield sec


def parse_fio_config_iter(fio_cfg):
    for lineno, line in enumerate(fio_cfg.split("\n")):
        try:
            line = line.strip()

            if line.startswith("#") or line.startswith(";"):
                continue

            if line == "":
                continue

            if line.startswith('['):
                assert line.endswith(']'), "name should ends with ]"
                yield SECTION, line[1:-1], None
            elif '=' in line:
                opt_name, opt_val = line.split('=', 1)
                yield SETTING, opt_name.strip(), opt_val.strip()
            else:
                yield SETTING, line, None
        except Exception as exc:
            pref = "During parsing line number {0}\n".format(lineno)
            raise ValueError(pref + exc.message)


def format_fio_config(fio_cfg):
    res = ""
    for pos, (name, section) in enumerate(fio_cfg):
        if pos != 0:
            res += "\n"

        res += "[{0}]\n".format(name)
        for opt_name, opt_val in section.items():
            if opt_val is None:
                res += opt_name + "\n"
            else:
                res += "{0}={1}\n".format(opt_name, opt_val)
    return res


count = 0


def to_bytes(sz):
    sz = sz.lower()
    try:
        return int(sz)
    except ValueError:
        if sz[-1] == 'm':
            return (1024 ** 2) * int(sz[:-1])
        if sz[-1] == 'k':
            return 1024 * int(sz[:-1])
        raise


def do_run_fio_fake(bconf):
    def estimate_iops(sz, bw, lat):
        return 1 / (lat + float(sz) / bw)
    global count
    count += 1
    parsed_out = []

    BW = 120.0 * (1024 ** 2)
    LAT = 0.003

    for name, cfg in bconf:
        sz = to_bytes(cfg['blocksize'])
        curr_lat = LAT * ((random.random() - 0.5) * 0.1 + 1)
        curr_ulat = curr_lat * 1000000
        curr_bw = BW * ((random.random() - 0.5) * 0.1 + 1)
        iops = estimate_iops(sz, curr_bw, curr_lat)
        bw = iops * sz

        res = {'ctx': 10683,
               'error': 0,
               'groupid': 0,
               'jobname': name,
               'majf': 0,
               'minf': 30,
               'read': {'bw': 0,
                        'bw_agg': 0.0,
                        'bw_dev': 0.0,
                        'bw_max': 0,
                        'bw_mean': 0.0,
                        'bw_min': 0,
                        'clat': {'max': 0,
                                 'mean': 0.0,
                                 'min': 0,
                                 'stddev': 0.0},
                        'io_bytes': 0,
                        'iops': 0,
                        'lat': {'max': 0, 'mean': 0.0,
                                'min': 0, 'stddev': 0.0},
                        'runtime': 0,
                        'slat': {'max': 0, 'mean': 0.0,
                                 'min': 0, 'stddev': 0.0}
                        },
               'sys_cpu': 0.64,
               'trim': {'bw': 0,
                        'bw_agg': 0.0,
                        'bw_dev': 0.0,
                        'bw_max': 0,
                        'bw_mean': 0.0,
                        'bw_min': 0,
                        'clat': {'max': 0,
                                 'mean': 0.0,
                                 'min': 0,
                                 'stddev': 0.0},
                        'io_bytes': 0,
                        'iops': 0,
                        'lat': {'max': 0, 'mean': 0.0,
                                'min': 0, 'stddev': 0.0},
                        'runtime': 0,
                        'slat': {'max': 0, 'mean': 0.0,
                                 'min': 0, 'stddev': 0.0}
                        },
               'usr_cpu': 0.23,
               'write': {'bw': 0,
                         'bw_agg': 0,
                         'bw_dev': 0,
                         'bw_max': 0,
                         'bw_mean': 0,
                         'bw_min': 0,
                         'clat': {'max': 0, 'mean': 0,
                                  'min': 0, 'stddev': 0},
                         'io_bytes': 0,
                         'iops': 0,
                         'lat': {'max': 0, 'mean': 0,
                                 'min': 0, 'stddev': 0},
                         'runtime': 0,
                         'slat': {'max': 0, 'mean': 0.0,
                                  'min': 0, 'stddev': 0.0}
                         }
               }

        if cfg['rw'] in ('read', 'randread'):
            key = 'read'
        elif cfg['rw'] in ('write', 'randwrite'):
            key = 'write'
        else:
            raise ValueError("Uknown op type {0}".format(key))

        res[key]['bw'] = bw
        res[key]['iops'] = iops
        res[key]['runtime'] = 30
        res[key]['io_bytes'] = res[key]['runtime'] * bw
        res[key]['bw_agg'] = bw
        res[key]['bw_dev'] = bw / 30
        res[key]['bw_max'] = bw * 1.5
        res[key]['bw_min'] = bw / 1.5
        res[key]['bw_mean'] = bw
        res[key]['clat'] = {'max': curr_ulat * 10, 'mean': curr_ulat,
                            'min': curr_ulat / 2, 'stddev': curr_ulat}
        res[key]['lat'] = res[key]['clat'].copy()
        res[key]['slat'] = res[key]['clat'].copy()

        parsed_out.append(res)

    return zip(parsed_out, bconf)


def do_run_fio(bconf):
    benchmark_config = format_fio_config(bconf)
    cmd = ["fio", "--output-format=json", "-"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)

    # set timeout
    raw_out, _ = p.communicate(benchmark_config)

    try:
        parsed_out = json.loads(raw_out)["jobs"]
    except Exception:
        msg = "Can't parse fio output: {0!r}\nError: {1}"
        raise ValueError(msg.format(raw_out, traceback.format_exc()))

    return zip(parsed_out, bconf)

# limited by fio
MAX_JOBS = 1000


def next_test_portion(whole_conf, runcycle):
    jcount = 0
    runtime = 0
    bconf = []

    for pos, (name, sec) in enumerate(whole_conf):
        jc = int(sec.get('numjobs', '1'))

        if runcycle is not None:
            curr_task_time = calculate_execution_time([(name, sec)])
        else:
            curr_task_time = 0

        if jc > MAX_JOBS:
            err_templ = "Can't process job {0!r} - too large numjobs"
            raise ValueError(err_templ.format(name))

        if runcycle is not None and len(bconf) != 0:
            rc_ok = curr_task_time + runtime <= runcycle
        else:
            rc_ok = True

        if jc + jcount <= MAX_JOBS and rc_ok:
            runtime += curr_task_time
            jcount += jc
            bconf.append((name, sec))
            continue

        assert len(bconf) != 0
        yield bconf

        runtime = curr_task_time
        jcount = jc
        bconf = [(name, sec)]

    if bconf != []:
        yield bconf


def get_test_sync_mode(jconfig):
        is_sync = jconfig.get("sync", "0") == "1"
        is_direct = jconfig.get("direct_io", "0") == "1"

        if is_sync and is_direct:
            return 'sd'
        elif is_sync:
            return 's'
        elif is_direct:
            return 'd'
        else:
            return 'a'


def add_job_results(jname, job_output, jconfig, res):
    if job_output['write']['iops'] != 0:
        raw_result = job_output['write']
    else:
        raw_result = job_output['read']

    if jname not in res:
        j_res = {}
        j_res["action"] = jconfig["rw"]
        j_res["sync_mode"] = get_test_sync_mode(jconfig)
        j_res["concurence"] = int(jconfig.get("numjobs", 1))
        j_res["blocksize"] = jconfig["blocksize"]
        j_res["jobname"] = job_output["jobname"]
        j_res["timings"] = [int(jconfig.get("runtime", 0)),
                            int(jconfig.get("ramp_time", 0))]
    else:
        j_res = res[jname]
        assert j_res["action"] == jconfig["rw"]
        assert j_res["sync_mode"] == get_test_sync_mode(jconfig)
        assert j_res["concurence"] == int(jconfig.get("numjobs", 1))
        assert j_res["blocksize"] == jconfig["blocksize"]
        assert j_res["jobname"] == job_output["jobname"]

        # ramp part is skipped for all tests, except first
        # assert j_res["timings"] == (jconfig.get("runtime"),
        #                             jconfig.get("ramp_time"))

    def j_app(name, x):
        j_res.setdefault(name, []).append(x)

    # 'bw_dev bw_mean bw_max bw_min'.split()
    # probably fix fio bug - iops is scaled to joncount, but bw - isn't
    j_app("bw_mean", raw_result["bw_mean"] * j_res["concurence"])
    j_app("iops", raw_result["iops"])
    j_app("lat", raw_result["lat"]["mean"])
    j_app("clat", raw_result["clat"]["mean"])
    j_app("slat", raw_result["slat"]["mean"])

    res[jname] = j_res


def run_fio(benchmark_config,
            params,
            runcycle=None,
            raw_results_func=None,
            skip_tests=0,
            fake_fio=False):

    whole_conf = list(parse_fio_config_full(benchmark_config, params))
    whole_conf = whole_conf[skip_tests:]
    res = {}
    curr_test_num = skip_tests
    executed_tests = 0
    try:
        for bconf in next_test_portion(whole_conf, runcycle):

            if fake_fio:
                res_cfg_it = do_run_fio_fake(bconf)
            else:
                res_cfg_it = do_run_fio(bconf)

            res_cfg_it = enumerate(res_cfg_it, curr_test_num)

            for curr_test_num, (job_output, (jname, jconfig)) in res_cfg_it:
                executed_tests += 1
                if raw_results_func is not None:
                    raw_results_func(executed_tests,
                                     [job_output, jname, jconfig])

                assert jname == job_output["jobname"], \
                    "{0} != {1}".format(jname, job_output["jobname"])

                if jname.startswith('_'):
                    continue

                add_job_results(jname, job_output, jconfig, res)

    except (SystemExit, KeyboardInterrupt):
        raise

    except Exception:
        traceback.print_exc()

    return res, executed_tests


def run_benchmark(binary_tp, *argv, **kwargs):
    if 'fio' == binary_tp:
        return run_fio(*argv, **kwargs)
    raise ValueError("Unknown behcnmark {0}".format(binary_tp))


def read_config(fd, timeout=10):
    job_cfg = ""
    etime = time.time() + timeout
    while True:
        wtime = etime - time.time()
        if wtime <= 0:
            raise IOError("No config provided")

        r, w, x = select.select([fd], [], [], wtime)
        if len(r) == 0:
            raise IOError("No config provided")

        char = fd.read(1)
        if '' == char:
            return job_cfg

        job_cfg += char


def estimate_cfg(job_cfg, params):
    bconf = list(parse_fio_config_full(job_cfg, params))
    return calculate_execution_time(bconf)


def sec_to_str(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return "{0}:{1:02d}:{2:02d}".format(h, m, s)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run fio' and return result")
    parser.add_argument("--type", metavar="BINARY_TYPE",
                        choices=['fio'], default='fio',
                        help=argparse.SUPPRESS)
    parser.add_argument("--start-at", metavar="START_AT_UTC", type=int,
                        help="Start execution at START_AT_UTC")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Json output format")
    parser.add_argument("--output", default='-', metavar="FILE_PATH",
                        help="Store results to FILE_PATH")
    parser.add_argument("--estimate", action="store_true", default=False,
                        help="Only estimate task execution time")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Compile config file to fio config")
    parser.add_argument("--num-tests", action="store_true", default=False,
                        help="Show total number of tests")
    parser.add_argument("--runcycle", type=int, default=None,
                        metavar="MAX_CYCLE_SECONDS",
                        help="Max cycle length in seconds")
    parser.add_argument("--show-raw-results", action='store_true',
                        default=False, help="Output raw input and results")
    parser.add_argument("--skip-tests", type=int, default=0, metavar="NUM",
                        help="Skip NUM tests")
    parser.add_argument("--faked-fio", action='store_true',
                        default=False, help="Emulate fio with 0 test time")
    parser.add_argument("--params", nargs="*", metavar="PARAM=VAL",
                        default=[],
                        help="Provide set of pairs PARAM=VAL to" +
                             "format into job description")
    parser.add_argument("jobfile")
    return parser.parse_args(argv)


def main(argv):
    argv_obj = parse_args(argv)

    if argv_obj.jobfile == '-':
        job_cfg = read_config(sys.stdin)
    else:
        job_cfg = open(argv_obj.jobfile).read()

    if argv_obj.output == '-':
        out_fd = sys.stdout
    else:
        out_fd = open(argv_obj.output, "w")

    params = {}
    for param_val in argv_obj.params:
        assert '=' in param_val
        name, val = param_val.split("=", 1)
        params[name] = val

    if argv_obj.estimate:
        print sec_to_str(estimate_cfg(job_cfg, params))
        return 0

    if argv_obj.num_tests or argv_obj.compile:
        bconf = list(parse_fio_config_full(job_cfg, params))
        bconf = bconf[argv_obj.skip_tests:]

        if argv_obj.compile:
            out_fd.write(format_fio_config(bconf))
            out_fd.write("\n")

        if argv_obj.num_tests:
            print len(bconf)

        return 0

    if argv_obj.start_at is not None:
        ctime = time.time()
        if argv_obj.start_at >= ctime:
            time.sleep(ctime - argv_obj.start_at)

    def raw_res_func(test_num, data):
        pref = "========= RAW_RESULTS({0}) =========\n".format(test_num)
        out_fd.write(pref)
        out_fd.write(json.dumps(data))
        out_fd.write("\n========= END OF RAW_RESULTS =========\n")
        out_fd.flush()

    rrfunc = raw_res_func if argv_obj.show_raw_results else None

    stime = time.time()
    job_res, num_tests = run_benchmark(argv_obj.type,
                                       job_cfg,
                                       params,
                                       argv_obj.runcycle,
                                       rrfunc,
                                       argv_obj.skip_tests,
                                       argv_obj.faked_fio)
    etime = time.time()

    res = {'__meta__': {'raw_cfg': job_cfg, 'params': params}, 'res': job_res}

    oformat = 'json' if argv_obj.json else 'eval'
    out_fd.write("\nRun {0} tests in {1} seconds\n".format(num_tests,
                                                           int(etime - stime)))
    out_fd.write("========= RESULTS(format={0}) =========\n".format(oformat))
    if argv_obj.json:
        out_fd.write(json.dumps(res))
    else:
        out_fd.write(pprint.pformat(res) + "\n")
    out_fd.write("\n========= END OF RESULTS =========\n".format(oformat))

    return 0


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
