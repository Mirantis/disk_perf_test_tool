import sys
import time
import json
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
    for i in range(repeat):
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
                iterable_values.append(i.strip() for i in content.split(','))
            else:
                processed_vals[val_name] = val.format(**params)

        if iterable_values == []:
            params['UNIQ'] = 'UN{0}'.format(counter[0])
            counter[0] += 1
            params['TEST_SUMM'] = get_test_summary(processed_vals)
            yield name.format(**params), processed_vals
        else:
            for it_vals in itertools.product(*iterable_values):
                processed_vals.update(dict(zip(iterable_names, it_vals)))
                params['UNIQ'] = 'UN{0}'.format(counter[0])
                counter[0] += 1
                params['TEST_SUMM'] = get_test_summary(processed_vals)
                yield name.format(**params), processed_vals


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


def add_job_results(jname, job_output, jconfig, res):
    if job_output['write']['iops'] != 0:
        raw_result = job_output['write']
    else:
        raw_result = job_output['read']

    if jname not in res:
        j_res = {}
        j_res["action"] = jconfig["rw"]
        j_res["direct_io"] = jconfig.get("direct", "0") == "1"
        j_res["sync"] = jconfig.get("sync", "0") == "1"
        j_res["concurence"] = int(jconfig.get("numjobs", 1))
        j_res["size"] = jconfig["size"]
        j_res["jobname"] = job_output["jobname"]
        j_res["timings"] = (jconfig.get("runtime"),
                            jconfig.get("ramp_time"))
    else:
        j_res = res[jname]
        assert j_res["action"] == jconfig["rw"]

        assert j_res["direct_io"] == \
            (jconfig.get("direct", "0") == "1")

        assert j_res["sync"] == (jconfig.get("sync", "0") == "1")
        assert j_res["concurence"] == int(jconfig.get("numjobs", 1))
        assert j_res["size"] == jconfig["size"]
        assert j_res["jobname"] == job_output["jobname"]
        assert j_res["timings"] == (jconfig.get("runtime"),
                                    jconfig.get("ramp_time"))

    def j_app(name, x):
        j_res.setdefault(name, []).append(x)

    # 'bw_dev bw_mean bw_max bw_min'.split()
    j_app("bw_mean", raw_result["bw_mean"])
    j_app("iops", raw_result["iops"])
    j_app("lat", raw_result["lat"]["mean"])
    j_app("clat", raw_result["clat"]["mean"])
    j_app("slat", raw_result["slat"]["mean"])

    res[jname] = j_res


def run_fio(benchmark_config,
            params,
            runcycle=None,
            raw_results_func=None,
            skip_tests=0):

    whole_conf = list(parse_fio_config_full(benchmark_config, params))
    whole_conf = whole_conf[skip_tests:]
    res = {}
    curr_test_num = skip_tests
    execited_tests = 0
    try:
        for bconf in next_test_portion(whole_conf, runcycle):
            res_cfg_it = do_run_fio(bconf)
            res_cfg_it = enumerate(res_cfg_it, curr_test_num)

            for curr_test_num, (job_output, (jname, jconfig)) in res_cfg_it:
                execited_tests += 1
                if raw_results_func is not None:
                    raw_results_func(curr_test_num,
                                     [job_output, jname, jconfig])

                assert jname == job_output["jobname"]

                if jname.startswith('_'):
                    continue

                add_job_results(jname, job_output, jconfig, res)

    except (SystemExit, KeyboardInterrupt):
        pass

    except Exception:
        traceback.print_exc()

    return res, execited_tests


def run_benchmark(binary_tp, *argv, **kwargs):
    if 'fio' == binary_tp:
        return run_fio(*argv, **kwargs)
    raise ValueError("Unknown behcnmark {0}".format(binary_tp))


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
    parser.add_argument("--params", nargs="*", metavar="PARAM=VAL",
                        default=[],
                        help="Provide set of pairs PARAM=VAL to" +
                             "format into job description")
    parser.add_argument("jobfile")
    return parser.parse_args(argv)


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

    if argv_obj.num_tests or argv_obj.compile or argv_obj.estimate:
        bconf = list(parse_fio_config_full(job_cfg, params))
        bconf = bconf[argv_obj.skip_tests:]

        if argv_obj.compile:
            out_fd.write(format_fio_config(bconf))
            out_fd.write("\n")

        if argv_obj.num_tests:
            print len(bconf)

        if argv_obj.estimate:
            seconds = calculate_execution_time(bconf)

            h = seconds // 3600
            m = (seconds % 3600) // 60
            s = seconds % 60

            print "{0}:{1}:{2}".format(h, m, s)
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
                                       argv_obj.skip_tests)
    etime = time.time()

    res = {'__meta__': {'raw_cfg': job_cfg}, 'res': job_res}

    oformat = 'json' if argv_obj.json else 'eval'
    out_fd.write("\nRun {} tests in {} seconds\n".format(num_tests,
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