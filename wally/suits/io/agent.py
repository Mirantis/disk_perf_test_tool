import sys
import time
import json
import copy
import select
import pprint
import argparse
import traceback
import subprocess
import itertools
from collections import OrderedDict


SECTION = 0
SETTING = 1


class FioJobSection(object):
    def __init__(self, name):
        self.name = name
        self.vals = OrderedDict()
        self.format_params = {}

    def copy(self):
        return copy.deepcopy(self)


def to_bytes(sz):
    sz = sz.lower()
    try:
        return int(sz)
    except ValueError:
        if sz[-1] == 'm':
            return (1024 ** 2) * int(sz[:-1])
        if sz[-1] == 'k':
            return 1024 * int(sz[:-1])
        if sz[-1] == 'g':
            return (1024 ** 3) * int(sz[:-1])
        raise


def fio_config_lexer(fio_cfg):
    for lineno, line in enumerate(fio_cfg.split("\n")):
        try:
            line = line.strip()

            if line.startswith("#") or line.startswith(";"):
                continue

            if line == "":
                continue

            if line.startswith('['):
                assert line.endswith(']'), "name should ends with ]"
                yield lineno, SECTION, line[1:-1], None
            elif '=' in line:
                opt_name, opt_val = line.split('=', 1)
                yield lineno, SETTING, opt_name.strip(), opt_val.strip()
            else:
                yield lineno, SETTING, line, None
        except Exception as exc:
            pref = "During parsing line number {0}\n".format(lineno)
            raise ValueError(pref + exc.message)


def fio_config_parse(lexer_iter, format_params):
    orig_format_params_keys = set(format_params)
    format_params = format_params.copy()
    in_defaults = False
    curr_section = None
    defaults = OrderedDict()

    for lineno, tp, name, val in lexer_iter:
        if tp == SECTION:
            if curr_section is not None:
                yield curr_section

            if name == 'defaults':
                in_defaults = True
                curr_section = None
            else:
                in_defaults = False
                curr_section = FioJobSection(name)
                curr_section.format_params = format_params.copy()
                curr_section.vals = defaults.copy()
        else:
            assert tp == SETTING
            if name == name.upper():
                msg = "Param not in default section in line " + str(lineno)
                assert in_defaults, msg
                if name not in orig_format_params_keys:
                    # don't make parse_value for PARAMS
                    # they would be parsed later
                    # or this would breakes arrays
                    format_params[name] = val
            elif in_defaults:
                defaults[name] = parse_value(val)
            else:
                msg = "data outside section, line " + str(lineno)
                assert curr_section is not None, msg
                curr_section.vals[name] = parse_value(val)

    if curr_section is not None:
        yield curr_section


def parse_value(val):
    if val is None:
        return None

    try:
        return int(val)
    except ValueError:
        pass

    try:
        return float(val)
    except ValueError:
        pass

    if val.startswith('{%'):
        assert val.endswith("%}")
        content = val[2:-2]
        vals = list(i.strip() for i in content.split(','))
        return map(parse_value, vals)
    return val


def process_repeats(sec_iter):

    for sec in sec_iter:
        if '*' in sec.name:
            msg = "Only one '*' allowed in section name"
            assert sec.name.count('*') == 1, msg

            name, count = sec.name.split("*")
            sec.name = name.strip()
            count = count.strip()

            try:
                count = int(count.strip().format(**sec.format_params))
            except KeyError:
                raise ValueError("No parameter {0} given".format(count[1:-1]))
            except ValueError:
                msg = "Parameter {0} nas non-int value {1!r}"
                raise ValueError(msg.format(count[1:-1],
                                 count.format(**sec.format_params)))

            yield sec

            if 'ramp_time' in sec.vals:
                sec = sec.copy()
                sec.vals['_ramp_time'] = sec.vals.pop('ramp_time')

            for _ in range(count - 1):
                yield sec.copy()
        else:
            yield sec


def process_cycles(sec_iter):
    # insert parametrized cycles
    sec_iter = try_format_params_into_section(sec_iter)

    for sec in sec_iter:

        cycles_var_names = []
        cycles_var_values = []

        for name, val in sec.vals.items():
            if isinstance(val, list):
                cycles_var_names.append(name)
                cycles_var_values.append(val)

        if len(cycles_var_names) == 0:
            yield sec
        else:
            for combination in itertools.product(*cycles_var_values):
                new_sec = sec.copy()
                new_sec.vals.update(zip(cycles_var_names, combination))
                yield new_sec


def try_format_params_into_section(sec_iter):
    for sec in sec_iter:
        params = sec.format_params
        for name, val in sec.vals.items():
            if isinstance(val, basestring):
                try:
                    sec.vals[name] = parse_value(val.format(**params))
                except:
                    pass

        yield sec


def format_params_into_section_finall(sec_iter, counter=[0]):
    group_report_err_msg = "Group reporting should be set if numjobs != 1"

    for sec in sec_iter:

        num_jobs = int(sec.vals.get('numjobs', '1'))
        if num_jobs != 1:
            assert 'group_reporting' in sec.vals, group_report_err_msg

        params = sec.format_params

        fsize = to_bytes(sec.vals['size'])
        params['PER_TH_OFFSET'] = fsize // num_jobs

        for name, val in sec.vals.items():
            if isinstance(val, basestring):
                sec.vals[name] = parse_value(val.format(**params))
            else:
                assert isinstance(val, (int, float)) or val is None

        params['UNIQ'] = 'UN{0}'.format(counter[0])
        counter[0] += 1
        params['TEST_SUMM'] = get_test_summary(sec.vals)
        sec.name = sec.name.format(**params)

        yield sec


def fio_config_to_str(sec_iter):
    res = ""

    for pos, sec in enumerate(sec_iter):
        if pos != 0:
            res += "\n"

        res += "[{0}]\n".format(sec.name)

        for name, val in sec.vals.items():
            if name.startswith('_'):
                continue

            if val is None:
                res += name + "\n"
            else:
                res += "{0}={1}\n".format(name, val)

    return res


def get_test_sync_mode(config):
    try:
        return config['sync_mode']
    except KeyError:
        pass

    is_sync = str(config.get("sync", "0")) == "1"
    is_direct = str(config.get("direct", "0")) == "1"

    if is_sync and is_direct:
        return 'x'
    elif is_sync:
        return 's'
    elif is_direct:
        return 'd'
    else:
        return 'a'


def get_test_summary(params):
    rw = {"randread": "rr",
          "randwrite": "rw",
          "read": "sr",
          "write": "sw"}[params["rw"]]

    sync_mode = get_test_sync_mode(params)
    th_count = params.get('numjobs')

    if th_count is None:
        th_count = params.get('concurence', 1)

    return "{0}{1}{2}th{3}".format(rw,
                                   sync_mode,
                                   params['blocksize'],
                                   th_count)


def calculate_execution_time(sec_iter):
    time = 0
    for sec in sec_iter:
        time += sec.vals.get('ramp_time', 0)
        time += sec.vals.get('runtime', 0)
    return time


def slice_config(sec_iter, runcycle=None, max_jobs=1000):
    jcount = 0
    runtime = 0
    curr_slice = []

    for pos, sec in enumerate(sec_iter):

        jc = sec.vals.get('numjobs', 1)
        msg = "numjobs should be integer, not {0!r}".format(jc)
        assert isinstance(jc, int), msg

        curr_task_time = calculate_execution_time([sec])

        if jc > max_jobs:
            err_templ = "Can't process job {0!r} - too large numjobs"
            raise ValueError(err_templ.format(sec.name))

        if runcycle is not None and len(curr_slice) != 0:
            rc_ok = curr_task_time + runtime <= runcycle
        else:
            rc_ok = True

        if jc + jcount <= max_jobs and rc_ok:
            runtime += curr_task_time
            jcount += jc
            curr_slice.append(sec)
            continue

        assert len(curr_slice) != 0
        yield curr_slice

        if '_ramp_time' in sec.vals:
            sec.vals['ramp_time'] = sec.vals.pop('_ramp_time')
            curr_task_time = calculate_execution_time([sec])

        runtime = curr_task_time
        jcount = jc
        curr_slice = [sec]

    if curr_slice != []:
        yield curr_slice


def parse_all_in_1(source, test_params):
    lexer_it = fio_config_lexer(source)
    sec_it = fio_config_parse(lexer_it, test_params)
    sec_it = process_cycles(sec_it)
    sec_it = process_repeats(sec_it)
    return format_params_into_section_finall(sec_it)


def parse_and_slice_all_in_1(source, test_params, **slice_params):
    sec_it = parse_all_in_1(source, test_params)
    return slice_config(sec_it, **slice_params)


def compile_all_in_1(source, test_params, **slice_params):
    slices_it = parse_and_slice_all_in_1(source, test_params, **slice_params)
    for slices in slices_it:
        yield fio_config_to_str(slices)


def do_run_fio(config_slice):
    benchmark_config = fio_config_to_str(config_slice)
    cmd = ["fio", "--output-format=json", "--alloc-size=262144", "-"]
    p = subprocess.Popen(cmd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    # set timeout
    raw_out, raw_err = p.communicate(benchmark_config)

    # HACK
    raw_out = "{" + raw_out.split('{', 1)[1]

    if 0 != p.returncode:
        msg = "Fio failed with code: {0}\nOutput={1}"
        raise OSError(msg.format(p.returncode, raw_err))

    try:
        parsed_out = json.loads(raw_out)["jobs"]
    except KeyError:
        msg = "Can't parse fio output {0!r}: no 'jobs' found"
        raw_out = raw_out[:100]
        raise ValueError(msg.format(raw_out))

    except Exception as exc:
        msg = "Can't parse fio output: {0!r}\nError: {1}"
        raw_out = raw_out[:100]
        raise ValueError(msg.format(raw_out, exc.message))

    return zip(parsed_out, config_slice)


def add_job_results(section, job_output, res):
    if job_output['write']['iops'] != 0:
        raw_result = job_output['write']
    else:
        raw_result = job_output['read']

    vals = section.vals
    if section.name not in res:
        j_res = {}
        j_res["rw"] = vals["rw"]
        j_res["sync_mode"] = get_test_sync_mode(vals)
        j_res["concurence"] = int(vals.get("numjobs", 1))
        j_res["blocksize"] = vals["blocksize"]
        j_res["jobname"] = job_output["jobname"]
        j_res["timings"] = [int(vals.get("runtime", 0)),
                            int(vals.get("ramp_time", 0))]
    else:
        j_res = res[section.name]
        assert j_res["rw"] == vals["rw"]
        assert j_res["rw"] == vals["rw"]
        assert j_res["sync_mode"] == get_test_sync_mode(vals)
        assert j_res["concurence"] == int(vals.get("numjobs", 1))
        assert j_res["blocksize"] == vals["blocksize"]
        assert j_res["jobname"] == job_output["jobname"]

        # ramp part is skipped for all tests, except first
        # assert j_res["timings"] == (vals.get("runtime"),
        #                             vals.get("ramp_time"))

    def j_app(name, x):
        j_res.setdefault(name, []).append(x)

    j_app("bw", raw_result["bw"])
    j_app("iops", raw_result["iops"])
    j_app("lat", raw_result["lat"]["mean"])
    j_app("clat", raw_result["clat"]["mean"])
    j_app("slat", raw_result["slat"]["mean"])

    res[section.name] = j_res


def run_fio(sliced_it, raw_results_func=None):
    sliced_list = list(sliced_it)
    ok = True

    try:
        curr_test_num = 0
        executed_tests = 0
        result = {}

        for i, test_slice in enumerate(sliced_list):
            res_cfg_it = do_run_fio(test_slice)
            res_cfg_it = enumerate(res_cfg_it, curr_test_num)

            for curr_test_num, (job_output, section) in res_cfg_it:
                executed_tests += 1

                if raw_results_func is not None:
                    raw_results_func(executed_tests,
                                     [job_output, section])

                msg = "{0} != {1}".format(section.name, job_output["jobname"])
                assert section.name == job_output["jobname"], msg

                if section.name.startswith('_'):
                    continue

                add_job_results(section, job_output, result)

            curr_test_num += 1
            msg_template = "Done {0} tests from {1}. ETA: {2}"

            rest = sliced_list[i:]
            time_eta = sum(map(calculate_execution_time, rest))
            test_left = sum(map(len, rest))
            print msg_template.format(curr_test_num,
                                      test_left,
                                      sec_to_str(time_eta))

    except (SystemExit, KeyboardInterrupt):
        raise

    except Exception:
        print "=========== ERROR ============="
        traceback.print_exc()
        print "======== END OF ERROR ========="
        ok = False

    return result, executed_tests, ok


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

    slice_params = {
        'runcycle': argv_obj.runcycle,
    }

    sliced_it = parse_and_slice_all_in_1(job_cfg, params, **slice_params)

    if argv_obj.estimate:
        it = map(calculate_execution_time, sliced_it)
        print sec_to_str(sum(it))
        return 0

    if argv_obj.num_tests or argv_obj.compile:
        if argv_obj.compile:
            for test_slice in sliced_it:
                out_fd.write(fio_config_to_str(test_slice))
                out_fd.write("\n#" + "-" * 70 + "\n\n")

        if argv_obj.num_tests:
            print len(list(sliced_it))

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
    job_res, num_tests, ok = run_benchmark(argv_obj.type, sliced_it, rrfunc)
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
    out_fd.write("\n========= END OF RESULTS =========\n")

    return 0 if ok else 1


def fake_main(x):
    import yaml
    time.sleep(60)
    out_fd = sys.stdout
    fname = "/tmp/perf_tests/metempirical_alisha/raw_results.yaml"
    res = yaml.load(open(fname).read())[0][1]
    out_fd.write("========= RESULTS(format=json) =========\n")
    out_fd.write(json.dumps(res))
    out_fd.write("\n========= END OF RESULTS =========\n")
    return 0


if __name__ == '__main__':
    # exit(fake_main(sys.argv[1:]))
    exit(main(sys.argv[1:]))
