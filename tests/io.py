import sys
import time
import json
import select
import pprint
import argparse
import subprocess
from StringIO import StringIO
from ConfigParser import RawConfigParser


def run_fio(benchmark_config):
    cmd = ["fio", "--output-format=json", "-"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    raw_out, _ = p.communicate(benchmark_config)
    job_output = json.loads(raw_out)["jobs"][0]

    if job_output['write']['iops'] != 0:
        raw_result = job_output['write']
    else:
        raw_result = job_output['read']

    res = {}

    # 'bw_dev bw_mean bw_max bw_min'.split()
    for field in ["bw_mean", "iops"]:
        res[field] = raw_result[field]

    res["lat"] = raw_result["lat"]["mean"]
    res["clat"] = raw_result["clat"]["mean"]
    res["slat"] = raw_result["slat"]["mean"]
    res["util"] = json.loads(raw_out)["disk_util"][0]

    res["util"] = dict((str(k), v) for k, v in res["util"].items())

    return res


def run_benchmark(binary_tp, *argv, **kwargs):
    if 'fio' == binary_tp:
        return run_fio(*argv, **kwargs)
    raise ValueError("Unknown behcnmark {0}".format(binary_tp))


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run fio' and return result")
    parser.add_argument(
        "--type", metavar="BINARY_TYPE",
        choices=['fio'], required=True)
    parser.add_argument("--start-at", metavar="START_TIME", type=int)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("jobfile")
    return parser.parse_args(argv)


def main(argv):
    argv_obj = parse_args(argv)
    if argv_obj.jobfile == '-':
        job_cfg = ""
        dtime = 10
        while True:
            r, w, x = select.select([sys.stdin], [], [], dtime)
            if len(r) == 0:
                raise IOError("No config provided")
            char = sys.stdin.read(1)
            if '' == char:
                break
            job_cfg += char
            dtime = 1
    else:
        job_cfg = open(argv_obj.jobfile).read()

    rcp = RawConfigParser()
    rcp.readfp(StringIO(job_cfg))
    assert len(rcp.sections()) == 1

    if argv_obj.start_at is not None:
        ctime = time.time()
        if argv_obj.start_at >= ctime:
            time.sleep(ctime - argv_obj.start_at)

    res = run_benchmark(argv_obj.type, job_cfg)
    res['__meta__'] = dict(rcp.items(rcp.sections()[0]))
    res['__meta__']['raw'] = job_cfg

    if argv_obj.json:
        sys.stdout.write(json.dumps(res))
    else:
        sys.stdout.write(pprint.pformat(res))
        sys.stdout.write("\n")
    return 0

if __name__ == '__main__':
    exit(main(sys.argv[1:]))
