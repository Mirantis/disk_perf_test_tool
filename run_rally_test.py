import os
import sys
import json
import pprint
import os.path
import argparse
import datetime
import multiprocessing

import io_scenario
import rally_runner
from itest import IOPerfTest


def log(x):
    dt_str = datetime.datetime.now().strftime("%H:%M:%S")
    pref = dt_str + " " + str(os.getpid()) + " >>>> "
    sys.stderr.write(pref + x.replace("\n", "\n" + pref) + "\n")


def run_io_test(tool,
                script_args,
                test_runner,
                keep_temp_files=False):

    files_dir = os.path.dirname(io_scenario.__file__)

    path = 'iozone' if 'iozone' == tool else 'fio'
    src_testtool_path = os.path.join(files_dir, path)

    result_queue = multiprocessing.Queue()

    obj = IOPerfTest(script_args,
                     src_testtool_path,
                     result_queue.put,
                     keep_temp_files)

    test_runner(obj)

    test_result = []
    while not result_queue.empty():
        test_result.append(result_queue.get())

    return test_result


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run rally disk io performance test")
    parser.add_argument("tool_type", help="test tool type",
                        choices=['iozone', 'fio'])
    parser.add_argument("-l", dest='extra_logs',
                        action='store_true', default=False,
                        help="print some extra log info")
    parser.add_argument("-o", "--io-opts", dest='io_opts',
                        required=True,
                        help="cmd line options for io.py")
    parser.add_argument("-t", "--test-directory", help="directory with test",
                        dest="test_directory", required=True)
    parser.add_argument("--max-preparation-time", default=300,
                        type=int, dest="max_preparation_time")
    parser.add_argument("-k", "--keep", default=False,
                        help="keep temporary files",
                        dest="keep_temp_files", action='store_true')
    parser.add_argument("--rally-extra-opts", dest="rally_extra_opts",
                        default="", help="rally extra options")

    return parser.parse_args(argv)


def main(argv):
    opts = parse_args(argv)

    if not opts.extra_logs:
        global log

        def nolog(x):
            pass

        log = nolog
    else:
        rally_runner.log = log

    script_args = [opt.strip()
                   for opt in opts.io_opts.split(" ")
                   if opt.strip() != ""]

    runner = rally_runner.get_rally_runner(
        files_dir=os.path.dirname(io_scenario.__file__),
        rally_extra_opts=opts.rally_extra_opts.split(" "),
        max_preparation_time=opts.max_preparation_time,
        keep_temp_files=opts.keep_temp_files)

    res = run_io_test(opts.tool_type,
                      script_args,
                      runner,
                      opts.keep_temp_files)

    print "=" * 80
    print pprint.pformat(res)
    print "=" * 80

    if len(res) != 0:
        bw_mean = 0.0
        for measurement in res:
            bw_mean += measurement["bw_mean"]

        bw_mean /= len(res)

        it = ((bw_mean - measurement["bw_mean"]) ** 2 for measurement in res)
        bw_dev = sum(it) ** 0.5

        meta = res[0]['__meta__']
        key = "{0} {1} {2}k".format(meta['action'],
                                    's' if meta['sync'] else 'a',
                                    meta['blocksize'])

        print
        print "====> " + json.dumps({key: (int(bw_mean), int(bw_dev))})
        print
        print "=" * 80

    return 0


ostack_prepare = """
glance image-create --name 'ubuntu' --disk-format qcow2
--container-format bare --is-public true --copy-from
https://cloud-images.ubuntu.com/trusty/current/trusty-server-cloudimg-amd64-disk1.img

nova flavor-create ceph.512 ceph.512 512 50 1
nova server-group-create --policy anti-affinity ceph
"""


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
