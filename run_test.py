import os
import sys
import json
import time
import pprint
import os.path
import argparse
import traceback

import io_scenario
from itest import IOPerfTest
from log import setlogger

import ssh_runner
import rally_runner

from starts_vms import nova_connect, create_vms_mt, clear_all


def run_io_test(tool,
                script_args,
                test_runner,
                keep_temp_files=False):

    files_dir = os.path.dirname(io_scenario.__file__)

    path = 'iozone' if 'iozone' == tool else 'fio'
    src_testtool_path = os.path.join(files_dir, path)

    obj = IOPerfTest(script_args,
                     src_testtool_path,
                     None,
                     keep_temp_files)

    return test_runner(obj)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run disk io performance test")

    parser.add_argument("tool_type", help="test tool type",
                        choices=['iozone', 'fio'])

    parser.add_argument("-l", dest='extra_logs',
                        action='store_true', default=False,
                        help="print some extra log info")

    parser.add_argument("-o", "--io-opts", dest='io_opts',
                        help="cmd line options for io.py")

    parser.add_argument("-f", "--io-opts-file", dest='io_opts_file',
                        type=argparse.FileType('r'), default=None,
                        help="file with cmd line options for io.py")

    parser.add_argument("-t", "--test-directory", help="directory with test",
                        dest="test_directory", required=True)

    parser.add_argument("--max-preparation-time", default=300,
                        type=int, dest="max_preparation_time")

    parser.add_argument("-k", "--keep", default=False,
                        help="keep temporary files",
                        dest="keep_temp_files", action='store_true')

    parser.add_argument("--runner", required=True,
                        choices=["ssh", "rally"],
                        help="runner type")

    parser.add_argument("--runner-extra-opts", default="",
                        dest="runner_opts", help="runner extra options")

    return parser.parse_args(argv)


def print_measurements_stat(res):
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


def get_io_opts(io_opts_file, io_opts):
    if io_opts_file is not None and io_opts is not None:
        print "Options --io-opts-file and --io-opts can't be " + \
            "provided same time"
        exit(1)

    if io_opts_file is None and io_opts is None:
        print "Either --io-opts-file or --io-opts should " + \
            "be provided"
        exit(1)

    if io_opts_file is not None:
        io_opts = []

        opt_lines = io_opts_file.readlines()
        opt_lines = [i for i in opt_lines if i != "" and not i.startswith("#")]

        for opt_line in opt_lines:
            io_opts.append([opt.strip()
                           for opt in opt_line.split(" ")
                           if opt.strip() != ""])
    else:
        io_opts = [[opt.strip()
                   for opt in io_opts.split(" ")
                   if opt.strip() != ""]]

    if len(io_opts) == 0:
        print "Can't found parameters for io. Check" + \
            "--io-opts-file or --io-opts options"
        exit(1)

    return io_opts


def main(argv):
    opts = parse_args(argv)

    if not opts.extra_logs:
        def nolog(x):
            pass

        setlogger(nolog)

    io_opts = get_io_opts(opts.io_opts_file, opts.io_opts)

    if opts.runner == "rally":
        for script_args in io_opts:
            runner = rally_runner.get_rally_runner(
                files_dir=os.path.dirname(io_scenario.__file__),
                rally_extra_opts=opts.runner_opts.split(" "),
                max_preparation_time=opts.max_preparation_time,
                keep_temp_files=opts.keep_temp_files)

            res = run_io_test(opts.tool_type,
                              script_args,
                              runner,
                              opts.keep_temp_files)

            print "=" * 80
            print pprint.pformat(res)
            print "=" * 80

            print_measurements_stat(res)

    elif opts.runner == "ssh":
        create_vms_opts = {}
        for opt in opts.runner_opts.split(","):
            name, val = opt.split("=", 1)
            create_vms_opts[name] = val

        user = create_vms_opts.pop("user")
        key_file = create_vms_opts.pop("key_file")
        aff_group = create_vms_opts.pop("aff_group", None)
        raw_count = create_vms_opts.pop("count", "x1")

        if raw_count.startswith("x"):
            raise NotImplementedError("xXXXX count not implemented yet")
        else:
            count = int(raw_count)

        if aff_group is not None:
            scheduler_hints = {'group': aff_group}
        else:
            scheduler_hints = None

        create_vms_opts['scheduler_hints'] = scheduler_hints

        latest_start_time = opts.max_preparation_time + time.time()

        nova = nova_connect()

        # nova, amount, keypair_name, img_name,
        # flavor_name, vol_sz=None, network_zone_name=None,
        # flt_ip_pool=None, name_templ='ceph-test-{}',
        # scheduler_hints=None

        try:
            ips = [i[0] for i in create_vms_mt(nova, count, **create_vms_opts)]

            uris = ["{0}@{1}::{2}".format(user, ip, key_file) for ip in ips]

            for script_args in io_opts:
                runner = ssh_runner.get_ssh_runner(uris,
                                                   latest_start_time,
                                                   opts.keep_temp_files)
                res = run_io_test(opts.tool_type,
                                  script_args,
                                  runner,
                                  opts.keep_temp_files)
                print "=" * 80
                print pprint.pformat(res)
                print "=" * 80

                print_measurements_stat(res)
        except:
            traceback.print_exc()
        finally:
            clear_all(nova)

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
