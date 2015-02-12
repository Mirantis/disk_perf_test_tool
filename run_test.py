import os
import sys
import json
import time
import pprint
import os.path
import argparse

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
                        required=True,
                        help="cmd line options for io.py")

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


def main(argv):
    opts = parse_args(argv)

    if not opts.extra_logs:
        def nolog(x):
            pass

        setlogger(nolog)

    script_args = [opt.strip()
                   for opt in opts.io_opts.split(" ")
                   if opt.strip() != ""]

    if opts.runner == "rally":
        runner = rally_runner.get_rally_runner(
            files_dir=os.path.dirname(io_scenario.__file__),
            rally_extra_opts=opts.runner_opts.split(" "),
            max_preparation_time=opts.max_preparation_time,
            keep_temp_files=opts.keep_temp_files)
        res = run_io_test(opts.tool_type,
                          script_args,
                          runner,
                          opts.keep_temp_files)
    elif opts.runner == "ssh":
        user, key_file = opts.runner_opts.split(" ", 1)

        latest_start_time = opts.max_preparation_time + time.time()

        nova = nova_connect()

        # nova, amount, keypair_name, img_name,
        # flavor_name, vol_sz=None, network_zone_name=None,
        # flt_ip_pool=None, name_templ='ceph-test-{}',
        # scheduler_hints=None

        try:
            ips = [i[0] for i in create_vms_mt(nova, 3,
                                               keypair_name='ceph',
                                               img_name='ubuntu',
                                               flavor_name='ceph.512',
                                               network_zone_name='net04',
                                               flt_ip_pool='net04_ext')]

            uris = ["{0}@{1}::{2}".format(user, ip, key_file) for ip in ips]

            runner = ssh_runner.get_ssh_runner(uris,
                                               latest_start_time,
                                               opts.keep_temp_files)
            res = run_io_test(opts.tool_type,
                              script_args,
                              runner,
                              opts.keep_temp_files)
        finally:
            clear_all(nova)

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
