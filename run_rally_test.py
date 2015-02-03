import os
import re
import sys
import time
import yaml
import json
import pprint
import os.path
import argparse
import datetime
import warnings
import functools
import contextlib
import multiprocessing

from rally import exceptions
from rally.cmd import cliutils
from rally.cmd.main import categories
from rally.benchmark.scenarios.vm.utils import VMScenario

from ssh_copy_directory import put_dir_recursively, ssh_copy_file


def log(x):
    dt_str = datetime.datetime.now().strftime("%H:%M:%S")
    pref = dt_str + " " + str(os.getpid()) + " >>>> "
    sys.stderr.write(pref + x.replace("\n", "\n" + pref) + "\n")


def get_barrier(count):
    val = multiprocessing.Value('i', count)
    cond = multiprocessing.Condition()

    def closure(timeout):
        me_released = False
        with cond:
            val.value -= 1
            if val.value == 0:
                me_released = True
                cond.notify_all()
            else:
                cond.wait(timeout)
            return val.value == 0

        if me_released:
            log("Test begins!")

    return closure


# should actually use mock module for this,
# but don't wanna to add new dependency

@contextlib.contextmanager
def patch_VMScenario_run_command_over_ssh(paths,
                                          on_result_cb,
                                          barrier=None,
                                          latest_start_time=None):

    try:
        orig = VMScenario.run_action
    except AttributeError:
        # rally code was changed
        log("VMScenario class was changed and have no run_action"
            " method anymore. Update patch code.")
        raise exceptions.ScriptError("monkeypatch code fails on "
                                     "VMScenario.run_action")

    @functools.wraps(orig)
    def closure(self, ssh, *args, **kwargs):
        try:
            sftp = ssh._client.open_sftp()
        except AttributeError:
            # rally code was changed
            log("Prototype of VMScenario.run_command_over_ssh "
                "was changed. Update patch code.")
            raise exceptions.ScriptError("monkeypatch code fails on "
                                         "ssh._client.open_sftp()")
        try:
            for src, dst in paths.items():
                try:
                    if os.path.isfile(src):
                        ssh_copy_file(sftp, src, dst)
                    elif os.path.isdir(src):
                        put_dir_recursively(sftp, src, dst)
                    else:
                        templ = "Can't copy {0!r} - " + \
                                "it neither a file not a directory"
                        msg = templ.format(src)
                        log(msg)
                        raise exceptions.ScriptError(msg)
                except exceptions.ScriptError:
                    raise
                except Exception as exc:
                    tmpl = "Scp {0!r} => {1!r} failed - {2!r}"
                    msg = tmpl.format(src, dst, exc)
                    log(msg)
                    raise exceptions.ScriptError(msg)
        finally:
            sftp.close()

        log("Start io test")

        if barrier is not None:
            if latest_start_time is not None:
                timeout = latest_start_time - time.time()
            else:
                timeout = None

            if timeout is not None and timeout > 0:
                msg = "Ready and waiting on barrier. " + \
                      "Will wait at most {0} seconds"
                log(msg.format(int(timeout)))

                if not barrier(timeout):
                    log("Barrier timeouted")

        try:
            code, out, err = orig(self, ssh, *args, **kwargs)
        except Exception as exc:
            log("Rally raises exception {0}".format(exc.message))
            raise

        if 0 != code:
            templ = "Script returns error! code={0}\n {1}"
            log(templ.format(code, err.rstrip()))
        else:
            log("Test finished")

            try:
                for line in out.split("\n"):
                    if line.strip() != "":
                        result = json.loads(line)
                        on_result_cb(result)
            except Exception as err:
                log("Error during postprocessing results: {0!r}".format(err))

            # result = {"rally": 0, "srally": 1}
            result = {"rally": 0}
            out = json.dumps(result)

        return code, out, err

    VMScenario.run_action = closure

    try:
        yield
    finally:
        VMScenario.run_action = orig


def run_rally(rally_args):
    return cliutils.run(['rally'] + rally_args, categories)


def prepare_files(testtool_py_args_v, dst_testtool_path, files_dir):

    # we do need temporary named files
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        py_file = os.tmpnam()
        yaml_file = os.tmpnam()

    testtool_py_inp_path = os.path.join(files_dir, "io.py")
    py_src_cont = open(testtool_py_inp_path).read()
    args_repl_rr = r'INSERT_TOOL_ARGS\(sys\.argv.*?\)'
    py_dst_cont = re.sub(args_repl_rr, repr(testtool_py_args_v), py_src_cont)

    if py_dst_cont == args_repl_rr:
        templ = "Can't find replace marker in file {0}"
        msg = templ.format(testtool_py_inp_path)
        log(msg)
        raise ValueError(msg)

    yaml_src_cont = open(os.path.join(files_dir, "io.yaml")).read()
    task_params = yaml.load(yaml_src_cont)
    rcd_params = task_params['VMTasks.boot_runcommand_delete']
    rcd_params[0]['args']['script'] = py_file
    yaml_dst_cont = yaml.dump(task_params)

    open(py_file, "w").write(py_dst_cont)
    open(yaml_file, "w").write(yaml_dst_cont)

    return yaml_file, py_file


def run_test(tool, testtool_py_args_v, dst_testtool_path, files_dir,
             rally_extra_opts, max_preparation_time=300):

    path = 'iozone' if 'iozone' == tool else 'fio'
    testtool_local = os.path.join(files_dir, path)
    yaml_file, py_file = prepare_files(testtool_py_args_v,
                                       dst_testtool_path,
                                       files_dir)
    try:
        config = yaml.load(open(yaml_file).read())

        vm_sec = 'VMTasks.boot_runcommand_delete'
        concurrency = config[vm_sec][0]['runner']['concurrency']
        copy_files = {testtool_local: dst_testtool_path}

        result_queue = multiprocessing.Queue()
        results_cb = result_queue.put

        do_patch = patch_VMScenario_run_command_over_ssh

        barrier = get_barrier(concurrency)
        max_release_time = time.time() + max_preparation_time

        with do_patch(copy_files, results_cb, barrier, max_release_time):
            opts = ['task', 'start', yaml_file] + list(rally_extra_opts)
            log("Start rally with opts '{0}'".format(" ".join(opts)))
            run_rally(opts)

        rally_result = []
        while not result_queue.empty():
            rally_result.append(result_queue.get())

        return rally_result

    finally:
        os.unlink(yaml_file)
        os.unlink(py_file)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run rally disk io performance test")
    parser.add_argument("tool_type", help="test tool type",
                        choices=['iozone', 'fio'])
    parser.add_argument("-l", dest='extra_logs',
                        action='store_true', default=False,
                        help="print some extra log info")
    parser.add_argument("-o", "--io-opts", dest='io_opts',
                        nargs="*", default=[],
                        help="cmd line options for io.py")
    parser.add_argument("-t", "--test-directory", help="directory with test",
                        dest="test_directory", required=True)
    parser.add_argument("rally_extra_opts", nargs="*",
                        default=[], help="rally extra options")
    parser.add_argument("--max-preparation-time", default=300,
                        type=int, dest="max_preparation_time")

    return parser.parse_args(argv)


def main(argv):
    opts = parse_args(argv)
    dst_testtool_path = '/tmp/io_tool'

    if not opts.extra_logs:
        global log

        def nolog(x):
            pass

        log = nolog

    if opts.io_opts == []:
        testtool_py_args_v = []

        block_sizes = ["4k", "64k"]
        ops = ['randwrite']
        iodepths = ['8']
        syncs = [True]

        for block_size in block_sizes:
            for op in ops:
                for iodepth in iodepths:
                    for sync in syncs:
                        tt_argv = ['--type', opts.tool_type,
                                   '-a', op,
                                   '--iodepth', iodepth,
                                   '--blocksize', block_size,
                                   '--iosize', '20M']
                        if sync:
                            tt_argv.append('-s')
            testtool_py_args_v.append(tt_argv)
    else:
        testtool_py_args_v = [o.split(" ") for o in opts.io_opts]

    for io_argv_list in testtool_py_args_v:
        io_argv_list.extend(['--binary-path', dst_testtool_path])

    res = run_test(opts.tool_type,
                   testtool_py_args_v,
                   dst_testtool_path,
                   files_dir=opts.test_directory,
                   rally_extra_opts=opts.rally_extra_opts,
                   max_preparation_time=opts.max_preparation_time)

    print "Results = ",
    pprint.pprint(res)

    return 0

# ubuntu cloud image
# https://cloud-images.ubuntu.com/trusty/current/trusty-server-cloudimg-amd64-disk1.img

# glance image-create --name 'ubuntu' --disk-format qcow2
# --container-format bare --is-public true --copy-from
# https://cloud-images.ubuntu.com/trusty/current/trusty-server-cloudimg-amd64-disk1.img

if __name__ == '__main__':
    exit(main(sys.argv[1:]))
