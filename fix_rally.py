import os
import re
import sys
import time
import yaml
import json
import os.path
import datetime
import warnings
import functools
import contextlib
import multiprocessing

from rally import exceptions
from rally.cmd import cliutils
from rally.cmd.main import categories
from rally.benchmark.scenarios.vm.utils import VMScenario


def log(x):
    dt_str = datetime.datetime.now().strftime("%H:%M:%S")
    pref = dt_str + " " + str(os.getpid()) + " >>>> "
    sys.stderr.write(pref + x.replace("\n", "\n" + pref) + "\n")


def get_barrier(count):
    val = multiprocessing.Value('i', count)
    cond = multiprocessing.Condition()

    def closure(timeout):
        with cond:
            val.value -= 1

            log("barrier value == {0}".format(val.value))

            if val.value == 0:
                cond.notify_all()
            else:
                cond.wait(timeout)
            return val.value == 0

    return closure


MAX_WAIT_TOUT = 60


@contextlib.contextmanager
def patch_VMScenario_run_command_over_ssh(paths,
                                          add_meta_cb,
                                          barrier=None,
                                          latest_start_time=None):

    orig = VMScenario.run_command_over_ssh

    @functools.wraps(orig)
    def closure(self, ssh, *args, **kwargs):
        try:
            sftp = ssh._client.open_sftp()
        except AttributeError:
            # rally code was changed
            log("Prototype of VMScenario.run_command_over_ssh "
                "was changed. Update patch code.")
            raise exceptions.ScriptError("monkeypath code fails on "
                                         "ssh._client.open_sftp()")

        for src, dst in paths.items():
            try:
                sftp.put(src, dst)
            except Exception as exc:
                tmpl = "Scp {0!r} => {1!r} failed - {2!r}"
                msg = tmpl.format(src, dst, exc)
                log(msg)
                raise exceptions.ScriptError(
                    "monkeypath code fails on " + msg)

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
                try:
                    result = json.loads(out)
                except:
                    pass
                else:
                    if '__meta__' in result:
                        add_meta_cb(result.pop('__meta__'))
                    out = json.dumps(result)
            except Exception as err:
                log("Error during postprocessing results: {0!r}".format(err))

        return code, out, err

    VMScenario.run_command_over_ssh = closure

    try:
        yield
    finally:
        VMScenario.run_command_over_ssh = orig


def run_rally(rally_args):
    return cliutils.run(['rally'] + rally_args, categories)


def prepare_files(iozone_py_argv, dst_iozone_path, files_dir):

    # we do need temporary named files
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        py_file = os.tmpnam()
        yaml_file = os.tmpnam()

    iozone_py_inp_path = os.path.join(files_dir, "iozone.py")
    py_src_cont = open(iozone_py_inp_path).read()
    args_repl_rr = r'INSERT_IOZONE_ARGS\(sys\.argv.*?\)'
    py_dst_cont = re.sub(args_repl_rr, repr(iozone_py_argv), py_src_cont)

    if py_dst_cont == args_repl_rr:
        log("Can't find replace marker in file {0}".format(iozone_py_inp_path))
        exit(1)

    yaml_src_cont = open(os.path.join(files_dir, "iozone.yaml")).read()
    task_params = yaml.load(yaml_src_cont)
    rcd_params = task_params['VMTasks.boot_runcommand_delete']
    rcd_params[0]['args']['script'] = py_file
    yaml_dst_cont = yaml.dump(task_params)

    open(py_file, "w").write(py_dst_cont)
    open(yaml_file, "w").write(yaml_dst_cont)

    return yaml_file, py_file


def run_test(iozone_py_argv, dst_iozone_path, files_dir):
    iozone_local = os.path.join(files_dir, 'iozone')

    yaml_file, py_file = prepare_files(iozone_py_argv,
                                       dst_iozone_path,
                                       files_dir)

    config = yaml.load(open(yaml_file).read())

    vm_sec = 'VMTasks.boot_runcommand_delete'
    concurrency = config[vm_sec][0]['runner']['concurrency']

    max_preparation_time = 300

    try:
        copy_files = {iozone_local: dst_iozone_path}

        result_queue = multiprocessing.Queue()
        cb = result_queue.put

        do_patch = patch_VMScenario_run_command_over_ssh

        barrier = get_barrier(concurrency)
        max_release_time = time.time() + max_preparation_time

        with do_patch(copy_files, cb, barrier, max_release_time):
            log("Start rally with 'task start {0}'".format(yaml_file))
            rally_result = run_rally(['task', 'start', yaml_file])

        # while not result_queue.empty():
        #     log("meta = {0!r}\n".format(result_queue.get()))

        return rally_result

    finally:
        os.unlink(yaml_file)
        os.unlink(py_file)
    # store and process meta and results


def main(argv):
    files_dir = '.'
    dst_iozone_path = '/tmp/iozone'
    iozone_py_argv = ['-a', 'randwrite',
                      '--iodepth', '2',
                      '--blocksize', '4k',
                      '--iosize', '20M',
                      '--iozone-path', dst_iozone_path,
                      '-d']
    run_test(iozone_py_argv, dst_iozone_path, files_dir)

# ubuntu cloud image
# https://cloud-images.ubuntu.com/trusty/current/trusty-server-cloudimg-amd64-disk1.img

# glance image-create --name 'ubuntu' --disk-format qcow2
# --container-format bare --is-public true --copy-from
# https://cloud-images.ubuntu.com/trusty/current/trusty-server-cloudimg-amd64-disk1.img

if __name__ == '__main__':
    exit(main(sys.argv))
