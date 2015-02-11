import os
import json
import time
import yaml
import warnings
import functools
import contextlib

from rally import exceptions
from rally.cmd import cliutils
from rally.cmd.main import categories
from rally.benchmark.scenarios.vm.utils import VMScenario
from rally.benchmark.scenarios.vm.vmtasks import VMTasks

import itest
from utils import get_barrier


def log(x):
    pass


@contextlib.contextmanager
def patch_VMTasks_boot_runcommand_delete():

    try:
        orig = VMTasks.boot_runcommand_delete
    except AttributeError:
        # rally code was changed
        log("VMTasks class was changed and have no boot_runcommand_delete"
            " method anymore. Update patch code.")
        raise exceptions.ScriptError("monkeypatch code fails on "
                                     "VMTasks.boot_runcommand_delete")

    @functools.wraps(orig)
    def new_boot_runcommand_delete(self, *args, **kwargs):
        if 'rally_affinity_group' in os.environ:
            group_id = os.environ['rally_affinity_group']
            kwargs['scheduler_hints'] = {'group': group_id}
        return orig(self, *args, **kwargs)

    VMTasks.boot_runcommand_delete = new_boot_runcommand_delete

    try:
        yield
    finally:
        VMTasks.boot_runcommand_delete = orig


# should actually use mock module for this,
# but don't wanna to add new dependency
@contextlib.contextmanager
def patch_VMScenario_run_command_over_ssh(test_obj,
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
            ssh._client.open_sftp
        except AttributeError:
            # rally code was changed
            log("Prototype of VMScenario.run_command_over_ssh "
                "was changed. Update patch code.")
            raise exceptions.ScriptError("monkeypatch code fails on "
                                         "ssh._client.open_sftp()")

        test_iter = itest.run_test_iter(test_obj, ssh)

        next(test_iter)

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
            code, out, err = next(test_iter)
        except Exception as exc:
            log("Rally raises exception {0}".format(exc.message))
            raise

        if 0 != code:
            templ = "Script returns error! code={0}\n {1}"
            log(templ.format(code, err.rstrip()))
        else:
            log("Test finished")

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


def prepare_files(dst_testtool_path, files_dir):

    # we do need temporary named files
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yaml_file = os.tmpnam()

    yaml_src_cont = open(os.path.join(files_dir, "io.yaml")).read()
    task_params = yaml.load(yaml_src_cont)
    rcd_params = task_params['VMTasks.boot_runcommand_delete']
    rcd_params[0]['args']['script'] = os.path.join(files_dir, "io.py")
    yaml_dst_cont = yaml.dump(task_params)

    open(yaml_file, "w").write(yaml_dst_cont)

    return yaml_file


def run_tests_using_rally(obj,
                          files_dir,
                          testtool_py_args,
                          dst_testtool_path,
                          max_preparation_time,
                          rally_extra_opts,
                          keep_temp_files):

    yaml_file, py_file = prepare_files(testtool_py_args,
                                       dst_testtool_path,
                                       files_dir)

    try:
        do_patch1 = patch_VMScenario_run_command_over_ssh
        config = yaml.load(open(yaml_file).read())

        vm_sec = 'VMTasks.boot_runcommand_delete'
        concurrency = config[vm_sec][0]['runner']['concurrency']

        barrier = get_barrier(concurrency)
        max_release_time = time.time() + max_preparation_time

        with patch_VMTasks_boot_runcommand_delete():
            with do_patch1(obj, barrier, max_release_time):
                opts = ['task', 'start', yaml_file] + list(rally_extra_opts)
                log("Start rally with opts '{0}'".format(" ".join(opts)))
                run_rally(opts)
    finally:
        if not keep_temp_files:
            os.unlink(yaml_file)
            os.unlink(py_file)


def get_rally_runner(files_dir,
                     max_preparation_time,
                     rally_extra_opts,
                     keep_temp_files):

    def closure(obj):
        run_tests_using_rally(obj,
                              files_dir,
                              max_preparation_time,
                              rally_extra_opts,
                              keep_temp_files)
    return closure
