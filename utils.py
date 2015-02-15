import time
import logging
import threading
import contextlib
import multiprocessing

import paramiko


logger = logging.getLogger("io-perf-tool")


def get_barrier(count, threaded=False):
    if threaded:
        class val(object):
            value = count
        cond = threading.Condition()
    else:
        val = multiprocessing.Value('i', count)
        cond = multiprocessing.Condition()

    def closure(timeout):
        with cond:
            val.value -= 1
            if val.value == 0:
                cond.notify_all()
            else:
                cond.wait(timeout)
            return val.value == 0

    return closure


def ssh_connect(host, user, key_file, retry_count=60, timeout=1):
    ssh = paramiko.SSHClient()
    ssh.load_host_keys('/dev/null')
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.known_hosts = None

    for i in range(retry_count):
        try:
            ssh.connect(host, username=user, key_filename=key_file,
                        look_for_keys=False)
            return ssh
        except:
            if i == retry_count - 1:
                raise
            time.sleep(timeout)


def wait_on_barrier(barrier, latest_start_time):
    if barrier is not None:
        if latest_start_time is not None:
            timeout = latest_start_time - time.time()
        else:
            timeout = None

        if timeout is not None and timeout > 0:
            msg = "Ready and waiting on barrier. " + \
                  "Will wait at most {0} seconds"
            logger.debug(msg.format(int(timeout)))

            if not barrier(timeout):
                logger.debug("Barrier timeouted")


@contextlib.contextmanager
def log_error(action, types=(Exception,)):
    if not action.startswith("!"):
        logger.debug("Starts : " + action)
    else:
        action = action[1:]

    try:
        yield
    except Exception as exc:
        if isinstance(exc, types) and not isinstance(exc, StopIteration):
            templ = "Error during {0} stage: {1}"
            logger.debug(templ.format(action, exc.message))
        raise


def run_over_ssh(conn, cmd):
    "should be replaces by normal implementation, with select"

    stdin, stdout, stderr = conn.exec_command(cmd)
    out = stdout.read()
    err = stderr.read()
    code = stdout.channel.recv_exit_status()
    return code, out, err
