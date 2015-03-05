import time
import socket
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


def ssh_connect(creds, retry_count=60, timeout=1):
    ssh = paramiko.SSHClient()
    ssh.load_host_keys('/dev/null')
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.known_hosts = None
    for i in range(retry_count):
        try:
            if creds.passwd is not None:
                ssh.connect(creds.host,
                            username=creds.user,
                            password=creds.passwd,
                            port=creds.port,
                            allow_agent=False,
                            look_for_keys=False)
                return ssh

            if creds.key_file is not None:
                ssh.connect(creds.host,
                            username=creds.user,
                            key_filename=creds.key_file,
                            look_for_keys=False,
                            port=creds.port)
                return ssh
            raise ValueError("Wrong credentials {0}".format(creds.__dict__))
        except paramiko.PasswordRequiredException:
            raise
        except socket.error:
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
            else:
                logger.debug("Passing barrier, starting test")


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


def kb_to_ssize(ssize):
    size_ext = {
        4: 'P',
        3: 'T',
        2: 'G',
        1: 'M',
        0: 'K'
    }

    for idx in reversed(sorted(size_ext)):
        if ssize > 1024 ** idx:
            ext = size_ext[idx]
            return "{0}{1}".format(int(ssize / 1024 ** idx), ext)
    raise ValueError("Can't convert {0} to kb".format(ssize))


def ssize_to_kb(ssize):
    try:
        smap = dict(k=1, K=1, M=1024, m=1024, G=1024**2, g=1024**2)
        for ext, coef in smap.items():
            if ssize.endswith(ext):
                return int(ssize[:-1]) * coef

        if int(ssize) % 1024 != 0:
            raise ValueError()

        return int(ssize) / 1024

    except (ValueError, TypeError, AttributeError):
        tmpl = "Unknow size format {0!r} (or size not multiples 1024)"
        raise ValueError(tmpl.format(ssize))
