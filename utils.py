import time
import socket
import os.path
import logging
import threading
import contextlib
import multiprocessing


logger = logging.getLogger("io-perf-tool")


def parse_creds(creds):
    # parse user:passwd@host
    user, passwd_host = creds.split(":", 1)

    if '@' not in passwd_host:
        passwd, host = passwd_host, None
    else:
        passwd, host = passwd_host.rsplit('@', 1)

    return user, passwd, host


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


def ssize_to_b(ssize):
    try:
        smap = dict(k=1, K=1, M=1024, m=1024, G=1024**2, g=1024**2)
        for ext, coef in smap.items():
            if ssize.endswith(ext):
                return int(ssize[:-1]) * coef * 1024

        return int(ssize)
    except (ValueError, TypeError, AttributeError):
        tmpl = "Unknow size format {0!r} (or size not multiples 1024)"
        raise ValueError(tmpl.format(ssize))
