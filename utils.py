import time
import socket
import logging
import threading
import contextlib


logger = logging.getLogger("io-perf-tool")


def parse_creds(creds):
    # parse user:passwd@host
    user, passwd_host = creds.split(":", 1)

    if '@' not in passwd_host:
        passwd, host = passwd_host, None
    else:
        passwd, host = passwd_host.rsplit('@', 1)

    return user, passwd, host


class TaksFinished(Exception):
    pass


class Barrier(object):
    def __init__(self, count):
        self.count = count
        self.curr_count = 0
        self.cond = threading.Condition()
        self.exited = False

    def wait(self, timeout=None):
        with self.cond:
            if self.exited:
                raise TaksFinished()

            self.curr_count += 1
            if self.curr_count == self.count:
                self.curr_count = 0
                self.cond.notify_all()
            else:
                self.cond.wait(timeout=timeout)

    def exit(self):
        with self.cond:
            self.exited = True


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


def run_over_ssh(conn, cmd, stdin_data=None, exec_timeout=60):
    "should be replaces by normal implementation, with select"
    transport = conn.get_transport()
    session = transport.open_session()
    try:
        session.set_combine_stderr(True)

        stime = time.time()
        session.exec_command(cmd)

        if stdin_data is not None:
            session.sendall(stdin_data)

        session.settimeout(1)
        session.shutdown_write()
        output = ""

        while True:
            try:
                ndata = session.recv(1024)
                output += ndata
                if "" == ndata:
                    break
            except socket.timeout:
                pass
            if time.time() - stime > exec_timeout:
                return 1, output + "\nExecution timeout"
        code = session.recv_exit_status()
    finally:
        session.close()

    return code, output


SMAP = dict(k=1024, m=1024 ** 2, g=1024 ** 3, t=1024 ** 4)


def ssize_to_b(ssize):
    try:
        ssize = ssize.lower()

        if ssize.endswith("b"):
            ssize = ssize[:-1]
        if ssize[-1] in SMAP:
            return int(ssize[:-1]) * SMAP[ssize[-1]]
        return int(ssize)
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Unknow size format {0!r}".format(ssize))
