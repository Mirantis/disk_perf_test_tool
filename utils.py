import re
import logging
import threading
import contextlib
import subprocess


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
                return True
            else:
                self.cond.wait(timeout=timeout)
                return False

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


def get_ip_for_target(target_ip):
    cmd = 'ip route get to'.split(" ") + [target_ip]
    data = subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout.read()

    rr = r'{0} via [.0-9]+ dev (?P<dev>.*?) src (?P<ip>[.0-9]+)$'
    rr = rr.replace(" ", r'\s+')
    rr = rr.format(target_ip.replace('.', r'\.'))

    data_line = data.split("\n")[0].strip()
    res = re.match(rr, data_line)

    if res is None:
        raise OSError("Can't define interface for {0}".format(target_ip))

    return res.group('ip')
