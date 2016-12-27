import re
import os
import abc
import sys
import math
import time
import uuid
import socket
import logging
import ipaddress
import threading
import contextlib
import subprocess
import collections

from .node_interfaces import IRPCNode
from typing import (Any, Tuple, Union, List, Iterator, Dict, Iterable, Optional,
                    IO, Sequence, NamedTuple, cast, TypeVar)

try:
    import psutil
except ImportError:
    psutil = None

try:
    from petname import Generate as pet_generate
except ImportError:
    def pet_generate(x: str, y: str) -> str:
        return str(uuid.uuid4())


logger = logging.getLogger("wally")
TNumber = TypeVar('TNumber', int, float)
Number = Union[int, float]


class StopTestError(RuntimeError):
    pass


class LogError:
    def __init__(self, message: str, exc_logger: logging.Logger = None) -> None:
        self.message = message
        self.exc_logger = exc_logger

    def __enter__(self) -> 'LogError':
        return self

    def __exit__(self, tp: type, value: Exception, traceback: Any) -> bool:
        if value is None or isinstance(value, StopTestError):
            return False

        if self.exc_logger is None:
            exc_logger = sys._getframe(1).f_globals.get('logger', logger)
        else:
            exc_logger = self.exc_logger

        exc_logger.exception(self.message, exc_info=(tp, value, traceback))
        raise StopTestError(self.message) from value


class IStorable(metaclass=abc.ABCMeta):
    """Interface for type, which can be stored"""

    @abc.abstractmethod
    def raw(self) -> Dict[str, Any]:
        pass

    @abc.abstractclassmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'IStorable':
        pass


Basic = Union[int, str, bytes, bool, None]
Storable = Union[IStorable, Dict[str, Any], List[Any], int, str, bytes, bool, None]


class TaskFinished(Exception):
    pass


class Barrier:
    def __init__(self, count: int) -> None:
        self.count = count
        self.curr_count = 0
        self.cond = threading.Condition()
        self.exited = False

    def wait(self, timeout: int=None) -> bool:
        with self.cond:
            if self.exited:
                raise TaskFinished()

            self.curr_count += 1
            if self.curr_count == self.count:
                self.curr_count = 0
                self.cond.notify_all()
                return True
            else:
                self.cond.wait(timeout=timeout)
                return False

    def exit(self) -> None:
        with self.cond:
            self.exited = True


class Timeout(Iterable[float]):
    def __init__(self, timeout: int, message: str = None, min_tick: int = 1, no_exc: bool = False) -> None:
        self.end_time = time.time() + timeout
        self.message = message
        self.min_tick = min_tick
        self.prev_tick_at = time.time()
        self.no_exc = no_exc

    def tick(self) -> bool:
        current_time = time.time()

        if current_time > self.end_time:
            if self.message:
                msg = "Timeout: {}".format(self.message)
            else:
                msg = "Timeout"

            if self.no_exc:
                return False

            raise TimeoutError(msg)

        sleep_time = self.min_tick - (current_time - self.prev_tick_at)
        if sleep_time > 0:
            time.sleep(sleep_time)
            self.prev_tick_at = time.time()
        else:
            self.prev_tick_at = current_time

        return True

    def __iter__(self) -> Iterator[float]:
        return cast(Iterator[float], self)

    def __next__(self) -> float:
        if not self.tick():
            raise StopIteration()
        return self.end_time - time.time()


def greater_digit_pos(val: Number) -> int:
    return int(math.floor(math.log10(val))) + 1


def round_digits(val: TNumber, num_digits: int = 3) -> TNumber:
    pow = 10 ** (greater_digit_pos(val) - num_digits)
    return type(val)(int(val / pow) * pow)


def is_ip(data: str) -> bool:
    try:
        ipaddress.ip_address(data)
        return True
    except ValueError:
        return False


def log_block(message: str, exc_logger:logging.Logger = None) -> LogError:
    logger.debug("Starts : " + message)
    return LogError(message, exc_logger)


def check_input_param(is_ok: bool, message: str) -> None:
    if not is_ok:
        logger.error(message)
        raise StopTestError(message)


def parse_creds(creds: str) -> Tuple[str, str, str]:
    """Parse simple credentials format user[:passwd]@host"""
    user, passwd_host = creds.split(":", 1)

    if '@' not in passwd_host:
        passwd, host = passwd_host, None
    else:
        passwd, host = passwd_host.rsplit('@', 1)

    return user, passwd, host


SMAP = dict(k=1024, m=1024 ** 2, g=1024 ** 3, t=1024 ** 4)


def ssize2b(ssize: Union[str, int]) -> int:
    try:
        if isinstance(ssize, int):
            return ssize

        ssize = ssize.lower()
        if ssize[-1] in SMAP:
            return int(ssize[:-1]) * SMAP[ssize[-1]]
        return int(ssize)
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Unknow size format {!r}".format(ssize))


RSMAP = [('K', 1024),
         ('M', 1024 ** 2),
         ('G', 1024 ** 3),
         ('T', 1024 ** 4)]


def b2ssize(size: int) -> str:
    if size < 1024:
        return str(size)

    # make mypy happy
    scale = 1
    name = ""

    for name, scale in RSMAP:
        if size < 1024 * scale:
            if size % scale == 0:
                return "{} {}i".format(size // scale, name)
            else:
                return "{:.1f} {}i".format(float(size) / scale, name)

    return "{}{}i".format(size // scale, name)


RSMAP_10 = [('k', 1000),
            ('m', 1000 ** 2),
            ('g', 1000 ** 3),
            ('t', 1000 ** 4)]


def b2ssize_10(size: int) -> str:
    if size < 1000:
        return str(size)

    # make mypy happy
    scale = 1
    name = ""

    for name, scale in RSMAP_10:
        if size < 1000 * scale:
            if size % scale == 0:
                return "{} {}".format(size // scale, name)
            else:
                return "{:.1f} {}".format(float(size) / scale, name)

    return "{}{}".format(size // scale, name)


def run_locally(cmd: Union[str, List[str]], input_data: str="", timeout:int =20) -> str:
    if isinstance(cmd, str):
        shell = True
        cmd_str = cmd
    else:
        shell = False
        cmd_str = " ".join(cmd)

    proc = subprocess.Popen(cmd,
                            shell=shell,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    res = []  # type: List[Tuple[bytes, bytes]]

    def thread_func() -> None:
        rr = proc.communicate(input_data.encode("utf8"))
        res.extend(rr)

    thread = threading.Thread(target=thread_func,
                              name="Local cmd execution")
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        if psutil is not None:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        else:
            proc.kill()

        thread.join()
        raise RuntimeError("Local process timeout: " + cmd_str)

    stdout_data, stderr_data = zip(*res)  # type: List[bytes], List[bytes]

    out = b"".join(stdout_data).decode("utf8")
    err = b"".join(stderr_data).decode("utf8")

    if 0 != proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode,
                                            cmd_str, out + err)

    return out


def get_ip_for_target(target_ip: str) -> str:
    if not is_ip(target_ip):
        target_ip = socket.gethostbyname(target_ip)

    first_dig = map(int, target_ip.split("."))
    if first_dig == 127:
        return '127.0.0.1'

    data = run_locally('ip route get to'.split(" ") + [target_ip])

    rr1 = r'{0} via [.0-9]+ dev (?P<dev>.*?) src (?P<ip>[.0-9]+)$'
    rr1 = rr1.replace(" ", r'\s+')
    rr1 = rr1.format(target_ip.replace('.', r'\.'))

    rr2 = r'{0} dev (?P<dev>.*?) src (?P<ip>[.0-9]+)$'
    rr2 = rr2.replace(" ", r'\s+')
    rr2 = rr2.format(target_ip.replace('.', r'\.'))

    data_line = data.split("\n")[0].strip()
    res1 = re.match(rr1, data_line)
    res2 = re.match(rr2, data_line)

    if res1 is not None:
        return res1.group('ip')

    if res2 is not None:
        return res2.group('ip')

    raise OSError("Can't define interface for {0}".format(target_ip))


def open_for_append_or_create(fname: str) -> IO[str]:
    if not os.path.exists(fname):
        return open(fname, "w")

    fd = open(fname, 'r+')
    fd.seek(0, os.SEEK_END)
    return fd


def sec_to_str(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return "{}:{:02d}:{:02d}".format(h, m, s)


def yamable(data: Any) -> Any:
    if isinstance(data, (tuple, list)):
        return map(yamable, data)

    if isinstance(data, dict):
        res = {}
        for k, v in data.items():
            res[yamable(k)] = yamable(v)
        return res

    return data


def flatten(data: Iterable[Any]) -> List[Any]:
    res = []
    for i in data:
        if isinstance(i, (list, tuple, set)):
            res.extend(flatten(i))
        else:
            res.append(i)
    return res


def get_creds_openrc(path: str) -> Tuple[str, str, str, str, bool]:
    fc = open(path).read()

    echo = 'echo "$OS_INSECURE:$OS_TENANT_NAME:$OS_USERNAME:$OS_PASSWORD@$OS_AUTH_URL"'

    msg = "Failed to get creads from openrc file"
    with LogError(msg):
        data = run_locally(['/bin/bash'], input_data=fc + "\n" + echo)

    msg = "Failed to get creads from openrc file: " + data
    with LogError(msg):
        data = data.strip()
        insecure_str, user, tenant, passwd_auth_url = data.split(':', 3)
        insecure = (insecure_str in ('1', 'True', 'true'))
        passwd, auth_url = passwd_auth_url.rsplit("@", 1)
        assert (auth_url.startswith("https://") or
                auth_url.startswith("http://"))

    return user, passwd, tenant, auth_url, insecure


OSRelease = NamedTuple("OSRelease",
                       [("distro", str),
                        ("release", str),
                        ("arch", str)])


def get_os(node: IRPCNode) -> OSRelease:
    """return os type, release and architecture for node.
    """
    arch = node.run("arch", nolog=True).strip()

    try:
        node.run("ls -l /etc/redhat-release", nolog=True)
        return OSRelease('redhat', None, arch)
    except:
        pass

    try:
        node.run("ls -l /etc/debian_version", nolog=True)

        release = None
        for line in node.run("lsb_release -a", nolog=True).split("\n"):
            if ':' not in line:
                continue
            opt, val = line.split(":", 1)

            if opt == 'Codename':
                release = val.strip()

        return OSRelease('ubuntu', release, arch)
    except:
        pass

    raise RuntimeError("Unknown os")


@contextlib.contextmanager
def empty_ctx(val: Any = None) -> Iterator[Any]:
    yield val


def log_nodes_statistic(nodes: Sequence[IRPCNode]) -> None:
    logger.info("Found {0} nodes total".format(len(nodes)))

    per_role = collections.defaultdict(int)  # type: Dict[str, int]
    for node in nodes:
        for role in node.info.roles:
            per_role[role] += 1

    for role, count in sorted(per_role.items()):
        logger.debug("Found {0} nodes with role {1}".format(count, role))


def which(program: str) -> Optional[str]:
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    for path in os.environ["PATH"].split(os.pathsep):
        path = path.strip('"')
        exe_file = os.path.join(path, program)
        if is_exe(exe_file):
            return exe_file

    return None


def get_uniq_path_uuid(path: str, max_iter: int = 10) -> Tuple[str, str]:
    for i in range(max_iter):
        run_uuid = pet_generate(2, "_")
        results_dir = os.path.join(path, run_uuid)
        if not os.path.exists(results_dir):
            break
    else:
        run_uuid = str(uuid.uuid4())
        results_dir = os.path.join(path, run_uuid)

    return results_dir, run_uuid


def to_ip(host_or_ip: str) -> str:
    # translate hostname to address
    try:
        ipaddress.ip_address(host_or_ip)
        return host_or_ip
    except ValueError:
        ip_addr = socket.gethostbyname(host_or_ip)
        logger.info("Will use ip_addr %r instead of hostname %r", ip_addr, host_or_ip)
        return ip_addr
