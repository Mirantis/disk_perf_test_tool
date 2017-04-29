import re
import os
import sys
import math
import time
import uuid
import socket
import logging
import datetime
import ipaddress
import threading
import contextlib
import subprocess
from fractions import Fraction


from typing import Any, Tuple, Union, List, Iterator, Iterable, Optional, IO, cast, TypeVar, Callable

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


STORAGE_ROLES = {'ceph-osd'}


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


class TaskFinished(Exception):
    pass


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


def b2ssize(value: Union[int, float]) -> str:
    if isinstance(value, float) and value < 100:
        return b2ssize_10(value)

    value = int(value)
    if value < 1024:
        return str(value) + " "

    # make mypy happy
    scale = 1
    name = ""

    for name, scale in RSMAP:
        if value < 1024 * scale:
            if value % scale == 0:
                return "{} {}i".format(value // scale, name)
            else:
                return "{:.1f} {}i".format(float(value) / scale, name)

    return "{}{}i".format(value // scale, name)


RSMAP_10 = [(' f', 0.001 ** 4),
            (' n', 0.001 ** 3),
            (' u', 0.001 ** 2),
            (' m', 0.001),
            (' ', 1),
            (' K', 1000),
            (' M', 1000 ** 2),
            (' G', 1000 ** 3),
            (' T', 1000 ** 4),
            (' P', 1000 ** 5),
            (' E', 1000 ** 6)]


def has_next_digit_after_coma(x: float) -> bool:
    return x * 10 - int(x * 10) > 1


def has_second_digit_after_coma(x: float) -> bool:
    return (x * 10 - int(x * 10)) * 10 > 1


def b2ssize_10(value: Union[int, float]) -> str:
    # make mypy happy
    scale = 1
    name = " "

    if value == 0.0:
        return "0 "

    if value / RSMAP_10[0][1] < 1.0:
        return "{:.2e} ".format(value)

    for name, scale in RSMAP_10:
        cval = value / scale
        if cval < 1000:
            # detect how many digits after dot to show
            if cval > 100:
                return "{}{}".format(int(cval), name)
            if cval > 10:
                if has_next_digit_after_coma(cval):
                    return "{:.1f}{}".format(cval, name)
                else:
                    return "{}{}".format(int(cval), name)
            if cval >= 1:
                if has_second_digit_after_coma(cval):
                    return "{:.2f}{}".format(cval, name)
                elif has_next_digit_after_coma(cval):
                    return "{:.1f}{}".format(cval, name)
                return "{}{}".format(int(cval), name)
            raise AssertionError("Can't get here")

    return "{}{}".format(int(value // scale), name)


def run_locally(cmd: Union[str, List[str]], input_data: str = "", timeout: int = 20) -> str:
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


def which(program: str) -> Optional[str]:
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    for path in os.environ["PATH"].split(os.pathsep):
        path = path.strip('"')
        exe_file = os.path.join(path, program)
        if is_exe(exe_file):
            return exe_file

    return None


@contextlib.contextmanager
def empty_ctx(val: Any = None) -> Iterator[Any]:
    yield val


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


def get_time_interval_printable_info(seconds: int) -> Tuple[str, str]:
    exec_time_s = sec_to_str(seconds)
    now_dt = datetime.datetime.now()
    end_dt = now_dt + datetime.timedelta(0, seconds)
    return exec_time_s, "{:%H:%M:%S}".format(end_dt)


FM_FUNC_INPUT = TypeVar("FM_FUNC_INPUT")
FM_FUNC_RES = TypeVar("FM_FUNC_RES")


def flatmap(func: Callable[[FM_FUNC_INPUT], Iterable[FM_FUNC_RES]],
            inp_iter: Iterable[FM_FUNC_INPUT]) -> Iterator[FM_FUNC_RES]:
    for val in inp_iter:
        for res in func(val):
            yield res


_coefs = {
    'n': Fraction(1, 1000**3),
    'u': Fraction(1, 1000**2),
    'm': Fraction(1, 1000),
    'K': 1000,
    'M': 1000 ** 2,
    'G': 1000 ** 3,
    'Ki': 1024,
    'Mi': 1024 ** 2,
    'Gi': 1024 ** 3,
}


def split_unit(units: str) -> Tuple[Union[Fraction, int], str]:
    if len(units) > 2 and units[:2] in _coefs:
        return _coefs[units[:2]], units[2:]
    if len(units) > 1 and units[0] in _coefs:
        return _coefs[units[0]], units[1:]
    else:
        return 1, units


def unit_conversion_coef(from_unit: str, to_unit: str) -> Union[Fraction, int]:
    f1, u1 = split_unit(from_unit)
    f2, u2 = split_unit(to_unit)

    assert u1 == u2, "Can't convert {!r} to {!r}".format(from_unit, to_unit)

    if isinstance(f1, int) and isinstance(f2, int):
        if f1 % f2 != 0:
            return Fraction(f1, f2)
        else:
            return f1 // f2

    res = f1 / f2

    if isinstance(res, Fraction) and cast(Fraction, res).denominator == 1:
        return cast(Fraction, res).numerator

    return res


def shape2str(shape: Iterable[int]) -> str:
    return "*".join(map(str, shape))


def str2shape(shape: str) -> Tuple[int, ...]:
    return tuple(map(int, shape.split('*')))
