import os
import sys
import uuid
import logging
import datetime
import contextlib

from typing import Any, Tuple, Iterator, Iterable

try:
    from petname import Generate as pet_generate
except ImportError:
    def pet_generate(_1: str, _2: str) -> str:
        return str(uuid.uuid4())

from cephlib.common import run_locally, sec_to_str


logger = logging.getLogger("wally")


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


def log_block(message: str, exc_logger:logging.Logger = None) -> LogError:
    logger.debug("Starts : " + message)
    return LogError(message, exc_logger)


def check_input_param(is_ok: bool, message: str) -> None:
    if not is_ok:
        logger.error(message)
        raise StopTestError(message)


def yamable(data: Any) -> Any:
    if isinstance(data, (tuple, list)):
        return map(yamable, data)

    if isinstance(data, dict):
        res = {}
        for k, v in data.items():
            res[yamable(k)] = yamable(v)
        return res

    return data


def get_creds_openrc(path: str) -> Tuple[str, str, str, str, bool]:
    fc = open(path).read()

    echo = 'echo "$OS_INSECURE:$OS_TENANT_NAME:$OS_USERNAME:$OS_PASSWORD@$OS_AUTH_URL"'

    msg = "Failed to get creads from openrc file"
    with LogError(msg):
        data = run_locally(['/bin/bash'], input_data=(fc + "\n" + echo).encode('utf8')).decode("utf8")

    msg = "Failed to get creads from openrc file: " + data
    with LogError(msg):
        data = data.strip()
        insecure_str, user, tenant, passwd_auth_url = data.split(':', 3)
        insecure = (insecure_str in ('1', 'True', 'true'))
        passwd, auth_url = passwd_auth_url.rsplit("@", 1)
        assert (auth_url.startswith("https://") or
                auth_url.startswith("http://"))

    return user, passwd, tenant, auth_url, insecure


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


def get_time_interval_printable_info(seconds: int) -> Tuple[str, str]:
    exec_time_s = sec_to_str(seconds)
    now_dt = datetime.datetime.now()
    end_dt = now_dt + datetime.timedelta(0, seconds)
    return exec_time_s, "{:%H:%M:%S}".format(end_dt)


def shape2str(shape: Iterable[int]) -> str:
    return "*".join(map(str, shape))


def str2shape(shape: str) -> Tuple[int, ...]:
    return tuple(map(int, shape.split('*')))
