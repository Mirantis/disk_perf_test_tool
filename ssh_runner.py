import re
import Queue
import logging
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor

from utils import ssh_connect

import itest
from utils import get_barrier, log_error, wait_on_barrier


logger = logging.getLogger("io-perf-tool")

conn_uri_attrs = ("user", "passwd", "host", "port", "path")


class ConnCreds(object):
    def __init__(self):
        for name in conn_uri_attrs:
            setattr(self, name, None)


uri_reg_exprs = []


class URIsNamespace(object):
    class ReParts(object):
        user_rr = "[^:]*?"
        host_rr = "[^:]*?"
        port_rr = "\\d+"
        key_file_rr = "[^:@]*"
        passwd_rr = ".*?"

    re_dct = ReParts.__dict__

    for attr_name, val in re_dct.items():
        if attr_name.endswith('_rr'):
            new_rr = "(?P<{0}>{1})".format(attr_name[:-3], val)
            setattr(ReParts, attr_name, new_rr)

    re_dct = ReParts.__dict__

    templs = [
        "^{host_rr}$",
        "^{user_rr}@{host_rr}::{key_file_rr}$",
        "^{user_rr}@{host_rr}:{port_rr}:{key_file_rr}$",
        "^{user_rr}:{passwd_rr}@@{host_rr}$",
        "^{user_rr}:{passwd_rr}@@{host_rr}:{port_rr}$",
    ]

    for templ in templs:
        uri_reg_exprs.append(templ.format(**re_dct))


def parse_ssh_uri(uri):
    # user:passwd@ip_host:port
    # user:passwd@ip_host
    # user@ip_host:port
    # user@ip_host
    # ip_host:port
    # ip_host
    # user@ip_host:port:path_to_key_file
    # user@ip_host::path_to_key_file
    # ip_host:port:path_to_key_file
    # ip_host::path_to_key_file

    res = ConnCreds()
    res.port = "22"

    for rr in uri_reg_exprs:
        rrm = re.match(rr, uri)
        if rrm is not None:
            res.__dict__.update(rrm.groupdict())
            return res
    raise ValueError("Can't parse {0!r} as ssh uri value".format(uri))


def connect(uri):
    creds = parse_ssh_uri(uri)
    creds.port = int(creds.port)
    return ssh_connect(creds)


def conn_func(obj, barrier, latest_start_time, conn):
    try:
        test_iter = itest.run_test_iter(obj, conn)
        next(test_iter)

        wait_on_barrier(barrier, latest_start_time)

        with log_error("!Run test"):
            return next(test_iter)
    except:
        print traceback.format_exc()
        raise


def get_ssh_runner(uris,
                   latest_start_time=None,
                   keep_temp_files=False):
    logger.debug("Connecting to servers")

    with ThreadPoolExecutor(max_workers=16) as executor:
        connections = list(executor.map(connect, uris))

    result_queue = Queue.Queue()
    barrier = get_barrier(len(uris), threaded=True)

    def closure(obj):
        ths = []
        obj.set_result_cb(result_queue.put)

        params = (obj, barrier, latest_start_time)

        logger.debug("Start tests")
        for conn in connections:
            th = threading.Thread(None, conn_func, None,
                                  params + (conn,))
            th.daemon = True
            th.start()
            ths.append(th)

        for th in ths:
            th.join()

        test_result = []
        while not result_queue.empty():
            test_result.append(result_queue.get())

        logger.debug("Done. Closing connection")
        for conn in connections:
            conn.close()

        return test_result

    return closure
