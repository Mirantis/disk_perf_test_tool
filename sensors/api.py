import Queue
import threading

from contextlib import contextmanager

from deploy_sensors import (deploy_and_start_sensors,
                            stop_and_remove_sensors)
from protocol import create_protocol, Timeout


Empty = Queue.Empty


def recv_main(proto, data_q, cmd_q):
    while True:
        try:
            data_q.put(proto.recv(0.1))
        except Timeout:
            pass

        try:
            val = cmd_q.get(False)

            if val is None:
                return

        except Queue.Empty:
            pass


@contextmanager
def start_monitoring(uri, config=None, connected_config=None):
    deploy_and_start_sensors(uri, config=config,
                             connected_config=connected_config)
    try:
        data_q = Queue.Queue()
        cmd_q = Queue.Queue()
        proto = create_protocol(uri, receiver=True)
        th = threading.Thread(None, recv_main, None, (proto, data_q, cmd_q))
        th.daemon = True
        th.start()

        try:
            yield data_q
        finally:
            cmd_q.put(None)
            th.join()
    finally:
        stop_and_remove_sensors(config,
                                connected_config=connected_config)
