import Queue
import logging
import threading

from .deploy_sensors import (deploy_and_start_sensors,
                             stop_and_remove_sensors)
from .protocol import create_protocol, Timeout


__all__ = ['Empty', 'recv_main',
           'deploy_and_start_sensors',
           'SensorConfig',
           'stop_and_remove_sensors',
           'start_listener_thread',
           ]


Empty = Queue.Empty
logger = logging.getLogger("wally.sensors")


class SensorConfig(object):
    def __init__(self, conn, url, sensors, source_id,
                 monitor_url=None):
        self.conn = conn
        self.url = url
        self.sensors = sensors
        self.source_id = source_id
        self.monitor_url = monitor_url


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


def start_listener_thread(uri):
    data_q = Queue.Queue()
    cmd_q = Queue.Queue()
    logger.debug("Listening for sensor data on " + uri)
    proto = create_protocol(uri, receiver=True)
    th = threading.Thread(None, recv_main, None, (proto, data_q, cmd_q))
    th.daemon = True
    th.start()

    def stop_thread():
        cmd_q.put(None)
        th.join()

    return data_q, stop_thread
