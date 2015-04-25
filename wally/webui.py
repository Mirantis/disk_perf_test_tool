import time
import random
import os.path
import logging
import calendar
import datetime
import threading

import cherrypy
from cherrypy import tools

import wally

logger = logging.getLogger("wally.webui")


def to_timestamp(str_datetime):
    dt, str_gmt_offset = str_datetime.split("GMT", 1)
    dt = dt.strip().split(" ", 1)[1]
    dto = datetime.datetime.strptime(dt, "%b %d %Y %H:%M:%S")
    timestamp = calendar.timegm(dto.timetuple())
    str_gmt_offset = str_gmt_offset.strip().split(" ", 1)[0]
    gmt_offset = int(str_gmt_offset)
    gmt_offset_sec = gmt_offset // 100 * 3600 + (gmt_offset % 100) * 60
    return timestamp - gmt_offset_sec


def backfill_thread(dstore):
    with dstore.lock:
        for i in range(600):
            dstore.data['disk_io'].append(int(random.random() * 100))
            dstore.data['net_io'].append(int(random.random() * 100))

    while True:
        time.sleep(1)
        with dstore.lock:
            dstore.data['disk_io'].append(int(random.random() * 100))
            dstore.data['net_io'].append(int(random.random() * 100))


class WebWally(object):

    def __init__(self, sensors_data_storage):
        self.storage = sensors_data_storage

    @cherrypy.expose
    @tools.json_out()
    def sensors(self, start, stop, step, name):
        try:
            start = to_timestamp(start)
            stop = to_timestamp(stop)

            with self.storage.lock:
                data = self.storage.data[name]
        except Exception:
            logger.exception("During parse input data")
            raise cherrypy.HTTPError("Wrong date format")

        if step != 1000:
            raise cherrypy.HTTPError("Step must be equals to 1s")

        num = stop - start

        if len(data) > num:
            data = data[-num:]
        else:
            data = [0] * (num - len(data)) + data

        return data

    @cherrypy.expose
    def index(self):
        idx = os.path.dirname(wally.__file__)
        idx = os.path.join(idx, "sensors.html")
        return open(idx).read()


def web_main_thread(sensors_data_storage):

    cherrypy.config.update({'environment': 'embedded',
                            'server.socket_port': 8089,
                            'engine.autoreload_on': False})

    th = threading.Thread(None, backfill_thread, "backfill_thread",
                          (sensors_data_storage,))
    th.daemon = True
    th.start()

    cherrypy.quickstart(WebWally(sensors_data_storage), '/')


def web_main_stop():
    cherrypy.engine.stop()
