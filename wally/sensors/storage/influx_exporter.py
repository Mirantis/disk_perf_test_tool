from urlparse import urlparse
from influxdb import InfluxDBClient


def connect(url):
    parsed_url = urlparse(url)
    user_passwd, host_port = parsed_url.netloc.rsplit("@", 1)
    user, passwd = user_passwd.split(":", 1)
    host, port = host_port.split(":")
    return InfluxDBClient(host, int(port), user, passwd, parsed_url.path[1:])


def add_data(conn, hostname, data):
    per_sensor_data = {}
    for serie in data:
        serie = serie.copy()
        gtime = serie.pop('time')
        for key, val in serie.items():
            dev, sensor = key.split('.')
            data = per_sensor_data.setdefault(sensor, [])
            data.append([gtime, hostname, dev, val])

    infl_data = []
    columns = ['time', 'host', 'device', 'value']
    for sensor_name, points in per_sensor_data.items():
        infl_data.append(
            {'columns': columns,
             'name': sensor_name,
             'points': points})

    conn.write_points(infl_data)
