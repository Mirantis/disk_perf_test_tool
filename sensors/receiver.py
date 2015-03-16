import yaml

from api import start_monitoring, Empty
from influx_exporter import connect, add_data

monitor_config = yaml.load(open("config.yaml").read())

uri = "udp://192.168.0.104:12001"
infldb_url = "influxdb://perf:perf@192.168.152.42:8086/perf"
conn = connect(infldb_url)

sw_per_ip = {}
count = 4
expected = ['192.168.0.104', '192.168.152.41',
            '192.168.152.39', '192.168.152.40']

with start_monitoring(uri, monitor_config) as queue:
    while True:
        try:
            (ip, port), data = queue.get(True, 1)

            if 'sda1.sectors_written' in data:
                val = data['sda1.sectors_written']
            elif 'sdb.sectors_written' in data:
                val = data['sdb.sectors_written']
            else:
                val = 0

            sw_per_ip[ip] = sw_per_ip.get(ip, 0) + val
            count -= 1

            if 0 == count:
                try:
                    vals = [sw_per_ip[ip] for ip in expected]
                    print ("{:>6}" * 4).format(*vals)
                    sw_per_ip = {}
                    count = 4
                except KeyError:
                    pass

            add_data(conn, ip, [data])
        except Empty:
            pass
