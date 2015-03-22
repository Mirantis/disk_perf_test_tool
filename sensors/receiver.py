from api import start_monitoring, Empty
# from influx_exporter import connect, add_data

uri = "udp://192.168.0.104:12001"
# infldb_url = "influxdb://perf:perf@192.168.152.42:8086/perf"
# conn = connect(infldb_url)

monitor_config = {'127.0.0.1':
                  {"block-io": {'allowed_prefixes': ['sda1', 'rbd1']},
                   "net-io": {"allowed_prefixes": ["virbr2"]}}}

with start_monitoring(uri, monitor_config) as queue:
    while True:
        try:
            (ip, port), data = queue.get(True, 1)
            print (ip, port), data
            # add_data(conn, ip, [data])
        except Empty:
            pass
