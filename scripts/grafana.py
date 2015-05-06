import json


query = """
select value from "{series}"
where $timeFilter and
host='{host}' and device='{device}'
order asc
"""


def make_dashboard_file(config):
    series = ['writes_completed', 'sectors_written']
    dashboards = []

    for serie in series:
        dashboard = dict(title=serie, type='graph',
                         span=12, fill=1, linewidth=2,
                         tooltip={'shared': True})

        targets = []

        for ip, devs in config.items():
            for device in devs:
                params = {
                    'series': serie,
                    'host': ip,
                    'device': device
                }

                target = dict(
                    target="disk io",
                    query=query.replace("\n", " ").format(**params).strip(),
                    interval="",
                    alias="{0} io {1}".format(ip, device),
                    rawQuery=True
                )
                targets.append(target)

        dashboard['targets'] = targets
        dashboards.append(dashboard)

    fc = open("grafana_template.js").read()
    return fc % (json.dumps(dashboards),)


print make_dashboard_file({'192.168.0.104': ['sda1', 'rbd1']})
