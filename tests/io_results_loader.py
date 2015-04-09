import re
import json


from disk_perf_test_tool.utils import ssize_to_b
from disk_perf_test_tool.scripts.postprocessing import data_stat


def parse_output(out_err):
    start_patt = r"(?ims)=+\s+RESULTS\(format=json\)\s+=+"
    end_patt = r"(?ims)=+\s+END OF RESULTS\s+=+"

    for block in re.split(start_patt, out_err)[1:]:
        data, garbage = re.split(end_patt, block)
        yield json.loads(data.strip())

    start_patt = r"(?ims)=+\s+RESULTS\(format=eval\)\s+=+"
    end_patt = r"(?ims)=+\s+END OF RESULTS\s+=+"

    for block in re.split(start_patt, out_err)[1:]:
        data, garbage = re.split(end_patt, block)
        yield eval(data.strip())


def filter_data(name_prefix, fields_to_select, **filters):
    def closure(data):
        for result in data:
            if name_prefix is not None:
                if not result['jobname'].startswith(name_prefix):
                    continue

            for k, v in filters.items():
                if result.get(k) != v:
                    break
            else:
                yield map(result.get, fields_to_select)
    return closure


def load_data(raw_data):
    data = list(parse_output(raw_data))[0]

    for key, val in data['res'].items():
        if 'blocksize' not in val:
            val['blocksize'] = key.split('_')[2][3:].split('th')[0]

        val['blocksize_b'] = ssize_to_b(val['blocksize'])

        val['iops_mediana'], val['iops_stddev'] = \
            data_stat.med_dev(val['iops'])
        val['bw_mediana'], val['bw_stddev'] = data_stat.med_dev(val['bw_mean'])
        val['lat_mediana'], val['lat_stddev'] = data_stat.med_dev(val['lat'])
        yield val


def load_files(*fnames):
    for fname in fnames:
        for i in load_data(open(fname).read()):
            yield i
