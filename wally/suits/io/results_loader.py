import re
import json


def parse_output(out_err):
    err_start_patt = r"(?ims)=+\s+ERROR\s+=+"
    err_end_patt = r"(?ims)=+\s+END OF ERROR\s+=+"

    for block in re.split(err_start_patt, out_err)[1:]:
        tb, garbage = re.split(err_end_patt, block)
        msg = "Test fails with error:\n" + tb.strip() + "\n"
        raise OSError(msg)

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


def filter_processed_data(name_prefix, fields_to_select, **filters):
    def closure(data):
        for name, result in data.items():
            if name_prefix is not None:
                if not name.startswith(name_prefix):
                    continue

            for k, v in filters.items():
                if result.raw.get(k) != v:
                    break
            else:
                yield map(result.raw.get, fields_to_select)
    return closure
