__doc__ = "functions for make pretty yaml files"
__all__ = ['dumps']


def dumps_simple(val):
    bad_symbols = set(" \r\t\n,':{}[]><;")

    if isinstance(val, basestring):
        if isinstance(val, unicode):
            val = val.encode('utf8')

        try:
            float(val)
            val = repr(val)
        except ValueError:
            if len(bad_symbols & set(val)) != 0:
                val = repr(val)

        return val
    elif val is True:
        return 'true'
    elif val is False:
        return 'false'
    elif val is None:
        return 'null'

    return str(val)


def is_simple(val):
    simple_type = isinstance(val, (str, unicode, int, long, bool, float))
    return simple_type or val is None


def all_nums(vals):
    return all(isinstance(val, (int, float, long)) for val in vals)


def dumpv(data, tab_sz=4, width=160, min_width=40):
    tab = ' ' * tab_sz

    if width < min_width:
        width = min_width

    res = []
    if is_simple(data):
        return [dumps_simple(data)]

    if isinstance(data, (list, tuple)):
        if all(map(is_simple, data)):
            if all_nums(data):
                one_line = "[{0}]".format(", ".join(map(dumps_simple, data)))
            else:
                one_line = "[{0}]".format(",".join(map(dumps_simple, data)))
        elif len(data) == 0:
            one_line = "[]"
        else:
            one_line = None

        if one_line is None or len(one_line) > width:
            pref = "-" + ' ' * (tab_sz - 1)

            for val in data:
                items = dumpv(val, tab_sz, width - tab_sz, min_width)
                items = [pref + items[0]] + \
                        [tab + item for item in items[1:]]
                res.extend(items)
        else:
            res.append(one_line)
    elif isinstance(data, dict):
        if len(data) == 0:
            res.append("{}")
        else:
            assert all(map(is_simple, data.keys()))

            one_line = None
            if all(map(is_simple, data.values())):
                one_line = ", ".join(
                    "{0}: {1}".format(dumps_simple(k), dumps_simple(v))
                    for k, v in sorted(data.items()))
                one_line = "{" + one_line + "}"
                if len(one_line) > width:
                    one_line = None

            if one_line is None:
                for k, v in data.items():
                    key_str = dumps_simple(k) + ": "
                    val_res = dumpv(v, tab_sz, width - tab_sz, min_width)

                    if len(val_res) == 1 and \
                       len(key_str + val_res[0]) < width and \
                       not isinstance(v, dict) and \
                       not val_res[0].strip().startswith('-'):
                        res.append(key_str + val_res[0])
                    else:
                        res.append(key_str)
                        res.extend(tab + i for i in val_res)
            else:
                res.append(one_line)
    else:
        try:
            get_yamable = data.get_yamable
        except AttributeError:
            raise ValueError("Can't pack {0!r}".format(data))
        res = dumpv(get_yamable(), tab_sz, width, min_width)

    return res


def dumps(data, tab_sz=4, width=120, min_width=40):
    return "\n".join(dumpv(data, tab_sz, width, min_width))
