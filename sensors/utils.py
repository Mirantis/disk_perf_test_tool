import os

from collections import namedtuple

SensorInfo = namedtuple("SensorInfo", ['value', 'is_accumulated'])


def is_dev_accepted(name, disallowed_prefixes, allowed_prefixes):
    dev_ok = True

    if disallowed_prefixes is not None:
        dev_ok = all(not name.startswith(prefix)
                     for prefix in disallowed_prefixes)

    if dev_ok and allowed_prefixes is not None:
        dev_ok = any(name.startswith(prefix)
                     for prefix in allowed_prefixes)

    return dev_ok


def get_pid_list(disallowed_prefixes, allowed_prefixes):
    """ Return pid list from list of pids and names """
    # exceptions
    but = disallowed_prefixes if disallowed_prefixes is not None else []
    if allowed_prefixes is None:
        # if nothing setted - all ps will be returned except setted
        result = [pid
                  for pid in os.listdir('/proc')
                  if pid.isdigit() and pid not in but]
    else:
        result = []
        for pid in os.listdir('/proc'):
            if pid.isdigit() and pid not in but:
                name = get_pid_name(pid)
                if pid in allowed_prefixes or \
                   any(name.startswith(val) for val in allowed_prefixes):
                    print name
                    # this is allowed pid?
                    result.append(pid)
    return result


def get_pid_name(pid):
    """ Return name by pid """
    try:
        with open(os.path.join('/proc/', pid, 'cmdline'), 'r') as pidfile:
            try:
                cmd = pidfile.readline().split()[0]
                return os.path.basename(cmd).rstrip('\x00')
            except IndexError:
                # no cmd returned
                return "no_name"
    except IOError:
        return "no_such_process"


def delta(func, only_upd=True):
    prev = {}
    while True:
        for dev_name, vals in func():
            if dev_name not in prev:
                prev[dev_name] = {}
                for name, (val, _) in vals.items():
                    prev[dev_name][name] = val
            else:
                dev_prev = prev[dev_name]
                res = {}
                for stat_name, (val, accum_val) in vals.items():
                    if accum_val:
                        if stat_name in dev_prev:
                            delta = int(val) - int(dev_prev[stat_name])
                            if not only_upd or 0 != delta:
                                res[stat_name] = str(delta)
                        dev_prev[stat_name] = val
                    elif not only_upd or '0' != val:
                        res[stat_name] = val

                if only_upd and len(res) == 0:
                    continue
                yield dev_name, res
        yield None, None
