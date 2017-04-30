#!/usr/bin/env python3

import re
import os
import sys
import os.path
import argparse
import itertools
from typing import Optional, Iterator, Union, Dict, Iterable, List, Tuple, NamedTuple, Any
from collections import OrderedDict


from ...utils import sec_to_str, ssize2b, flatmap
from .fio_job import Var, FioJobConfig

SECTION = 0
SETTING = 1
INCLUDE = 2


CfgLine = NamedTuple('CfgLine',
                     [('fname', str),
                      ('lineno', int),
                      ('oline', str),
                      ('tp', int),
                      ('name', str),
                      ('val', Any)])


class ParseError(ValueError):
    def __init__(self, msg: str, fname: str, lineno: int, line_cont:Optional[str] = "") -> None:
        ValueError.__init__(self, msg)
        self.file_name = fname
        self.lineno = lineno
        self.line_cont = line_cont

    def __str__(self) -> str:
        msg = "In {0}:{1} ({2}) : {3}"
        return msg.format(self.file_name,
                          self.lineno,
                          self.line_cont,
                          super(ParseError, self).__str__())


def is_name(name: str) -> bool:
    return re.match("[a-zA-Z_][a-zA-Z_0-9]*", name) is not None


def parse_value(val: str) -> Union[int, str, float, List, Var]:
    try:
        return int(val)
    except ValueError:
        pass

    try:
        return float(val)
    except ValueError:
        pass

    if val.startswith('{%'):
        assert val.endswith("%}")
        content = val[2:-2]
        vals = list(i.strip() for i in content.split(','))
        return list(map(parse_value, vals))

    if val.startswith('{'):
        assert val.endswith("}")
        assert is_name(val[1:-1])
        return Var(val[1:-1])

    return val


def fio_config_lexer(fio_cfg: str, fname: str) -> Iterator[CfgLine]:
    for lineno, oline in enumerate(fio_cfg.split("\n")):
        try:
            line = oline.strip()

            if line.startswith("#") or line.startswith(";"):
                continue

            if line == "":
                continue

            if '#' in line:
                raise ParseError("# isn't allowed inside line",
                                 fname, lineno, oline)

            if line.startswith('['):
                yield CfgLine(fname, lineno, oline, SECTION,
                              line[1:-1].strip(), None)
            elif '=' in line:
                opt_name, opt_val = line.split('=', 1)
                yield CfgLine(fname, lineno, oline, SETTING,
                              opt_name.strip(),
                              parse_value(opt_val.strip()))
            elif line.startswith("include "):
                yield CfgLine(fname, lineno, oline, INCLUDE,
                              line.split(" ", 1)[1], None)
            else:
                yield CfgLine(fname, lineno, oline, SETTING, line, '1')

        except Exception as exc:
            raise ParseError(str(exc), fname, lineno, oline)


def fio_config_parse(lexer_iter: Iterable[CfgLine]) -> Iterator[FioJobConfig]:
    in_globals = False
    curr_section = None
    glob_vals = OrderedDict()  # type: Dict[str, Any]
    sections_count = 0

    lexed_lines = list(lexer_iter)  # type: List[CfgLine]
    one_more = True
    includes = {}

    while one_more:
        new_lines = []  # type: List[CfgLine]
        one_more = False
        for line in lexed_lines:
            fname, lineno, oline, tp, name, val = line

            if INCLUDE == tp:
                if not os.path.exists(fname):
                    dirname = '.'
                else:
                    dirname = os.path.dirname(fname)

                new_fname = os.path.join(dirname, name)
                includes[new_fname] = (fname, lineno)

                try:
                    cont = open(new_fname).read()
                except IOError as err:
                    raise ParseError("Error while including file {}: {}".format(new_fname, err), fname, lineno, oline)

                new_lines.extend(fio_config_lexer(cont, new_fname))
                one_more = True
            else:
                new_lines.append(line)

        lexed_lines = new_lines

    suite_section_idx = 0

    for fname, lineno, oline, tp, name, val in lexed_lines:
        if tp == SECTION:
            if curr_section is not None:
                yield curr_section
                curr_section = None

            if name == 'global':
                if sections_count != 0:
                    raise ParseError("[global] section should be only one and first", fname, lineno, oline)
                in_globals = True
            else:
                in_globals = False
                curr_section = FioJobConfig(name, idx=suite_section_idx)
                suite_section_idx += 1
                curr_section.vals = glob_vals.copy()
            sections_count += 1
        else:
            assert tp == SETTING
            if in_globals:
                glob_vals[name] = val
            elif name == name.upper():
                raise ParseError("Param {!r} not in [global] section".format(name), fname, lineno, oline)
            elif curr_section is None:
                    raise ParseError("Data outside section", fname, lineno, oline)
            else:
                curr_section.vals[name] = val

    if curr_section is not None:
        yield curr_section


def process_cycles(sec: FioJobConfig) -> Iterator[FioJobConfig]:
    cycles = OrderedDict()  # type: Dict[str, Any]

    for name, val in sec.vals.items():
        if isinstance(val, list) and name.upper() != name:
            cycles[name] = val

    if len(cycles) == 0:
        yield sec
    else:
        # iodepth should changes faster
        numjobs = cycles.pop('iodepth', None)
        items = list(cycles.items())

        if items:
            keys, vals = zip(*items)
            keys = list(keys)
            vals = list(vals)
        else:
            keys = []
            vals = []

        if numjobs is not None:
            vals.append(numjobs)
            keys.append('iodepth')

        for combination in itertools.product(*vals):
            new_sec = sec.copy()
            new_sec.vals.update(zip(keys, combination))
            yield new_sec


FioParamsVal = Union[str, Var]
FioParams = Dict[str, FioParamsVal]


def apply_params(sec: FioJobConfig, params: FioParams) -> FioJobConfig:
    processed_vals = OrderedDict()  # type: Dict[str, Any]
    processed_vals.update(params)
    for name, val in sec.vals.items():
        if name in params:
            continue

        if isinstance(val, Var):
            if val.name in params:
                val = params[val.name]
            elif val.name in processed_vals:
                val = processed_vals[val.name]
        processed_vals[name] = val

    sec = sec.copy()
    sec.vals = processed_vals
    return sec


def abbv_name_to_full(name: str) -> str:
    assert len(name) == 3

    smode = {
        'a': 'async',
        's': 'sync',
        'd': 'direct',
        'x': 'sync direct'
    }
    off_mode = {'s': 'sequential', 'r': 'random'}
    oper = {'r': 'read', 'w': 'write', 'm': 'mixed'}
    return smode[name[2]] + " " + \
        off_mode[name[0]] + " " + oper[name[1]]


MAGIC_OFFSET = 0.1885


def final_process(sec: FioJobConfig, counter: List[int] = [0]) -> FioJobConfig:
    sec = sec.copy()

    sec.vals['unified_rw_reporting'] = '1'

    if isinstance(sec.vals['size'], Var):
        raise ValueError("Variable {0} isn't provided".format(
            sec.vals['size'].name))

    sz = ssize2b(sec.vals['size'])
    offset = sz * ((MAGIC_OFFSET * counter[0]) % 1.0)
    offset = int(offset) // 1024 ** 2
    new_vars = {'UNIQ_OFFSET': str(offset) + "m"}

    for name, val in sec.vals.items():
        if isinstance(val, Var):
            if val.name in new_vars:
                sec.vals[name] = new_vars[val.name]

    for vl in sec.vals.values():
        if isinstance(vl, Var):
            raise ValueError("Variable {0} isn't provided".format(vl.name))

    params = sec.vals.copy()
    params['UNIQ'] = 'UN{0}'.format(counter[0])
    params['COUNTER'] = str(counter[0])
    params['TEST_SUMM'] = sec.summary
    sec.name = sec.name.format(**params)
    counter[0] += 1

    return sec


def execution_time(sec: FioJobConfig) -> int:
    return sec.vals.get('ramp_time', 0) + sec.vals.get('runtime', 0)


def parse_all_in_1(source:str, fname: str = None) -> Iterator[FioJobConfig]:
    return fio_config_parse(fio_config_lexer(source, fname))


def get_log_files(sec: FioJobConfig, iops: bool = False) -> Iterator[Tuple[str, str, str]]:
    res = []  # type: List[Tuple[str, str, str]]

    keys = [('write_bw_log', 'bw', 'KiBps'),
            ('write_hist_log', 'lat', 'us')]
    if iops:
        keys.append(('write_iops_log', 'iops', 'IOPS'))

    for key, name, units in keys:
        log = sec.vals.get(key)
        if log is not None:
            yield (name, log, units)


def fio_cfg_compile(source: str, fname: str, test_params: FioParams) -> Iterator[FioJobConfig]:
    it = parse_all_in_1(source, fname)
    it = (apply_params(sec, test_params) for sec in it)
    it = flatmap(process_cycles, it)
    for sec in map(final_process, it):
        yield sec


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run fio' and return result")
    parser.add_argument("-p", "--params", nargs="*", metavar="PARAM=VAL",
                        default=[],
                        help="Provide set of pairs PARAM=VAL to" +
                             "format into job description")
    parser.add_argument("action", choices=['estimate', 'compile', 'num_tests'])
    parser.add_argument("jobfile")
    return parser.parse_args(argv)


def main(argv):
    argv_obj = parse_args(argv)

    if argv_obj.jobfile == '-':
        job_cfg = sys.stdin.read()
    else:
        job_cfg = open(argv_obj.jobfile).read()

    params = {}
    for param_val in argv_obj.params:
        assert '=' in param_val
        name, val = param_val.split("=", 1)
        params[name] = parse_value(val)

    sec_it = fio_cfg_compile(job_cfg, argv_obj.jobfile, params)

    if argv_obj.action == 'estimate':
        print(sec_to_str(sum(map(execution_time, sec_it))))
    elif argv_obj.action == 'num_tests':
        print(sum(map(len, map(list, sec_it))))
    elif argv_obj.action == 'compile':
        splitter = "\n#" + "-" * 70 + "\n\n"
        print(splitter.join(map(str, sec_it)))

    return 0


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
