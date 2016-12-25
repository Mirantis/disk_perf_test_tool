#!/usr/bin/env python3

import re
import os
import sys
import copy
import os.path
import argparse
import itertools
from typing import Optional, Iterator, Union, Dict, Iterable, List, TypeVar, Callable, Tuple, NamedTuple, Any
from collections import OrderedDict


from ...result_classes import IStorable
from ..itest import IterationConfig
from ...utils import sec_to_str, ssize2b


SECTION = 0
SETTING = 1
INCLUDE = 2


Var = NamedTuple('Var', [('name', str)])
CfgLine = NamedTuple('CfgLine',
                     [('fname', str),
                      ('lineno', int),
                      ('oline', str),
                      ('tp', int),
                      ('name', str),
                      ('val', Any)])

TestSumm = NamedTuple("TestSumm",
                      [("oper", str),
                       ("mode", str),
                       ("bsize", int),
                       ("iodepth", int),
                       ("vm_count", int)])


class FioJobSection(IterationConfig, IStorable):
    yaml_tag = 'fio_job'

    def __init__(self, name: str) -> None:
        self.name = name
        self.vals = OrderedDict()  # type: Dict[str, Any]
        self.summary = None

    def copy(self) -> 'FioJobSection':
        return copy.deepcopy(self)

    def required_vars(self) -> Iterator[Tuple[str, Var]]:
        for name, val in self.vals.items():
            if isinstance(val, Var):
                yield name, val

    def is_free(self) -> bool:
        return len(list(self.required_vars())) == 0

    def __str__(self) -> str:
        res = "[{0}]\n".format(self.name)

        for name, val in self.vals.items():
            if name.startswith('_') or name == name.upper():
                continue
            if isinstance(val, Var):
                res += "{0}={{{1}}}\n".format(name, val.name)
            else:
                res += "{0}={1}\n".format(name, val)

        return res

    def raw(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'vals': list(map(list, self.vals.items())),
            'summary': self.summary
        }

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'FioJobSection':
        obj = cls(data['name'])
        obj.summary = data['summary']
        obj.vals.update(data['vals'])
        return obj


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


def fio_config_parse(lexer_iter: Iterable[CfgLine]) -> Iterator[FioJobSection]:
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
                    msg = "Error while including file {0}: {1}"
                    raise ParseError(msg.format(new_fname, err),
                                     fname, lineno, oline)

                new_lines.extend(fio_config_lexer(cont, new_fname))
                one_more = True
            else:
                new_lines.append(line)

        lexed_lines = new_lines

    for fname, lineno, oline, tp, name, val in lexed_lines:
        if tp == SECTION:
            if curr_section is not None:
                yield curr_section
                curr_section = None

            if name == 'global':
                if sections_count != 0:
                    raise ParseError("[global] section should" +
                                     " be only one and first",
                                     fname, lineno, oline)
                in_globals = True
            else:
                in_globals = False
                curr_section = FioJobSection(name)
                curr_section.vals = glob_vals.copy()
            sections_count += 1
        else:
            assert tp == SETTING
            if in_globals:
                glob_vals[name] = val
            elif name == name.upper():
                raise ParseError("Param '" + name +
                                 "' not in [global] section",
                                 fname, lineno, oline)
            elif curr_section is None:
                    raise ParseError("Data outside section",
                                     fname, lineno, oline)
            else:
                curr_section.vals[name] = val

    if curr_section is not None:
        yield curr_section


def process_cycles(sec: FioJobSection) -> Iterator[FioJobSection]:
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


def apply_params(sec: FioJobSection, params: FioParams) -> FioJobSection:
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


def final_process(sec: FioJobSection, counter: List[int] = [0]) -> FioJobSection:
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
    params['TEST_SUMM'] = get_test_summary(sec)
    sec.name = sec.name.format(**params)
    counter[0] += 1

    return sec


def get_test_sync_mode(sec: FioJobSection) -> str:
    if isinstance(sec, dict):
        vals = sec
    else:
        vals = sec.vals

    is_sync = str(vals.get("sync", "0")) == "1"
    is_direct = str(vals.get("direct", "0")) == "1"

    if is_sync and is_direct:
        return 'x'
    elif is_sync:
        return 's'
    elif is_direct:
        return 'd'
    else:
        return 'a'


def get_test_summary_tuple(sec: FioJobSection, vm_count: int = None) -> TestSumm:
    if isinstance(sec, dict):
        vals = sec
    else:
        vals = sec.vals

    rw = {"randread": "rr",
          "randwrite": "rw",
          "read": "sr",
          "write": "sw",
          "randrw": "rm",
          "rw": "sm",
          "readwrite": "sm"}[vals["rw"]]

    sync_mode = get_test_sync_mode(sec)

    return TestSumm(rw,
                    sync_mode,
                    vals['blocksize'],
                    vals.get('iodepth', '1'),
                    vm_count)


def get_test_summary(sec: FioJobSection, vm_count: int = None, noiodepth: bool = False) -> str:
    tpl = get_test_summary_tuple(sec, vm_count)

    res = "{0.oper}{0.mode}{0.bsize}".format(tpl)
    if not noiodepth:
        res += "qd{}".format(tpl.iodepth)

    if tpl.vm_count is not None:
        res += "vm{}".format(tpl.vm_count)

    return res


def execution_time(sec: FioJobSection) -> int:
    return sec.vals.get('ramp_time', 0) + sec.vals.get('runtime', 0)


def parse_all_in_1(source:str, fname: str = None) -> Iterator[FioJobSection]:
    return fio_config_parse(fio_config_lexer(source, fname))


FM_FUNC_INPUT = TypeVar("FM_FUNC_INPUT")
FM_FUNC_RES = TypeVar("FM_FUNC_RES")


def flatmap(func: Callable[[FM_FUNC_INPUT], Iterable[FM_FUNC_RES]],
            inp_iter: Iterable[FM_FUNC_INPUT]) -> Iterator[FM_FUNC_RES]:
    for val in inp_iter:
        for res in func(val):
            yield res


def get_log_files(sec: FioJobSection) -> List[Tuple[str, str]]:
    res = []  # type: List[Tuple[str, str]]
    for key, name in (('write_iops_log', 'iops'), ('write_bw_log', 'bw'), ('write_hist_log', 'lat')):
        log = sec.vals.get(key)
        if log is not None:
            res.append((name, log))
    return res


def fio_cfg_compile(source: str, fname: str, test_params: FioParams) -> Iterator[FioJobSection]:
    it = parse_all_in_1(source, fname)
    it = (apply_params(sec, test_params) for sec in it)
    it = flatmap(process_cycles, it)
    for sec in map(final_process, it):
        sec.summary = get_test_summary(sec)
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
