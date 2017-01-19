#!/usr/bin/env python3

import re
import os
import sys
import copy
import os.path
import argparse
import itertools
from typing import Optional, Iterator, Union, Dict, Iterable, List, TypeVar, Callable, Tuple, NamedTuple, Any, cast
from collections import OrderedDict


from ...result_classes import TestJobConfig
from ...utils import sec_to_str, ssize2b, b2ssize, flatmap


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
FioTestSumm = NamedTuple("FioTestSumm",
                         [("oper", str),
                          ("sync_mode", str),
                          ("bsize", int),
                          ("qd", int),
                          ("thcount", int),
                          ("write_perc", Optional[int])])


def is_fio_opt_true(vl: Union[str, int]) -> bool:
    return str(vl).lower() in ['1', 'true', 't', 'yes', 'y']


class FioJobConfig(TestJobConfig):

    ds2mode = {(True, True): 'x',
               (True, False): 's',
               (False, True): 'd',
               (False, False): 'a'}

    sync2long = {'x': "sync direct",
                 's': "sync",
                 'd': "direct",
                 'a': "buffered"}

    op_type2short = {"randread": "rr",
                     "randwrite": "rw",
                     "read": "sr",
                     "write": "sw",
                     "randrw": "rx"}

    def __init__(self, name: str, idx: int) -> None:
        TestJobConfig.__init__(self, idx)
        self.name = name
        self._sync_mode = None  # type: Optional[str]
        self._ctuple = None  # type: Optional[FioTestSumm]
        self._ctuple_no_qd = None  # type: Optional[FioTestSumm]

    # ------------- BASIC PROPERTIES -----------------------------------------------------------------------------------

    @property
    def write_perc(self) -> Optional[int]:
        try:
            return int(self.vals["rwmixwrite"])
        except (KeyError, TypeError):
            try:
                return 100 - int(self.vals["rwmixread"])
            except (KeyError, TypeError):
                return None

    @property
    def qd(self) -> int:
        return int(self.vals['iodepth'])

    @property
    def bsize(self) -> int:
        return ssize2b(self.vals['blocksize']) // 1024

    @property
    def oper(self) -> str:
        return self.vals['rw']

    @property
    def op_type_short(self) -> str:
        return self.op_type2short[self.vals['rw']]

    @property
    def thcount(self) -> int:
        return int(self.vals.get('numjobs', 1))

    @property
    def sync_mode(self) -> str:
        if self._sync_mode is None:
            direct = is_fio_opt_true(self.vals.get('direct', '0')) or \
                     not is_fio_opt_true(self.vals.get('buffered', '0'))
            sync = is_fio_opt_true(self.vals.get('sync', '0'))
            self._sync_mode = self.ds2mode[(sync, direct)]
        return cast(str, self._sync_mode)

    @property
    def sync_mode_long(self) -> str:
        return self.sync2long[self.sync_mode]

    # ----------- COMPLEX PROPERTIES -----------------------------------------------------------------------------------

    @property
    def characterized_tuple(self) -> Tuple:
        if self._ctuple is None:
            self._ctuple = FioTestSumm(oper=self.oper,
                                       sync_mode=self.sync_mode,
                                       bsize=self.bsize,
                                       qd=self.qd,
                                       thcount=self.thcount,
                                       write_perc=self.write_perc)

        return cast(Tuple, self._ctuple)

    @property
    def characterized_tuple_no_qd(self) -> FioTestSumm:
        if self._ctuple_no_qd is None:
            self._ctuple_no_qd = FioTestSumm(oper=self.oper,
                                             sync_mode=self.sync_mode,
                                             bsize=self.bsize,
                                             qd=None,
                                             thcount=self.thcount,
                                             write_perc=self.write_perc)

        return cast(FioTestSumm, self._ctuple_no_qd)

    @property
    def long_summary(self) -> str:
        res = "{0.sync_mode_long} {0.oper} {1} QD={0.qd}".format(self, b2ssize(self.bsize * 1024))
        if self.thcount != 1:
            res += " threads={}".format(self.thcount)
        if self.write_perc is not None:
            res += " write_perc={}%".format(self.write_perc)
        return res

    @property
    def long_summary_no_qd(self) -> str:
        res = "{0.sync_mode_long} {0.oper} {1}".format(self, b2ssize(self.bsize * 1024))
        if self.thcount != 1:
            res += " threads={}".format(self.thcount)
        if self.write_perc is not None:
            res += " write_perc={}%".format(self.write_perc)
        return res

    @property
    def summary(self) -> str:
        tpl = cast(FioTestSumm, self.characterized_tuple)
        res = "{0.oper}{0.sync_mode}{0.bsize}_qd{0.qd}".format(tpl)

        if tpl.thcount != 1:
            res += "th" + str(tpl.thcount)
        if tpl.write_perc != 1:
            res += "wr" + str(tpl.write_perc)

        return res

    @property
    def summary_no_qd(self) -> str:
        tpl = cast(FioTestSumm, self.characterized_tuple)
        res = "{0.oper}{0.sync_mode}{0.bsize}".format(tpl)

        if tpl.thcount != 1:
            res += "th" + str(tpl.thcount)
        if tpl.write_perc != 1:
            res += "wr" + str(tpl.write_perc)

        return res
    # ------------------------------------------------------------------------------------------------------------------

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, FioJobConfig):
            return False
        return self.vals == cast(FioJobConfig, o).vals

    def copy(self) -> 'FioJobConfig':
        return copy.deepcopy(self)

    def required_vars(self) -> Iterator[Tuple[str, Var]]:
        for name, val in self.vals.items():
            if isinstance(val, Var):
                yield name, val

    def is_free(self) -> bool:
        return len(list(self.required_vars())) == 0

    def __str__(self) -> str:
        res = "[{0}]\n".format(self.summary)

        for name, val in self.vals.items():
            if name.startswith('_') or name == name.upper():
                continue
            if isinstance(val, Var):
                res += "{0}={{{1}}}\n".format(name, val.name)
            else:
                res += "{0}={1}\n".format(name, val)

        return res

    def __repr__(self) -> str:
        return str(self)

    def raw(self) -> Dict[str, Any]:
        res = self.__dict__.copy()
        del res['_sync_mode']
        res['vals'] = [[key, val] for key, val in self.vals.items()]
        return res

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'FioJobConfig':
        obj = cls.__new__(cls)
        data['vals'] = OrderedDict(data['vals'])
        data['_sync_mode'] = None
        obj.__dict__.update(data)
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


def get_log_files(sec: FioJobConfig, iops: bool = False) -> List[Tuple[str, str]]:
    res = []  # type: List[Tuple[str, str]]

    keys = [('write_bw_log', 'bw'), ('write_hist_log', 'lat')]
    if iops:
        keys.append(('write_iops_log', 'iops'))

    for key, name in keys:
        log = sec.vals.get(key)
        if log is not None:
            res.append((name, log))

    return res


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
