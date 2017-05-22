import copy
from collections import OrderedDict
from typing import Optional, Iterator, Union, Dict, Tuple, NamedTuple, Any, cast

from cephlib.units import ssize2b, b2ssize

from ..job import JobConfig, JobParams


Var = NamedTuple('Var', [('name', str)])


def is_fio_opt_true(vl: Union[str, int]) -> bool:
    return str(vl).lower() in ['1', 'true', 't', 'yes', 'y']


class FioJobParams(JobParams):
    """Class contains all parameters, which significantly affects fio results.

        oper - operation type - read/write/randread/...
        sync_mode - direct/sync/async/direct+sync
        bsize - block size in KiB
        qd - IO queue depth,
        thcount - thread count,
        write_perc - write perc for mixed(read+write) loads

    Like block size or operation type, but not file name or file size.
    Can be used as key in dictionary.
    """

    sync2long = {'x': "sync direct",
                 's': "sync",
                 'd': "direct",
                 'a': "buffered"}

    @property
    def sync_mode_long(self) -> str:
        return self.sync2long[self['sync_mode']]

    @property
    def summary(self) -> str:
        """Test short summary, used mostly for file names and short image description"""
        res = "{0[oper_short]}{0[sync_mode]}{0[bsize]}".format(self)
        if self['qd'] is not None:
            res += "_qd" + str(self['qd'])
        if self['thcount'] not in (1, None):
            res += "th" + str(self['thcount'])
        if self['write_perc'] is not None:
            res += "wr" + str(self['write_perc'])
        return res

    @property
    def long_summary(self) -> str:
        """Readable long summary for management and deployment engineers"""
        res = "{0[oper]}, {0.sync_mode_long}, block size {1}B".format(self, b2ssize(self['bsize'] * 1024))
        if self['qd'] is not None:
            res += ", QD = " + str(self['qd'])
        if self['thcount'] not in (1, None):
            res += ", threads={0[thcount]}".format(self)
        if self['write_perc'] is not None:
            res += ", write_perc={0[write_perc]}%".format(self)
        return res

    def copy(self, **kwargs: Dict[str, Any]) -> 'FioJobParams':
        np = self.params.copy()
        np.update(kwargs)
        return self.__class__(**np)

    @property
    def char_tpl(self) -> Tuple[Union[str, int], ...]:
        mint = lambda x: -10000000000 if x is None else int(x)
        return self['oper'], mint(self['bsize']), self['sync_mode'], \
            mint(self['thcount']), mint(self['qd']), mint(self['write_perc'])


class FioJobConfig(JobConfig):
    """Fio job configuration"""
    ds2mode = {(True, True): 'x',
               (True, False): 's',
               (False, True): 'd',
               (False, False): 'a'}

    op_type2short = {"randread": "rr",
                     "randwrite": "rw",
                     "read": "sr",
                     "write": "sw",
                     "randrw": "rx"}

    def __init__(self, name: str, idx: int) -> None:
        JobConfig.__init__(self, idx)
        self.name = name
        self._sync_mode = None  # type: Optional[str]
        self._params = None  # type: Optional[Dict[str, Any]]

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
        return int(self.vals.get('iodepth', '1'))

    @property
    def bsize(self) -> int:
        bsize = ssize2b(self.vals['blocksize'])
        assert bsize % 1024 == 0
        return bsize // 1024

    @property
    def oper(self) -> str:
        vl = self.vals['rw']
        return vl if ':' not in vl else vl.split(":")[0]

    @property
    def op_type_short(self) -> str:
        return self.op_type2short[self.oper]

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

    # ----------- COMPLEX PROPERTIES -----------------------------------------------------------------------------------

    @property
    def params(self) -> JobParams:
        if self._params is None:
            self._params = dict(oper=self.oper,
                                oper_short=self.op_type_short,
                                sync_mode=self.sync_mode,
                                bsize=self.bsize,
                                qd=self.qd,
                                thcount=self.thcount,
                                write_perc=self.write_perc)
        return cast(JobParams, FioJobParams(**cast(Dict[str, Any], self._params)))

    # ------------------------------------------------------------------------------------------------------------------

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, FioJobConfig):
            return False
        return dict(self.vals) == dict(cast(FioJobConfig, o).vals)

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
        res = super().raw()
        res['vals'] = list(map(list, self.vals.items()))
        return res

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'FioJobConfig':
        data['vals'] = OrderedDict(data['vals'])
        data['_sync_mode'] = None
        data['_params'] = None
        return cast(FioJobConfig, super().fromraw(data))
