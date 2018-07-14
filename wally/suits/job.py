import abc
from typing import Dict, Any, Tuple, cast, Union, NamedTuple
from collections import OrderedDict

from cephlib.istorage import Storable


Var = NamedTuple('Var', [('name', str)])


class JobParams(metaclass=abc.ABCMeta):
    """Class contains all job parameters, which significantly affects job results.
    Like block size or operation type, but not file name or file size.
    Can be used as key in dictionary
    """

    def __init__(self, **params: Dict[str, Any]) -> None:
        self.params = params

    @property
    @abc.abstractmethod
    def summary(self) -> str:
        """Test short summary, used mostly for file names and short image description"""
        pass

    @property
    @abc.abstractmethod
    def long_summary(self) -> str:
        """Readable long summary for management and deployment engineers"""
        pass

    @abc.abstractmethod
    def copy(self, **updated) -> 'JobParams':
        pass

    def __getitem__(self, name: str) -> Any:
        return self.params[name]

    def __setitem__(self, name: str, val: Any) -> None:
        self.params[name] = val

    def __hash__(self) -> int:
        return hash(self.char_tpl)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, self.__class__):
            raise TypeError(f"Can't compare {self.__class__.__qualname__!r} to {type(o).__qualname__!r}")
        return sorted(self.params.items()) == sorted(cast(JobParams, o).params.items())

    def __lt__(self, o: object) -> bool:
        if not isinstance(o, self.__class__):
            raise TypeError(f"Can't compare {self.__class__.__qualname__!r} to {type(o).__qualname__!r}")
        return self.char_tpl < cast(JobParams, o).char_tpl

    @property
    @abc.abstractmethod
    def char_tpl(self) -> Tuple[Union[str, int, float, bool], ...]:
        pass


class JobConfig(Storable, metaclass=abc.ABCMeta):
    """Job config class"""

    def __init__(self, idx: int) -> None:
        # job id, used in storage to distinct jobs with same summary
        self.idx = idx

        # time interval, in seconds, when test was running on all nodes
        self.reliable_info_range: Tuple[int, int] = None  # type: ignore

        # all job parameters, both from suite file and config file
        self.vals: Dict[str, Any] = OrderedDict()

    @property
    def reliable_info_range_s(self) -> Tuple[int, int]:
        return (self.reliable_info_range[0] // 1000, self.reliable_info_range[1] // 1000)

    @property
    def storage_id(self) -> str:
        """unique string, used as key in storage"""
        return f"{self.summary}_{self.idx}"

    @property
    @abc.abstractmethod
    def params(self) -> JobParams:
        """Should return a copy"""
        pass

    @property
    def summary(self) -> str:
        return self.params.summary
