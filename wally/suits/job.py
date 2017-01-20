import abc
from typing import Dict, Any, Tuple
from collections import OrderedDict

from ..common_types import Storable


class JobParams(metaclass=abc.ABCMeta):
    """Class contains all job parameters, which significantly affects job results.
    Like block size or operation type, but not file name or file size.
    Can be used as key in dictionary
    """

    def __init__(self, **params: Dict[str, Any]) -> None:
        self.params = params

    @abc.abstractproperty
    def summary(self) -> str:
        """Test short summary, used mostly for file names and short image description"""
        pass

    @abc.abstractproperty
    def long_summary(self) -> str:
        """Readable long summary for management and deployment engineers"""
        pass

    def __getitem__(self, name: str) -> Any:
        return self.params[name]

    def __setitem__(self, name: str, val: Any) -> None:
        self.params[name] = val

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.params.items())))

    def __eq__(self, o: 'JobParams') -> bool:
        return sorted(self.params.items()) == sorted(o.params.items())


class JobConfig(Storable, metaclass=abc.ABCMeta):
    """Job config class"""

    def __init__(self, idx: int) -> None:
        # job id, used in storage to distinct jobs with same summary
        self.idx = idx

        # time interval, in seconds, when test was running on all nodes
        self.reliable_info_time_range = None  # type: Tuple[int, int]

        # all job parameters, both from suite file and config file
        self.vals = OrderedDict()  # type: Dict[str, Any]

    @property
    def storage_id(self) -> str:
        """unique string, used as key in storage"""
        return "{}_{}".format(self.params.summary, self.idx)

    @abc.abstractproperty
    def params(self) -> JobParams:
        """Should return a copy"""
        pass
