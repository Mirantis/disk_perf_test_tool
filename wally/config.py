from typing import Any, Dict, Optional, Set

from .common_types import IStorable


ConfigBlock = Dict[str, Any]


class Config(IStorable):
    def __init__(self, dct: ConfigBlock) -> None:
        # make mypy happy, set fake dict
        self.__dict__['_dct'] = {}
        self.run_uuid = None  # type: str
        self.storage_url = None  # type: str
        self.comment = None  # type: str
        self.keep_vm = None  # type: bool
        self.dont_discover_nodes = None  # type: bool
        self.build_id = None  # type: str
        self.build_description = None  # type: str
        self.build_type = None  # type: str
        self.default_test_local_folder = None  # type: str
        self.settings_dir = None  # type: str
        self.connect_timeout = None  # type: int
        self.no_tests = False  # type: bool
        self.debug_agents = False  # type: bool

        # None, disabled, enabled, metadata, ignore_errors
        self.discover = None  # type: Optional[str]

        self.logging = None  # type: 'Config'
        self.ceph = None  # type: 'Config'
        self.openstack = None  # type: 'Config'
        self.fuel = None  # type: 'Config'
        self.test = None  # type: 'Config'
        self.sensors = None  # type: 'Config'
        self.discover = None  # type: Set[str]

        self._dct.clear()
        self._dct.update(dct)

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'Config':
        return cls(data)

    def raw(self) -> Dict[str, Any]:
        return self._dct

    def get(self, path: str, default: Any = None) -> Any:
        curr = self
        while path:
            if '/' in path:
                name, path = path.split('/', 1)
            else:
                name = path
                path = ""

            try:
                curr = getattr(curr, name)
            except AttributeError:
                return default

        return curr

    def __getattr__(self, name: str) -> Any:
        try:
            val = self._dct[name]
        except KeyError:
            raise AttributeError(name)

        if isinstance(val, dict):
            val = self.__class__(val)

        return val

    def __setattr__(self, name: str, val: Any):
        self._dct[name] = val

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None
