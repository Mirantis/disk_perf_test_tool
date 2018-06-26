from typing import Any, Dict, Optional, Set

from cephlib.storage import IStorable


ConfigBlock = Dict[str, Any]


class Config(IStorable):
    def __init__(self, dct: ConfigBlock) -> None:
        # make mypy happy, set fake dict
        self.__dict__['_dct'] = {}
        self.run_uuid: str = None
        self.storage_url: str = None
        self.comment: str = None
        self.keep_vm: bool = None
        self.dont_discover_nodes: bool = None
        self.build_id: str = None
        self.build_description: str = None
        self.build_type: str = None
        self.default_test_local_folder: str = None
        self.settings_dir: str = None
        self.connect_timeout: int = None
        self.no_tests: bool = False
        self.debug_agents: bool = False

        # None, disabled, enabled, metadata, ignore_errors
        self.discover: Optional[str] = None

        self.logging: 'Config' = None
        self.ceph: 'Config' = None
        self.openstack: 'Config' = None
        self.fuel: 'Config' = None
        self.test: 'Config' = None
        self.sensors: 'Config' = None
        self.discover: Set[str] = None

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
