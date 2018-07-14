from typing import Any, Dict, Optional, Set

from cephlib.storage import IStorable


ConfigBlock = Dict[str, Any]


class Config(IStorable):
    def __init__(self, dct: ConfigBlock) -> None:
        # make mypy happy, set fake dict
        self.__dict__['_dct'] = {}
        self.run_uuid: str = None  # type: ignore
        self.storage_url: str = None  # type: ignore
        self.comment: str = None  # type: ignore
        self.keep_vm: bool = None  # type: ignore
        self.dont_discover_nodes: bool = None  # type: ignore
        self.build_id: str = None  # type: ignore
        self.build_description: str = None  # type: ignore
        self.build_type: str = None  # type: ignore
        self.default_test_local_folder: str = None  # type: ignore
        self.settings_dir: str = None  # type: ignore
        self.connect_timeout: int = None  # type: ignore
        self.no_tests: bool = False
        self.debug_agents: bool = False

        self.logging: 'Config' = None  # type: ignore
        self.ceph: 'Config' = None  # type: ignore
        self.openstack: 'Config' = None  # type: ignore
        self.test: 'Config' = None  # type: ignore
        self.sensors: 'Config' = None  # type: ignore

        # None, disabled, enabled, metadata, ignore_errors
        self.discover: Set[str] = None  # type: ignore

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
