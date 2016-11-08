from typing import Any, Dict
from .storage import IStorable, IStorage

class NoData:
    @classmethod
    def get(cls: type, name: str, x: Any) -> type:
        return cls


class Config(IStorable):
    # for mypy only
    run_uuid = None  # type: str
    storage_url = None  # type: str
    comment = None  # type: str
    keep_vm = None  # type: bool
    no_tests = None  # type: bool
    dont_discover_nodes = None  # type: bool
    build_id = None  # type: str
    build_description = None  # type: str
    build_type = None  # type: str

    def __init__(self, dct: Dict[str, Any]) -> None:
        self.__dict__['_dct'] = dct

    def get(self, path: str, default: Any = NoData) -> Any:
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
            val = self.__dct[name]
        except KeyError:
            raise AttributeError(name)

        if isinstance(val, dict):
            val = self.__class__(val)

        return val

    def __setattr__(self, name: str, val: Any):
        self.__dct[name] = val


class Context:
    def __init__(self, config: Config, storage: IStorage):
        self.config = config
        self.storage = storage