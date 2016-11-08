from typing import Union, List, Dict, Any, IO


Basic = Union[List, Dict[str, Any]]


def load(stream: IO, loader: Any) -> Any: ...
class CLoader: ...

