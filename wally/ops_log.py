from typing import Any


log = []


def log_op(name: str, *params: Any) -> None:
    log.append([name] + list(params))

