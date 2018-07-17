import abc
from typing import Optional

from .test_run_class import TestRun
from .config import ConfigBlock


class StepOrder:
    DISCOVER = 0
    SPAWN = 10
    CONNECT = 20
    UPDATE_NODES_INFO = 30
    START_SENSORS = 40
    TEST = 50
    COLLECT_SENSORS = 60
    STOP_SENSORS = 70
    REPORT = 80


class Stage(metaclass=abc.ABCMeta):
    priority: int = None  # type: ignore
    config_block: Optional[str] = None

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    def validate_config(cls, cfg: ConfigBlock) -> None:
        pass

    @abc.abstractmethod
    def run(self, ctx: TestRun) -> None:
        pass

    def cleanup(self, ctx: TestRun) -> None:
        pass

