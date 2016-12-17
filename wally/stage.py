import abc
from typing import Optional

from .test_run_class import TestRun
from .config import ConfigBlock


class StepOrder:
    DISCOVER = 0
    SPAWN = 10
    CONNECT = 20
    START_SENSORS = 30
    TEST = 40
    COLLECT_SENSORS = 50
    REPORT = 60


class Stage(metaclass=abc.ABCMeta):
    priority = None  # type: int
    config_block = None  # type: Optional[str]

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

