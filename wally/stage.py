import logging
import contextlib
from typing import Callable

from .utils import StopTestError
from .test_run_class import TestRun


logger = logging.getLogger("wally")


@contextlib.contextmanager
def log_stage(stage) -> None:
    msg_templ = "Exception during {0}: {1!s}"
    msg_templ_no_exc = "During {0}"

    logger.info("Start " + stage.name)

    try:
        yield
    except StopTestError as exc:
        logger.error(msg_templ.format(stage.__name__, exc))
    except Exception:
        logger.exception(msg_templ_no_exc.format(stage.__name__))


StageType = Callable[[TestRun], None]
