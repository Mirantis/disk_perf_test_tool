import logging
import contextlib


from .utils import StopTestError

logger = logging.getLogger("wally")


class TestStage:
    name = ""

    def __init__(self, testrun, config):
        self.testrun = testrun
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self):
        return self


@contextlib.contextmanager
def log_stage(stage):
    msg_templ = "Exception during {0}: {1!s}"
    msg_templ_no_exc = "During {0}"

    logger.info("Start " + stage.name)

    try:
        yield
    except StopTestError as exc:
        logger.error(msg_templ.format(stage.name, exc))
    except Exception:
        logger.exception(msg_templ_no_exc.format(stage.name))
