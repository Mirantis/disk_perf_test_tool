import logging
from typing import Callable, IO, Optional


def color_me(color: int) -> Callable[[str], str]:
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"

    color_seq = COLOR_SEQ % (30 + color)

    def closure(msg):
        return color_seq + msg + RESET_SEQ
    return closure


class ColoredFormatter(logging.Formatter):
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

    colors = {
        'WARNING': color_me(YELLOW),
        'DEBUG': color_me(BLUE),
        'CRITICAL': color_me(YELLOW),
        'ERROR': color_me(RED)
    }

    def __init__(self, msg: str, use_color: bool=True, datefmt: str=None) -> None:
        logging.Formatter.__init__(self, msg, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        orig = record.__dict__
        record.__dict__ = record.__dict__.copy()
        levelname = record.levelname

        prn_name = levelname + ' ' * (8 - len(levelname))
        if levelname in self.colors:
            record.levelname = self.colors[levelname](prn_name)
        else:
            record.levelname = prn_name

        # super doesn't work here in 2.6 O_o
        res = logging.Formatter.format(self, record)

        # res = super(ColoredFormatter, self).format(record)

        # restore record, as it will be used by other formatters
        record.__dict__ = orig
        return res


def setup_loggers(def_level: int = logging.DEBUG, log_fname: str = None, log_fd: IO = None) -> None:

    log_format = '%(asctime)s - %(levelname)s - %(name)-15s - %(message)s'
    colored_formatter = ColoredFormatter(log_format, datefmt="%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setLevel(def_level)
    sh.setFormatter(colored_formatter)

    logger = logging.getLogger('wally')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    logger_api = logging.getLogger("wally.fuel_api")
    logger_api.setLevel(logging.WARNING)
    logger_api.addHandler(sh)

    if log_fname or log_fd:
        if log_fname:
            handler = logging.FileHandler(log_fname)  # type: Optional[logging.Handler]
        else:
            handler = logging.StreamHandler(log_fd)

        log_format = '%(asctime)s - %(levelname)8s - %(name)-15s - %(message)s'
        formatter = logging.Formatter(log_format, datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)

        logger.addHandler(handler)
        logger_api.addHandler(handler)

    logging.getLogger('paramiko').setLevel(logging.WARNING)
