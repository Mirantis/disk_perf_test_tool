import os
import uuid
import logging
import functools

import yaml

try:
    from petname import Generate as pet_generate
except ImportError:
    def pet_generate(x, y):
        return str(uuid.uuid4())


cfg_dict = {}


def mkdirs_if_unxists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_config(file_name, explicit_folder=None):
    first_load = len(cfg_dict) == 0
    cfg_dict.update(yaml.load(open(file_name).read()))

    if first_load:
        var_dir = cfg_dict.get('internal', {}).get('var_dir_root', '/tmp')

    run_uuid = None
    if explicit_folder is None:
        for i in range(10):
            run_uuid = pet_generate(2, "_")
            results_dir = os.path.join(var_dir, run_uuid)
            if not os.path.exists(results_dir):
                break
        else:
            run_uuid = str(uuid.uuid4())
            results_dir = os.path.join(var_dir, run_uuid)
        cfg_dict['run_uuid'] = run_uuid.replace('_', '-')
    else:
        results_dir = explicit_folder

    cfg_dict['var_dir'] = results_dir
    mkdirs_if_unxists(cfg_dict['var_dir'])

    in_var_dir = functools.partial(os.path.join, cfg_dict['var_dir'])

    cfg_dict['charts_img_path'] = in_var_dir('charts')
    mkdirs_if_unxists(cfg_dict['charts_img_path'])

    cfg_dict['vm_ids_fname'] = in_var_dir('os_vm_ids')
    cfg_dict['html_report_file'] = in_var_dir('report.html')
    cfg_dict['text_report_file'] = in_var_dir('report.txt')
    cfg_dict['log_file'] = in_var_dir('log.txt')
    cfg_dict['sensor_storage'] = in_var_dir('sensor_storage.txt')

    cfg_dict['test_log_directory'] = in_var_dir('test_logs')
    mkdirs_if_unxists(cfg_dict['test_log_directory'])


def color_me(color):
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

    def __init__(self, msg, use_color=True, datefmt=None):
        logging.Formatter.__init__(self, msg, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record):
        orig = record.__dict__
        record.__dict__ = record.__dict__.copy()
        levelname = record.levelname

        prn_name = levelname + ' ' * (8 - len(levelname))
        if levelname in self.colors:
            record.levelname = self.colors[levelname](prn_name)
        else:
            record.levelname = prn_name

        res = super(ColoredFormatter, self).format(record)

        # restore record, as it will be used by other formatters
        record.__dict__ = orig
        return res


def setup_loggers(def_level=logging.DEBUG, log_fname=None):
    logger = logging.getLogger('wally')
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(def_level)

    log_format = '%(asctime)s - %(levelname)s - %(name)-15s - %(message)s'
    colored_formatter = ColoredFormatter(log_format, datefmt="%H:%M:%S")

    sh.setFormatter(colored_formatter)
    logger.addHandler(sh)

    logger_api = logging.getLogger("wally.fuel_api")

    if log_fname is not None:
        fh = logging.FileHandler(log_fname)
        log_format = '%(asctime)s - %(levelname)8s - %(name)-15s - %(message)s'
        formatter = logging.Formatter(log_format, datefmt="%H:%M:%S")
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger_api.addHandler(fh)

    logger_api.addHandler(sh)
    logger_api.setLevel(logging.WARNING)
