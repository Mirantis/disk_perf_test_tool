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

import pretty_yaml

cfg_dict = {}


class NoData(object):
    @classmethod
    def get(cls, name, x):
        return cls


class Config(object):
    def get(self, name, defval=None):
        obj = self.__dict__
        for cname in name.split("."):
            obj = obj.get(cname, NoData)

        if obj is NoData:
            return defval
        return obj


cfg = Config()
cfg.__dict__ = cfg_dict


def mkdirs_if_unxists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_test_files(results_dir):
    in_var_dir = functools.partial(os.path.join, results_dir)

    res = dict(
        run_params_file='run_params.yaml',
        saved_config_file='config.yaml',
        vm_ids_fname='os_vm_ids',
        html_report_file='{0}_report.html',
        load_report_file='load_report.html',
        text_report_file='report.txt',
        log_file='log.txt',
        sensor_storage='sensor_storage',
        nodes_report_file='nodes.yaml',
        results='results',
        hwinfo_directory='hwinfo',
        hwreport_fname='hwinfo.txt',
        raw_results='raw_results.yaml')

    res = dict((k, in_var_dir(v)) for k, v in res.items())
    res['var_dir'] = results_dir
    return res


def load_config(file_name, explicit_folder=None):
    cfg_dict.update(yaml.load(open(file_name).read()))

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
        if not os.path.isdir(explicit_folder):
            ex2 = os.path.join(var_dir, explicit_folder)
            if os.path.isdir(ex2):
                explicit_folder = ex2
            else:
                raise RuntimeError("No such directory " + explicit_folder)

        results_dir = explicit_folder

    cfg_dict.update(get_test_files(results_dir))
    mkdirs_if_unxists(cfg_dict['var_dir'])

    if explicit_folder is not None:
        cfg_dict.update(load_run_params(cfg_dict['run_params_file']))
        run_uuid = cfg_dict['run_uuid']

    mkdirs_if_unxists(cfg_dict['sensor_storage'])

    if 'sensors_remote_path' not in cfg_dict:
        cfg_dict['sensors_remote_path'] = '/tmp/sensors'

    testnode_log_root = cfg_dict.get('testnode_log_root', '/var/wally')
    testnode_log_dir = os.path.join(testnode_log_root, "{0}/{{name}}")
    cfg_dict['default_test_local_folder'] = \
        testnode_log_dir.format(cfg_dict['run_uuid'])

    mkdirs_if_unxists(cfg_dict['results'])
    mkdirs_if_unxists(cfg_dict['hwinfo_directory'])

    return results_dir


def save_run_params():
    params = {
        'comment': cfg_dict['comment'],
        'run_uuid': cfg_dict['run_uuid']
    }
    with open(cfg_dict['run_params_file'], 'w') as fd:
        fd.write(pretty_yaml.dumps(params))


def load_run_params(run_params_file):
    with open(run_params_file) as fd:
        dt = yaml.load(fd)

    return dict(run_uuid=dt['run_uuid'],
                comment=dt.get('comment'))


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

        # super doesn't work here in 2.6 O_o
        res = logging.Formatter.format(self, record)

        # res = super(ColoredFormatter, self).format(record)

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
    else:
        fh = None

    logger_api.addHandler(sh)
    logger_api.setLevel(logging.WARNING)

    logger = logging.getLogger('paramiko')
    logger.setLevel(logging.WARNING)
    # logger.addHandler(sh)
    if fh is not None:
        logger.addHandler(fh)
