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


class NoData(object):
    @classmethod
    def get(cls, name, x):
        return cls


class Config(object):
    def __init__(self, val=None):
        if val is not None:
            self.update(val)

    def get(self, name, defval=None):
        obj = self.__dict__
        for cname in name.split("."):
            obj = obj.get(cname, NoData)

        if obj is NoData:
            return defval
        return obj

    def update(self, val):
        self.__dict__.update(val)


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
        results_storage='results',
        hwinfo_directory='hwinfo',
        hwreport_fname='hwinfo.txt',
        raw_results='raw_results.yaml')

    res = dict((k, in_var_dir(v)) for k, v in res.items())
    res['results_dir'] = results_dir
    return res


def load_config(file_name):
    file_name = os.path.abspath(file_name)

    defaults = dict(
        sensors_remote_path='/tmp/sensors',
        testnode_log_root='/tmp/wally',
        settings={}
    )

    raw_cfg = yaml.load(open(file_name).read())
    raw_cfg['config_folder'] = os.path.dirname(file_name)
    if 'include' in raw_cfg:
        default_path = os.path.join(raw_cfg['config_folder'],
                                    raw_cfg.pop('include'))
        default_cfg = yaml.load(open(default_path).read())

        # TODO: Need more intelectual configs merge?
        default_cfg.update(raw_cfg)
        raw_cfg = default_cfg

    cfg = Config(defaults)
    cfg.update(raw_cfg)

    results_storage = cfg.settings.get('results_storage', '/tmp')
    print results_storage
    results_storage = os.path.abspath(results_storage)

    existing = file_name.startswith(results_storage)

    if existing:
        cfg.results_dir = os.path.dirname(file_name)
        cfg.run_uuid = os.path.basename(cfg.results_dir)
    else:
        # genarate result folder name
        for i in range(10):
            cfg.run_uuid = pet_generate(2, "_")
            cfg.results_dir = os.path.join(results_storage,
                                           cfg.run_uuid)
            if not os.path.exists(cfg.results_dir):
                break
        else:
            cfg.run_uuid = str(uuid.uuid4())
            cfg.results_dir = os.path.join(results_storage,
                                           cfg.run_uuid)

    # setup all files paths
    cfg.update(get_test_files(cfg.results_dir))

    if existing:
        cfg.update(load_run_params(cfg.run_params_file))

    testnode_log_root = cfg.get('testnode_log_root')
    testnode_log_dir = os.path.join(testnode_log_root, "{0}/{{name}}")
    cfg.default_test_local_folder = testnode_log_dir.format(cfg.run_uuid)

    return cfg


def save_run_params(cfg):
    params = {
        'comment': cfg.comment,
        'run_uuid': cfg.run_uuid
    }

    with open(cfg.run_params_file, 'w') as fd:
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
