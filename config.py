import yaml
import os


def parse_config(file_name):
    with open(file_name) as f:
        cfg = yaml.load(f.read())

    return cfg


cfg_dict = parse_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))
basedir = cfg_dict['paths']['basedir']
TEST_PATH = cfg_dict['paths']['TEST_PATH']
SQLALCHEMY_MIGRATE_REPO = cfg_dict['paths']['SQLALCHEMY_MIGRATE_REPO']
DATABASE_URI = cfg_dict['paths']['DATABASE_URI']
CHARTS_IMG_PATH = cfg_dict['paths']['CHARTS_IMG_PATH']