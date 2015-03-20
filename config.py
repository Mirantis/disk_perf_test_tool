import yaml
import os


def parse_config(file_name):
    with open(file_name) as f:
        cfg = yaml.load(f.read())

    return cfg


parser = argparse.ArgumentParser(description="config file name")
parser.add_argument("-p", "--path")
parser.add_argument("-b", "--basedir")
parser.add_argument("-t", "--testpath")
parser.add_argument("-d", "--database")
parser.add_argument("-c", "--chartpath")

config = parser.parse_args(sys.argv[1:])
path = "config.yaml"

if not config.path is None:
    path = config.path

cfg_dict = parse_config(os.path.join(os.path.dirname(__file__), path))
basedir = cfg_dict['paths']['basedir']
TEST_PATH = cfg_dict['paths']['TEST_PATH']
SQLALCHEMY_MIGRATE_REPO = cfg_dict['paths']['SQLALCHEMY_MIGRATE_REPO']
DATABASE_URI = cfg_dict['paths']['DATABASE_URI']
CHARTS_IMG_PATH = cfg_dict['paths']['CHARTS_IMG_PATH']

if not config.basedir is None:
    basedir = config.basedir

if not config.testpath is None:
    TEST_PATH = config.testpath

if not config.database is None:
    DATABASE_URI = config.database

if not config.chartpath is None:
    CHARTS_IMG_PATH = config.chartpath


