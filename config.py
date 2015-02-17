import os

basedir = os.environ.get('DATABASE_FOLDER', os.path.abspath(os.path.dirname(__file__)))
DOCUMENT_ID = "1Xvd0aHA7mr-_b5C3b-sQ66BQsJiOGIT2UesP7kG26oU"
SHEET_NAME = "aaa"
WORK_SHEET = "Worksheet"
COL_COUNT = 2
ROW_COUNT = 10
DEFAULT_FILE_PATH = "test.json"
OUTPUT_FILE = "output.json"
TEST_PATH = os.environ.get("TEST_PATH", os.path.dirname(__file__) + "/test_results")
CHARTS_IMG_PATH = "static/images"
SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')

if os.environ.get('DATABASE_URL') is None:
    DATABASE_URI = ('sqlite:///' + os.path.join(basedir, 'app.db') +
                               '?check_same_thread=False')
else:
    DATABASE_URI = os.environ['DATABASE_URL']