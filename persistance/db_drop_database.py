from config import SQLALCHEMY_MIGRATE_REPO, basedir
from web_app.app import db
import shutil

import os.path
from os import remove

db.create_all()
if os.path.exists(SQLALCHEMY_MIGRATE_REPO):
    shutil.rmtree(SQLALCHEMY_MIGRATE_REPO)

db.drop_all()
if os.path.exists(os.path.join(basedir, 'app.db')):
    remove(os.path.join(basedir, 'app.db'))