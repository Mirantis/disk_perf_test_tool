import sqlite3
from migrate.versioning import api
from config import DATABASE_URI, basedir
from config import SQLALCHEMY_MIGRATE_REPO
from web_app.app import db

import os.path


sqlite3.connect(os.path.join(basedir, 'app.db'))

db.create_all()
if not os.path.exists(SQLALCHEMY_MIGRATE_REPO):
    api.create(SQLALCHEMY_MIGRATE_REPO, 'database repository')
    api.version_control(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)
else:
    api.version_control(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO,
                        api.version(SQLALCHEMY_MIGRATE_REPO))