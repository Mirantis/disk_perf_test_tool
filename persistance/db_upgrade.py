from migrate.versioning import api
from config import DATABASE_URI
from config import SQLALCHEMY_MIGRATE_REPO
api.upgrade(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)
v = api.db_version(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)
print('Current database version: ' + str(v))