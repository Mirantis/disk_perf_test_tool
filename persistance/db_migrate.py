import imp
from migrate.versioning import api
from config import SQLALCHEMY_MIGRATE_REPO, DATABASE_URI
from web_app.app import db


v = api.db_version(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)
migration = SQLALCHEMY_MIGRATE_REPO + ('/versions/%03d_migration.py' % (v+1))
tmp_module = imp.new_module('old_model')
old_model = api.create_model(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)

exec old_model in tmp_module.__dict__
script = api.make_update_script_for_model(DATABASE_URI,
                                          SQLALCHEMY_MIGRATE_REPO,
                                          tmp_module.meta, db.metadata)
open(migration, "wt").write(script)
api.upgrade(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)
v = api.db_version(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)
print('New migration saved as ' + migration)
print('Current database version: ' + str(v))