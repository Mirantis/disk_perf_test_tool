import argparse
import imp
import os.path
import shutil
import sqlite3
import sys

from os import remove
from web_app.app import db
from config import DATABASE_URI, SQLALCHEMY_MIGRATE_REPO, basedir
from migrate.versioning import api


ACTIONS = {}


def action(act):
    def wrap(f):
        ACTIONS[act] = f

        def inner(*args, **kwargs):
            return f(*args, **kwargs)
        return inner
    return wrap


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Manage DB")
    parser.add_argument("action",
                        choices=["dropdb", "createdb", "migrate", "downgrade"])
    return parser.parse_args(argv)


@action("createdb")
def createdb():
    sqlite3.connect(os.path.join(basedir, 'app.db'))

    db.create_all()
    if not os.path.exists(SQLALCHEMY_MIGRATE_REPO):
        api.create(SQLALCHEMY_MIGRATE_REPO, 'database repository')
        api.version_control(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)
    else:
        api.version_control(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO,
                            api.version(SQLALCHEMY_MIGRATE_REPO))


@action("dropdb")
def dropdb():
    db.create_all()
    if os.path.exists(SQLALCHEMY_MIGRATE_REPO):
        shutil.rmtree(SQLALCHEMY_MIGRATE_REPO)

    db.drop_all()
    if os.path.exists(os.path.join(basedir, 'app.db')):
        remove(os.path.join(basedir, 'app.db'))


@action("migrate")
def migrate():
    v = api.db_version(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)
    migration = SQLALCHEMY_MIGRATE_REPO + ('/versions/%03d_migration.py' %
                                           (v+1))
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


@action("upgrade")
def upgrade():
    api.upgrade(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)
    v = api.db_version(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)
    print('Current database version: ' + str(v))


@action("downgrade")
def downgrade():
    v = api.db_version(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)
    api.downgrade(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO, v - 1)
    v = api.db_version(DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)
    print('Current database version: ' + str(v))


def main(argv):
    opts = parse_args(argv)
    func = ACTIONS.get(opts.action)
    func()


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
