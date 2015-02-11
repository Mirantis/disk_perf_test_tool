import sys
import json
import sqlite3
import contextlib


def connect(url):
    return sqlite3.connect(url)


create_db_sql_templ = """
CREATE TABLE build (id integer primary key,
                    build text,
                    type text,
                    md5 text);

CREATE TABLE params_combination (id integer primary key, {params});
CREATE TABLE param (id integer primary key, name text, type text);

CREATE TABLE result (build_id integer,
                     params_combination integer,
                     bandwith float,
                     deviation float);
"""


PARAM_COUNT = 20


def get_all_tables(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return cursor.fetchall()


def drop_database(conn):
    cursor = conn.cursor()
    cursor.execute("drop table result")
    cursor.execute("drop table params_combination")
    cursor.execute("drop table build")
    cursor.execute("drop table param")


def init_database(conn):
    cursor = conn.cursor()

    params = ["param_{0} text".format(i) for i in range(PARAM_COUNT)]
    create_db_sql = create_db_sql_templ.format(params=",".join(params))

    for sql in create_db_sql.split(";"):
        cursor.execute(sql)


def insert_io_params(conn):
    sql = """insert into param (name, type) values ('operation',
                '{write,randwrite,read,randread}');
             insert into param (name, type) values ('sync', '{a,s}');
             insert into param (name, type) values ('block_size', 'size_kmg');
          """

    for insert in sql.split(";"):
        conn.execute(insert)


def insert_build(cursor, build_id, build_type, iso_md5):
    cursor.execute("insert into build (build, type, md5) values (?, ?, ?)",
                   (build_id, build_type, iso_md5))
    return cursor.lastrowid


def insert_params(cursor, *param_vals):
    param_vals = param_vals + ("",) * (PARAM_COUNT - len(param_vals))

    params = ",".join(['?'] * PARAM_COUNT)
    select_templ = "select id from params_combination where {params_where}"

    params_where = ["param_{0}=?".format(i) for i in range(PARAM_COUNT)]
    req = select_templ.format(params_where=" AND ".join(params_where))
    cursor.execute(req, param_vals)
    res = cursor.fetchall()
    if [] != res:
        return res[0][0]

    params = ",".join(['?'] * PARAM_COUNT)
    param_insert_templ = "insert into params_combination ({0}) values ({1})"
    param_names = ",".join("param_{0}".format(i) for i in range(PARAM_COUNT))
    req = param_insert_templ.format(param_names, params)
    cursor.execute(req, param_vals)
    return cursor.lastrowid


def insert_results(cursor, build_id, params_id, bw, dev):
    req = "insert into result values (?, ?, ?, ?)"
    cursor.execute(req, (build_id, params_id, bw, dev))


@contextlib.contextmanager
def transaction(conn):
    try:
        cursor = conn.cursor()
        yield cursor
    except:
        conn.rollback()
        raise
    else:
        conn.commit()


def json_to_db(json_data, conn):
    data = json.loads(json_data)
    with transaction(conn) as cursor:
        for build_data in data:
            build_id = insert_build(cursor,
                                    build_data.pop("build_id"),
                                    build_data.pop("type"),
                                    build_data.pop("iso_md5"))

            for params, (bw, dev) in build_data.items():
                param_id = insert_params(cursor, *params.split(" "))
                insert_results(cursor, build_id, param_id, bw, dev)


def to_db():
    conn = sqlite3.connect(sys.argv[1])
    json_data = open(sys.argv[2]).read()

    if len(get_all_tables(conn)) == 0:
        init_database(conn)

    json_to_db(json_data, conn)


def ssize_to_kb(ssize):
    try:
        smap = dict(k=1, K=1, M=1024, m=1024, G=1024**2, g=1024**2)
        for ext, coef in smap.items():
            if ssize.endswith(ext):
                return int(ssize[:-1]) * coef

        if int(ssize) % 1024 != 0:
            raise ValueError()

        return int(ssize) / 1024

    except (ValueError, TypeError, AttributeError):
        tmpl = "Unknow size format {0!r} (or size not multiples 1024)"
        raise ValueError(tmpl.format(ssize))


def load_slice(cursor, build_id, y_param, **params):
    params_id = {}
    for param in list(params) + [y_param]:
        cursor.execute("select id from param where name=?", (param,))
        params_id[param] = cursor.fetchone()

    sql = """select params_combination.param_{0}, result.bandwith
             from params_combination, result
             where result.build_id=?""".format(params_id[y_param])

    for param, val in params.items():
        pid = params_id[param]
        sql += " and params_combination.param_{0}='{1}'".format(pid, val)

    cursor.execute(sql)


def from_db():
    conn = sqlite3.connect(sys.argv[1])
    # sql = sys.argv[2]
    cursor = conn.cursor()

    sql = """select params_combination.param_2, result.bandwith
    from params_combination, result
    where params_combination.param_0="write"
          and params_combination.param_1="s"
          and params_combination.id=result.params_combination
          and result.build_id=60"""

    cursor.execute(sql)
    data = []

    for (sz, bw) in cursor.fetchall():
        data.append((ssize_to_kb(sz), sz, bw))

    data.sort()

    import matplotlib.pyplot as plt
    xvals = range(len(data))
    plt.plot(xvals, [dt[2] for dt in data])
    plt.ylabel('bandwith')
    plt.xlabel('block size')
    plt.xticks(xvals, [dt[1] for dt in data])
    plt.show()


from_db()
