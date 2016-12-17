import os
import contextlib
from unittest.mock import patch
from typing import Iterator


from wally import ssh_utils, ssh, node, node_interfaces


creds = "root@osd-0"


def test_ssh_url_parser():
    default_user = "default_user"

    creds = [
        ("test", ssh_utils.ConnCreds("test", default_user, port=22)),
        ("test:13", ssh_utils.ConnCreds("test", default_user, port=13)),
        ("test::xxx.key", ssh_utils.ConnCreds("test", default_user, port=22, key_file="xxx.key")),
        ("test:123:xxx.key", ssh_utils.ConnCreds("test", default_user, port=123, key_file="xxx.key")),
        ("user@test", ssh_utils.ConnCreds("test", "user", port=22)),
        ("user@test:13", ssh_utils.ConnCreds("test", "user", port=13)),
        ("user@test::xxx.key", ssh_utils.ConnCreds("test", "user", port=22, key_file="xxx.key")),
        ("user@test:123:xxx.key", ssh_utils.ConnCreds("test", "user", port=123, key_file="xxx.key")),
        ("user:passwd@test", ssh_utils.ConnCreds("test", "user", port=22, passwd="passwd")),
        ("user:passwd:@test", ssh_utils.ConnCreds("test", "user", port=22, passwd="passwd:")),
        ("user:passwd:@test:123", ssh_utils.ConnCreds("test", "user", port=123, passwd="passwd:"))
    ]

    for uri, expected in creds:
        with patch('getpass.getuser', lambda : default_user):
            parsed = ssh_utils.parse_ssh_uri(uri)

        assert parsed.user == expected.user, uri
        assert parsed.addr.port == expected.addr.port, uri
        assert parsed.addr.host == expected.addr.host, uri
        assert parsed.key_file == expected.key_file, uri
        assert parsed.passwd == expected.passwd, uri


CONNECT_URI = "localhost"


@contextlib.contextmanager
def conn_ctx(uri, *args):
    creds = ssh_utils.parse_ssh_uri(CONNECT_URI)
    node_info = node_interfaces.NodeInfo(creds, set())
    conn = node.connect(node_info, *args)
    try:
        yield conn
    finally:
        conn.disconnect()


def test_ssh_connect():
    with conn_ctx(CONNECT_URI) as conn:
        assert set(conn.run("ls -1 /").split()) == set(fname for fname in os.listdir("/") if not fname.startswith('.'))


def test_ssh_complex():
    pass


def test_file_copy():
    data1 = b"-" * 1024
    data2 = b"+" * 1024

    with conn_ctx(CONNECT_URI) as conn:
        path = conn.put_to_file(None, data1)
        assert data1 == open(path, 'rb').read()

        assert path == conn.put_to_file(path, data2)
        assert data2 == open(path, 'rb').read()

        assert len(data2) > 10
        assert path == conn.put_to_file(path, data2[10:])
        assert data2[10:] == open(path, 'rb').read()
