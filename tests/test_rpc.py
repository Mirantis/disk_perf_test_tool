import contextlib

from wally import ssh_utils, node, node_interfaces


CONNECT_URI = "localhost"


@contextlib.contextmanager
def rpc_conn_ctx(uri, log_level=None):
    creds = ssh_utils.parse_ssh_uri(uri)
    rpc_code, modules = node.get_rpc_server_code()

    ssh_conn = node.connect(node_interfaces.NodeInfo(creds, set()))
    try:
        rpc_conn = node.setup_rpc(ssh_conn, rpc_code, plugins=modules, log_level=log_level)
        try:
            yield rpc_conn
        finally:
            rpc_conn.conn.server.stop()
            rpc_conn.disconnect()
    finally:
        ssh_conn.disconnect()


def test_rpc_simple():
    with rpc_conn_ctx(CONNECT_URI) as conn:
        names = conn.conn.server.rpc_info()
        assert 'server.list_modules' in names
        assert 'server.load_module' in names
        assert 'server.rpc_info' in names
        assert 'server.stop' in names


def test_rpc_plugins():
    with rpc_conn_ctx(CONNECT_URI) as conn:
        print(conn.conn.server.rpc_info())
        assert conn.conn.fs.file_exists("/")
