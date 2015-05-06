import getpass

from wally.ssh_utils import parse_ssh_uri


class Node(object):

    def __init__(self, conn_url, roles):
        self.roles = roles
        self.conn_url = conn_url
        self.connection = None
        self.monitor_ip = None

    def get_ip(self):
        if self.conn_url == 'local':
            return '127.0.0.1'

        assert self.conn_url.startswith("ssh://")
        return parse_ssh_uri(self.conn_url[6:]).host

    def get_conn_id(self):
        if self.conn_url == 'local':
            return '127.0.0.1'

        assert self.conn_url.startswith("ssh://")
        creds = parse_ssh_uri(self.conn_url[6:])
        return "{0.host}:{0.port}".format(creds)

    def get_user(self):
        if self.conn_url == 'local':
            return getpass.getuser()

        assert self.conn_url.startswith("ssh://")
        creds = parse_ssh_uri(self.conn_url[6:])
        return creds.user

    def __str__(self):
        templ = "<Node: url={conn_url!r} roles={roles}" + \
                " connected={is_connected}>"
        return templ.format(conn_url=self.conn_url,
                            roles=", ".join(self.roles),
                            is_connected=self.connection is not None)

    def __repr__(self):
        return str(self)
