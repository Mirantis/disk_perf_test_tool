import urlparse


class Node(object):

    def __init__(self, conn_url, roles):
        self.roles = roles
        self.conn_url = conn_url
        self.connection = None

    def get_ip(self):
        return urlparse.urlparse(self.conn_url).hostname

    def __str__(self):
        templ = "<Node: url={conn_url!r} roles={roles}" + \
                " connected={is_connected}>"
        return templ.format(conn_url=self.conn_url,
                            roles=", ".join(self.roles),
                            is_connected=self.connection is not None)

    def __repr__(self):
        return str(self)
