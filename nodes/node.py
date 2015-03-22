class Node(object):

    def __init__(self, ip, roles, username=None,
                 password=None, key_path=None, port=None):
        self.roles = roles
        self.ip = ip
        self.username = username
        self.password = password
        self.port = port
        self.key_path = key_path
        self.connection = None

    def __str__(self):
        return "<Node: url={0!r} roles={1} >".format(self.ip,
                                                     ", ".join(self.roles))

    def __repr__(self):
        return str(self)

    def set_conn_attr(self, name, value):
        setattr(self, name, value)

    @property
    def connection_url(self):
        connection = []

        if self.username:
            connection.append(self.username)
            if self.password:
                connection.extend([":", self.password, "@"])
            connection.append("@")

        connection.append(self.ip)
        if self.port:
            connection.extend([":", self.port])
            if self.key_path:
                connection.extend([":", self.key_path])
        else:
            if self.key_path:
                connection.extend([":", ":", self.key_path])
        return "".join(connection)
