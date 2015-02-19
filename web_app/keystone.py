from functools import partial
import json
import urllib2
from keystoneclient.v2_0 import Client as keystoneclient
from keystoneclient import exceptions


class Urllib2HTTP(object):
    """
    class for making HTTP requests
    """

    allowed_methods = ('get', 'put', 'post', 'delete', 'patch', 'head')

    def __init__(self, root_url, headers=None, echo=False):
        """
        """
        if root_url.endswith('/'):
            self.root_url = root_url[:-1]
        else:
            self.root_url = root_url

        self.headers = headers if headers is not None else {}
        self.echo = echo

    def do(self, method, path, params=None):
        if path.startswith('/'):
            url = self.root_url + path
        else:
            url = self.root_url + '/' + path

        if method == 'get':
            assert params == {} or params is None
            data_json = None
        else:
            data_json = json.dumps(params)

        request = urllib2.Request(url,
                                  data=data_json,
                                  headers=self.headers)
        if data_json is not None:
            request.add_header('Content-Type', 'application/json')

        request.get_method = lambda: method.upper()
        response = urllib2.urlopen(request)

        if response.code < 200 or response.code > 209:
            raise IndexError(url)

        content = response.read()

        if '' == content:
            return None

        return json.loads(content)

    def __getattr__(self, name):
        if name in self.allowed_methods:
            return partial(self.do, name)
        raise AttributeError(name)


class KeystoneAuth(Urllib2HTTP):
    def __init__(self, root_url, creds, headers=None, echo=False,
                 admin_node_ip=None):
        super(KeystoneAuth, self).__init__(root_url, headers, echo)
        self.keystone_url = "http://{0}:5000/v2.0".format(admin_node_ip)
        self.keystone = keystoneclient(
            auth_url=self.keystone_url, **creds)
        self.refresh_token()

    def refresh_token(self):
        """Get new token from keystone and update headers"""
        try:
            self.keystone.authenticate()
            self.headers['X-Auth-Token'] = self.keystone.auth_token
        except exceptions.AuthorizationFailure:
            print 'asdfsf'

    def do(self, method, path, params=None):
        """Do request. If gets 401 refresh token"""
        try:
            return super(KeystoneAuth, self).do(method, path, params)
        except urllib2.HTTPError as e:
            if e.code == 401:
                self.refresh_token()
                return super(KeystoneAuth, self).do(method, path, params)
            else:
                raise
