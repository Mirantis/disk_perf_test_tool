import re
import abc
import json
import logging
import urllib.parse
import urllib.error
import urllib.request
from typing import Dict, Any, Iterator, List, Callable, cast
from functools import partial

import netaddr
from keystoneclient import exceptions
from keystoneclient.v2_0 import Client as keystoneclient


logger = logging.getLogger("wally")


class Connection(metaclass=abc.ABCMeta):
    host = None  # type: str

    @abc.abstractmethod
    def do(self, method: str, path: str, params: Dict = None) -> Dict:
        pass

    def get(self, path: str, params: Dict = None) -> Dict:
        return self.do("GET", path, params)


class Urllib2HTTP(Connection):
    """
    class for making HTTP requests
    """

    allowed_methods = ('get', 'put', 'post', 'delete', 'patch', 'head')

    def __init__(self, root_url: str, headers: Dict[str, str] = None) -> None:
        """
        """
        if root_url.endswith('/'):
            self.root_url = root_url[:-1]
        else:
            self.root_url = root_url

        self.host = urllib.parse.urlparse(self.root_url).hostname

        if headers is None:
            self.headers = {}  # type: Dict[str, str]
        else:
            self.headers  = headers

    def do(self, method: str, path: str, params: Dict = None) -> Any:
        if path.startswith('/'):
            url = self.root_url + path
        else:
            url = self.root_url + '/' + path

        if method == 'get':
            assert params == {} or params is None
            data_json = None
        else:
            data_json = json.dumps(params)

        logger.debug("HTTP: {0} {1}".format(method.upper(), url))

        request = urllib.request.Request(url,
                                         data=data_json.encode("utf8"),
                                         headers=self.headers)
        if data_json is not None:
            request.add_header('Content-Type', 'application/json')

        request.get_method = lambda: method.upper()  # type: ignore
        response = urllib.request.urlopen(request)
        code = response.code  # type: ignore

        logger.debug("HTTP Responce: {0}".format(code))
        if code < 200 or code > 209:
            raise IndexError(url)

        content = response.read()

        if '' == content:
            return None

        return json.loads(content.decode("utf8"))

    def __getattr__(self, name: str) -> Any:
        if name in self.allowed_methods:
            return partial(self.do, name)
        raise AttributeError(name)


class KeystoneAuth(Urllib2HTTP):
    def __init__(self, root_url: str, creds: Dict[str, str], headers: Dict[str, str] = None) -> None:
        super(KeystoneAuth, self).__init__(root_url, headers)
        admin_node_ip = urllib.parse.urlparse(root_url).hostname
        self.keystone_url = "http://{0}:5000/v2.0".format(admin_node_ip)
        self.keystone = keystoneclient(
            auth_url=self.keystone_url, **creds)
        self.refresh_token()

    def refresh_token(self) -> None:
        """Get new token from keystone and update headers"""
        try:
            self.keystone.authenticate()
            self.headers['X-Auth-Token'] = self.keystone.auth_token
        except exceptions.AuthorizationFailure:
            logger.warning(
                'Cant establish connection to keystone with url %s',
                self.keystone_url)

    def do(self, method: str, path: str, params: Dict[str, str] = None) -> Any:
        """Do request. If gets 401 refresh token"""
        try:
            return super(KeystoneAuth, self).do(method, path, params)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                logger.warning(
                    'Authorization failure: {0}'.format(e.read()))
                self.refresh_token()
                return super(KeystoneAuth, self).do(method, path, params)
            else:
                raise


def get_inline_param_list(url: str) -> Iterator[str]:
    format_param_rr = re.compile(r"\{([a-zA-Z_]+)\}")
    for match in format_param_rr.finditer(url):
        yield match.group(1)


class RestObj:
    name = None  # type: str
    id = None   # type: int

    def __init__(self, conn: Connection, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
        self.__connection__ = conn

    def __str__(self) -> str:
        res = ["{0}({1}):".format(self.__class__.__name__, self.name)]
        for k, v in sorted(self.__dict__.items()):
            if k.startswith('__') or k.endswith('__'):
                continue
            if k != 'name':
                res.append("    {0}={1!r}".format(k, v))
        return "\n".join(res)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def make_call(method: str, url: str) -> Callable:
    def closure(obj: Any, entire_obj: Any = None, **data: Any) -> Any:
        inline_params_vals = {}
        for name in get_inline_param_list(url):
            if name in data:
                inline_params_vals[name] = data[name]
                del data[name]
            else:
                inline_params_vals[name] = getattr(obj, name)
        result_url = url.format(**inline_params_vals)

        if entire_obj is not None:
            if data != {}:
                raise ValueError("Both entire_obj and data provided")
            data = entire_obj
        return obj.__connection__.do(method, result_url, params=data)
    return closure


RequestMethod = Callable[[str], Callable]


PUT = cast(RequestMethod, partial(make_call, 'put'))  # type: RequestMethod
GET = cast(RequestMethod, partial(make_call, 'get'))  # type: RequestMethod
DELETE = cast(RequestMethod, partial(make_call, 'delete')) # type: RequestMethod

# -------------------------------  ORM ----------------------------------------


def get_fuel_info(url: str) -> 'FuelInfo':
    conn = Urllib2HTTP(url)
    return FuelInfo(conn)


class FuelInfo(RestObj):

    """Class represents Fuel installation info"""

    get_nodes = GET('api/nodes')
    get_clusters = GET('api/clusters')
    get_cluster = GET('api/clusters/{id}')
    get_info = GET('api/releases')

    @property
    def nodes(self) -> 'NodeList':
        """Get all fuel nodes"""
        return NodeList([Node(self.__connection__, **node) for node
                         in self.get_nodes()])

    @property
    def free_nodes(self) -> 'NodeList':
        """Get unallocated nodes"""
        return NodeList([Node(self.__connection__, **node) for node in
                         self.get_nodes() if not node['cluster']])

    @property
    def clusters(self) -> List['Cluster']:
        """List clusters in fuel"""
        return [Cluster(self.__connection__, **cluster) for cluster
                in self.get_clusters()]

    def get_version(self) -> List[int]:
        for info in self.get_info():
            vers = info['version'].split("-")[1].split('.')
            return list(map(int, vers))
        raise ValueError("No version found")


class Node(RestObj):
    """Represents node in Fuel"""

    get_info = GET('/api/nodes/{id}')
    get_interfaces = GET('/api/nodes/{id}/interfaces')

    def get_network_data(self) -> Dict:
        """Returns node network data"""
        return self.get_info().get('network_data')

    def get_roles(self) -> List[str]:
        """Get node roles

        Returns: (roles, pending_roles)
        """
        return self.get_info().get('roles')

    def get_ip(self, network='public') -> netaddr.IPAddress:
        """Get node ip

        :param network: network to pick
        """
        nets = self.get_network_data()
        for net in nets:
            if net['name'] == network:
                iface_name = net['dev']
                for iface in self.get_info()['meta']['interfaces']:
                    if iface['name'] == iface_name:
                        try:
                            return iface['ip']
                        except KeyError:
                            return netaddr.IPNetwork(net['ip']).ip
        raise Exception('Network %s not found' % network)


class NodeList(list):
    """Class for filtering nodes through attributes"""
    allowed_roles = ['controller', 'compute', 'cinder', 'ceph-osd', 'mongo',
                     'zabbix-server']

    def __getattr__(self, name: str) -> List[Node]:
        if name in self.allowed_roles:
            return [node for node in self if name in node.roles]


class Cluster(RestObj):
    """Class represents Cluster in Fuel"""

    get_status = GET('api/clusters/{id}')
    get_networks = GET('api/clusters/{id}/network_configuration/neutron')
    get_attributes = GET('api/clusters/{id}/attributes')
    _get_nodes = GET('api/nodes?cluster_id={id}')

    def __init__(self, *dt, **mp) -> None:
        super(Cluster, self).__init__(*dt, **mp)
        self.nodes = NodeList([Node(self.__connection__, **node) for node in
                               self._get_nodes()])

    def check_exists(self) -> bool:
        """Check if cluster exists"""
        try:
            self.get_status()
            return True
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return False
            raise

    def get_openrc(self) -> Dict[str, str]:
        access = self.get_attributes()['editable']['access']
        creds = {'username': access['user']['value'],
                 'password': access['password']['value'],
                 'tenant_name': access['tenant']['value']}

        version = FuelInfo(self.__connection__).get_version()
        # only HTTPS since 7.0
        if version >= [7, 0]:
            creds['insecure'] = "True"
            creds['os_auth_url'] = "https://{0}:5000/v2.0".format(
                self.get_networks()['public_vip'])
        else:
            creds['os_auth_url'] = "http://{0}:5000/v2.0".format(
                self.get_networks()['public_vip'])
        return creds

    def get_nodes(self) -> Iterator[Node]:
        for node_descr in self._get_nodes():
            yield Node(self.__connection__, **node_descr)


def reflect_cluster(conn: Connection, cluster_id: int) -> Cluster:
    """Returns cluster object by id"""
    c = Cluster(conn, id=cluster_id)
    c.nodes = NodeList(list(c.get_nodes()))
    return c


def get_all_nodes(conn: Connection) -> Iterator[Node]:
    """Get all nodes from Fuel"""
    for node_desc in conn.get('api/nodes'):
        yield Node(conn, **node_desc)


def get_all_clusters(conn: Connection) -> Iterator[Cluster]:
    """Get all clusters"""
    for cluster_desc in conn.get('api/clusters'):
        yield Cluster(conn, **cluster_desc)


def get_cluster_id(conn: Connection, name: str) -> int:
    """Get cluster id by name"""
    for cluster in get_all_clusters(conn):
        if cluster.name == name:
            return cluster.id

    raise ValueError("Cluster {0} not found".format(name))

