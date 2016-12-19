import re
import os
import stat
import time
import os.path
import logging
import tempfile
import subprocess
import urllib.request
from typing import Dict, Any, Iterable, Iterator, NamedTuple, Optional, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor

from keystoneauth1 import loading, session
from novaclient.exceptions import NotFound
from novaclient.client import Client as NovaClient
from cinderclient.client import Client as CinderClient
from glanceclient import Client as GlanceClient

from .utils import Timeout, to_ip
from .node_interfaces import NodeInfo
from .ssh_utils import ConnCreds


__doc__ = """
Module used to reliably spawn set of VM's, evenly distributed across
compute servers in openstack cluster. Main functions:

    get_openstack_credentials - extract openstack credentials from different sources
    os_connect - connect to nova, cinder and glance API
    find_vms - find VM's with given prefix in name
    prepare_os - prepare tenant for usage
    launch_vms - reliably start set of VM in parallel with volumes and floating IP
    clear_nodes - clear VM and volumes
"""


logger = logging.getLogger("wally")


OSCreds = NamedTuple("OSCreds",
                     [("name", str),
                      ("passwd", str),
                      ("tenant", str),
                      ("auth_url", str),
                      ("insecure", bool)])


# TODO(koder): should correctly process different sources, not only env????
def get_openstack_credentials() -> OSCreds:
    is_insecure = os.environ.get('OS_INSECURE', 'false').lower() in ('true', 'yes')

    return OSCreds(os.environ.get('OS_USERNAME'),
                   os.environ.get('OS_PASSWORD'),
                   os.environ.get('OS_TENANT_NAME'),
                   os.environ.get('OS_AUTH_URL'),
                   is_insecure)


class OSConnection:
    def __init__(self, nova: NovaClient, cinder: CinderClient, glance: GlanceClient) -> None:
        self.nova = nova
        self.cinder = cinder
        self.glance = glance


def os_connect(os_creds: OSCreds, version: str = "2") -> OSConnection:
    loader = loading.get_plugin_loader('password')
    auth = loader.load_from_options(auth_url=os_creds.auth_url,
                                    username=os_creds.name,
                                    password=os_creds.passwd,
                                    project_id=os_creds.tenant)
    auth_sess = session.Session(auth=auth)

    glance = GlanceClient(version, session=auth_sess)
    nova = NovaClient(version, session=auth_sess)
    cinder = CinderClient(os_creds.name, os_creds.passwd, os_creds.tenant, os_creds.auth_url,
                          insecure=os_creds.insecure, api_version=version)
    return OSConnection(nova, cinder, glance)


def find_vms(conn: OSConnection, name_prefix: str) -> Iterable[Tuple[str, int]]:
    for srv in conn.nova.servers.list():
        if srv.name.startswith(name_prefix):
            # need to exit after found server first external IP
            # so have to rollout two cycles to avoid using exceptions
            all_ip = []  # type: List[Any]
            for ips in srv.addresses.values():
                all_ip.extend(ips)

            for ip in all_ip:
                if ip.get("OS-EXT-IPS:type", None) == 'floating':
                    yield ip['addr'], srv.id
                    break


def pause(conn: OSConnection, ids: Iterable[int], executor: ThreadPoolExecutor) -> None:
    def pause_vm(vm_id: str) -> None:
        vm = conn.nova.servers.get(vm_id)
        if vm.status == 'ACTIVE':
            vm.pause()

    for future in executor.map(pause_vm, ids):
        future.result()


def unpause(conn: OSConnection, ids: Iterable[int], executor: ThreadPoolExecutor, max_resume_time=10) -> None:
    def unpause(vm_id: str) -> None:
        vm = conn.nova.servers.get(vm_id)
        if vm.status == 'PAUSED':
            vm.unpause()

        for _ in Timeout(max_resume_time):
            vm = conn.nova.servers.get(vm_id)
            if vm.status != 'PAUSED':
                return
        raise RuntimeError("Can't unpause vm {0}".format(vm_id))

    for future in executor.map(unpause, ids):
        future.result()


def prepare_os(conn: OSConnection, params: Dict[str, Any], max_vm_per_node: int = 8) -> None:
    """prepare openstack for futher usage

    Creates server groups, security rules, keypair, flavor
    and upload VM image from web. In case if object with
    given name already exists, skip preparation part.
    Don't check, that existing object has required attributes

    params:
        nova: OSConnection
        params: dict {
            security_group:str - security group name with allowed ssh and ping
            aa_group_name:str - template for anti-affinity group names. Should
                                receive one integer parameter, like "cbt_aa_{0}"
            keypair_name: str - OS keypair name
            keypair_file_public: str - path to public key file
            keypair_file_private: str - path to private key file

            flavor:dict - flavor params
                name, ram_size, hdd_size, cpu_count
                    as for novaclient.Client.flavor.create call

            image:dict - image params
                'name': image name
                'url': image url
        }
        os_creds: OSCreds
        max_vm_per_compute: int=8 maximum expected amount of VM, per
                            compute host. Used to create appropriate
                            count of server groups for even placement
    """
    allow_ssh_and_ping(conn, params['security_group'])

    for idx in range(max_vm_per_node):
        get_or_create_aa_group(conn, params['aa_group_name'].format(idx))

    create_keypair(conn, params['keypair_name'], params['keypair_file_public'], params['keypair_file_private'])
    create_image(conn, params['image']['name'], params['image']['url'])
    create_flavor(conn, **params['flavor'])


def create_keypair(conn: OSConnection, name: str, pub_key_path: str, priv_key_path: str):
    """create and upload keypair into nova, if doesn't exists yet

    Create and upload keypair into nova, if keypair with given bane
    doesn't exists yet. Uses key from files, if file doesn't exists -
    create new keys, and store'em into files.

    parameters:
        conn: OSConnection
        name: str - ketpair name
        pub_key_path: str - path for public key
        priv_key_path: str - path for private key
    """

    pub_key_exists = os.path.exists(pub_key_path)
    priv_key_exists = os.path.exists(priv_key_path)

    try:
        kpair = conn.nova.keypairs.find(name=name)
        # if file not found- delete and recreate
    except NotFound:
        kpair = None

    if pub_key_exists and not priv_key_exists:
        raise EnvironmentError("Private key file doesn't exists")

    if not pub_key_exists and priv_key_exists:
        raise EnvironmentError("Public key file doesn't exists")

    if kpair is None:
        if pub_key_exists:
            with open(pub_key_path) as pub_key_fd:
                return conn.nova.keypairs.create(name, pub_key_fd.read())
        else:
            key = conn.nova.keypairs.create(name)

            with open(priv_key_path, "w") as priv_key_fd:
                priv_key_fd.write(key.private_key)
            os.chmod(priv_key_path, stat.S_IREAD | stat.S_IWRITE)

            with open(pub_key_path, "w") as pub_key_fd:
                pub_key_fd.write(key.public_key)
    elif not priv_key_exists:
        raise EnvironmentError("Private key file doesn't exists," +
                               " but key uploaded openstack." +
                               " Either set correct path to private key" +
                               " or remove key from openstack")


def get_or_create_aa_group(conn: OSConnection, name: str) -> int:
    """create anti-affinity server group, if doesn't exists yet

    parameters:
        conn: OSConnection
        name: str - group name

    returns: str - group id
    """
    try:
        return conn.nova.server_groups.find(name=name).id
    except NotFound:
        return conn.nova.server_groups.create(name=name, policies=['anti-affinity']).id


def allow_ssh_and_ping(conn: OSConnection, group_name: str) -> int:
    """create sequrity group for ping and ssh

    parameters:
        conn:
        group_name: str - group name

    returns: str - group id
    """
    try:
        secgroup = conn.nova.security_groups.find(name=group_name)
    except NotFound:
        secgroup = conn.nova.security_groups.create(group_name, "allow ssh/ping to node")

        conn.nova.security_group_rules.create(secgroup.id,
                                              ip_protocol="tcp",
                                              from_port="22",
                                              to_port="22",
                                              cidr="0.0.0.0/0")

        conn.nova.security_group_rules.create(secgroup.id,
                                              ip_protocol="icmp",
                                              from_port=-1,
                                              cidr="0.0.0.0/0",
                                              to_port=-1)
    return secgroup.id


def create_image(conn: OSConnection, name: str, url: str) -> None:
    """upload image into glance from given URL, if given image doesn't exisis yet

    parameters:
        nova: nova connection
        os_creds: OSCreds object - openstack credentials, should be same,
                                   as used when connectiong given novaclient
        name: str - image name
        url: str - image download url

    returns: None
    """
    try:
        conn.nova.images.find(name=name)
        return
    except NotFound:
        pass

    ok = False
    with tempfile.NamedTemporaryFile() as temp_fd:
        try:
            cmd = "wget --dns-timeout=30 --connect-timeout=30 --read-timeout=30 -o {} {}"
            subprocess.check_call(cmd.format(temp_fd.name, url))
            ok = True

        # TODO(koder): add proper error handling
        except Exception:
            pass

        if not ok:
            urllib.request.urlretrieve(url, temp_fd.name)

        image = conn.glance.images.create(name=name)
        with open(temp_fd.name, 'rb') as fd:
            conn.glance.images.upload(image.id, fd)


def create_flavor(conn: OSConnection, name: str, ram_size: int, hdd_size: int, cpu_count: int) -> None:
    """create flavor, if doesn't exisis yet

    parameters:
        nova: nova connection
        name: str - flavor name
        ram_size: int - ram size (UNIT?)
        hdd_size: int - root hdd size (UNIT?)
        cpu_count: int - cpu cores

    returns: None
    """
    try:
        conn.nova.flavors.find(name)
        return
    except NotFound:
        pass

    conn.nova.flavors.create(name, cpu_count, ram_size, hdd_size)


def create_volume(conn: OSConnection, size: int, name: str) -> Any:
    vol = conn.cinder.volumes.create(size=size, display_name=name)
    err_count = 0

    while vol.status != 'available':
        if vol.status == 'error':
            if err_count == 3:
                logger.critical("Fail to create volume")
                raise RuntimeError("Fail to create volume")
            else:
                err_count += 1
                conn.cinder.volumes.delete(vol)
                time.sleep(1)
                vol = conn.cinder.volumes.create(size=size, display_name=name)
                continue
        time.sleep(1)
        vol = conn.cinder.volumes.get(vol.id)
    return vol


def wait_for_server_active(conn: OSConnection, server: Any, timeout: int = 300) -> bool:
    """waiting till server became active

    parameters:
        nova: nova connection
        server: server object
        timeout: int - seconds to wait till raise an exception

    returns: None
    """

    for _ in Timeout(timeout, no_exc=True):
        server_state = getattr(server, 'OS-EXT-STS:vm_state').lower()

        if server_state == 'active':
            return True

        if server_state == 'error':
            return False

        server = conn.nova.servers.get(server)
    return False


class Allocate(object):
    pass


def get_floating_ips(conn: OSConnection, pool: Optional[str], amount: int) -> List[str]:
    """allocate floating ips

    parameters:
        nova: nova connection
        pool:str floating ip pool name
        amount:int - ip count

    returns: [ip object]
    """
    ip_list = conn.nova.floating_ips.list()

    if pool is not None:
        ip_list = [ip for ip in ip_list if ip.pool == pool]

    return [ip for ip in ip_list if ip.instance_id is None][:amount]


def launch_vms(conn: OSConnection,
               params: Dict[str, Any],
               executor: ThreadPoolExecutor,
               already_has_count: int = 0) -> Iterator[NodeInfo]:
    """launch virtual servers

    Parameters:
        nova: nova client
        params: dict {
            count: str or int - server count. If count is string it should be in
                                one of bext forms: "=INT" or "xINT". First mean
                                to spawn (INT - already_has_count) servers, and
                                all should be evenly distributed across all compute
                                nodes. xINT mean spawn COMPUTE_COUNT * INT servers.
            image: dict {'name': str - image name}
            flavor: dict {'name': str - flavor name}
            group_name: str - group name, used to create uniq server name
            keypair_name: str - ssh keypais name
            keypair_file_private: str - path to private key
            user: str - vm user name
            vol_sz: int or None - volume size, or None, if no volume
            network_zone_name: str - network zone name
            flt_ip_pool: str - floating ip pool
            name_templ: str - server name template, should receive two parameters
                              'group and id, like 'cbt-{group}-{id}'
            aa_group_name: str scheduler group name
            security_group: str - security group name
        }
        already_has_count: int=0 - how many servers already exists. Used to distribute
                                   new servers evenly across all compute nodes, taking
                                   old server in accout
    returns: generator of NodeInfo - server credentials, in format USER@IP:KEY_PATH

    """
    logger.debug("Calculating new vm count")
    count = params['count']  # type: int
    lst = conn.nova.services.list(binary='nova-compute')
    srv_count = len([srv for srv in lst if srv.status == 'enabled'])

    if isinstance(count, str):
        if count.startswith("x"):
            count = srv_count * int(count[1:])
        else:
            assert count.startswith('=')
            count = int(count[1:]) - already_has_count

    if count <= 0:
        logger.debug("Not need new vms")
        return

    logger.debug("Starting new nodes on openstack")

    assert isinstance(count, int)

    srv_params = "img: {image[name]}, flavor: {flavor[name]}".format(**params)
    msg_templ = "Will start {0} servers with next params: {1}"
    logger.info(msg_templ.format(count, srv_params))

    vm_params = dict(
        img_name=params['image']['name'],
        flavor_name=params['flavor']['name'],
        group_name=params['group_name'],
        keypair_name=params['keypair_name'],
        vol_sz=params.get('vol_sz'),
        network_zone_name=params.get("network_zone_name"),
        flt_ip_pool=params.get('flt_ip_pool'),
        name_templ=params.get('name_templ'),
        scheduler_hints={"group": params['aa_group_name']},
        security_group=params['security_group'],
        sec_group_size=srv_count
    )

    # precache all errors before start creating vms
    private_key_path = params['keypair_file_private']
    user = params['image']['user']

    for ip, os_node in create_vms_mt(conn, count, executor, **vm_params):
        info = NodeInfo(ConnCreds(to_ip(ip), user, key_file=private_key_path), set())
        info.os_vm_id = os_node.id
        yield info


def get_free_server_groups(conn: OSConnection, template: str) -> Iterator[str]:
    """get fre server groups, that match given name template

    parameters:
        nova: nova connection
        template:str - name template
        amount:int - ip count

    returns: generator or str - server group names
    """
    for server_group in conn.nova.server_groups.list():
        if not server_group.members:
            if re.match(template, server_group.name):
                yield str(server_group.id)


def create_vms_mt(conn: OSConnection,
                  amount: int,
                  executor: ThreadPoolExecutor,
                  group_name: str,
                  keypair_name: str,
                  img_name: str,
                  flavor_name: str,
                  vol_sz: int = None,
                  network_zone_name: str = None,
                  flt_ip_pool: str = None,
                  name_templ: str ='wally-{id}',
                  scheduler_hints: Dict = None,
                  security_group: str = None,
                  sec_group_size: int = None) -> List[Tuple[str, Any]]:

    if network_zone_name is not None:
        network_future = executor.submit(conn.nova.networks.find,
                                         label=network_zone_name)
    else:
        network_future = None

    fl_future = executor.submit(conn.nova.flavors.find, name=flavor_name)
    img_future = executor.submit(conn.nova.images.find, name=img_name)

    if flt_ip_pool is not None:
        ips_future = executor.submit(get_floating_ips,
                                     conn, flt_ip_pool, amount)
        logger.debug("Wait for floating ip")
        ips = ips_future.result()
        ips += [Allocate] * (amount - len(ips))
    else:
        ips = [None] * amount

    logger.debug("Getting flavor object")
    fl = fl_future.result()
    logger.debug("Getting image object")
    img = img_future.result()

    if network_future is not None:
        logger.debug("Waiting for network results")
        nics = [{'net-id': network_future.result().id}]
    else:
        nics = None

    names = []  # type: List[str]
    for i in range(amount):
        names.append(name_templ.format(group=group_name, id=i))

    futures = []
    logger.debug("Requesting new vm's")

    orig_scheduler_hints = scheduler_hints.copy()
    group_name_template = scheduler_hints['group'].format("\\d+")
    groups = list(get_free_server_groups(conn, group_name_template + "$"))
    groups.sort()

    for idx, (name, flt_ip) in enumerate(zip(names, ips), 2):

        scheduler_hints = None
        if orig_scheduler_hints is not None and sec_group_size is not None:
            if "group" in orig_scheduler_hints:
                scheduler_hints = orig_scheduler_hints.copy()
                scheduler_hints['group'] = groups[idx // sec_group_size]

        if scheduler_hints is None:
            scheduler_hints = orig_scheduler_hints.copy()

        params = (conn, name, keypair_name, img, fl,
                  nics, vol_sz, flt_ip, scheduler_hints,
                  flt_ip_pool, [security_group])

        futures.append(executor.submit(create_vm, *params))
    res = [future.result() for future in futures]
    logger.debug("Done spawning")
    return res


def create_vm(conn: OSConnection,
              name: str,
              keypair_name: str,
              img: Any,
              flavor: Any,
              nics: List,
              vol_sz: int = None,
              flt_ip: Any = False,
              scheduler_hints: Dict = None,
              pool: str = None,
              security_groups=None,
              max_retry: int = 3,
              delete_timeout: int = 120) -> Tuple[str, Any]:

    # make mypy/pylint happy
    srv = None  # type: Any
    for i in range(max_retry):
        srv = conn.nova.servers.create(name, flavor=flavor, image=img, nics=nics, key_name=keypair_name,
                                       scheduler_hints=scheduler_hints, security_groups=security_groups)

        if not wait_for_server_active(conn, srv):
            logger.debug("Server {} fails to start. Kill it and try again".format(srv))
            conn.nova.servers.delete(srv)

            try:
                for _ in Timeout(delete_timeout, "Server {} delete timeout".format(srv.id)):
                    srv = conn.nova.servers.get(srv.id)
            except NotFound:
                pass
        else:
            break
    else:
        raise RuntimeError("Failed to start server {}".format(srv.id))

    if vol_sz is not None:
        vol = create_volume(conn, vol_sz, name)
        conn.nova.volumes.create_server_volume(srv.id, vol.id, None)

    if flt_ip is Allocate:
        flt_ip = conn.nova.floating_ips.create(pool)

    if flt_ip is not None:
        srv.add_floating_ip(flt_ip)

    # pylint: disable=E1101
    return flt_ip.ip, conn.nova.servers.get(srv.id)


def clear_nodes(conn: OSConnection,
                ids: List[int] = None,
                name_templ: str = None,
                max_server_delete_time: int = 120):
    try:
        def need_delete(srv):
            if name_templ is not None:
                return re.match(name_templ.format("\\d+"), srv.name) is not None
            else:
                return srv.id in ids

        volumes_to_delete = []
        for vol in conn.cinder.volumes.list():
            for attachment in vol.attachments:
                if attachment['server_id'] in ids:
                    volumes_to_delete.append(vol)
                    break

        still_alive = set()
        for srv in conn.nova.servers.list():
            if need_delete(srv):
                logger.debug("Deleting server {0}".format(srv.name))
                conn.nova.servers.delete(srv)
                still_alive.add(srv.id)

        if still_alive:
            logger.debug("Waiting till all servers are actually deleted")
            tout = Timeout(max_server_delete_time, no_exc=True)
            while tout.tick() and still_alive:
                all_id = set(srv.id for srv in conn.nova.servers.list())
                still_alive = still_alive.intersection(all_id)

            if still_alive:
                logger.warning("Failed to remove servers {}. ".format(",".join(still_alive)) +
                               "You, probably, need to remove them manually (and volumes as well)")
                return

        if volumes_to_delete:
            logger.debug("Deleting volumes")

            # wait till vm actually deleted

            # logger.warning("Volume deletion commented out")
            for vol in volumes_to_delete:
                logger.debug("Deleting volume " + vol.display_name)
                conn.cinder.volumes.delete(vol)

        logger.debug("Clearing complete (yet some volumes may still be deleting)")
    except Exception:
        logger.exception("During removing servers. " +
                         "You, probably, need to remove them manually")
