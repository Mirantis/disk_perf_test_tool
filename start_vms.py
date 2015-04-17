import re
import os
import time
import logging
import subprocess

from concurrent.futures import ThreadPoolExecutor

from novaclient.exceptions import NotFound
from novaclient.client import Client as n_client
from cinderclient.v1.client import Client as c_client

from nodes.node import Node


logger = logging.getLogger("io-perf-tool")


def ostack_get_creds():
    env = os.environ.get
    name = env('OS_USERNAME')
    passwd = env('OS_PASSWORD')
    tenant = env('OS_TENANT_NAME')
    auth_url = env('OS_AUTH_URL')

    return name, passwd, tenant, auth_url


NOVA_CONNECTION = None


def nova_connect(name=None, passwd=None, tenant=None, auth_url=None):
    global NOVA_CONNECTION
    if NOVA_CONNECTION is None:
        if name is None:
            name, passwd, tenant, auth_url = ostack_get_creds()
        NOVA_CONNECTION = n_client('1.1', name, passwd, tenant, auth_url)
    return NOVA_CONNECTION


def nova_disconnect():
    global NOVA_CONNECTION
    if NOVA_CONNECTION is not None:
        NOVA_CONNECTION.close()
        NOVA_CONNECTION = None


def prepare_os_subpr(name=None, passwd=None, tenant=None, auth_url=None):
    if name is None:
        name, passwd, tenant, auth_url = ostack_get_creds()

    params = {
        'OS_USERNAME': name,
        'OS_PASSWORD':  passwd,
        'OS_TENANT_NAME':  tenant,
        'OS_AUTH_URL':  auth_url
    }

    params_s = " ".join("{}={}".format(k, v) for k, v in params.items())

    cmd_templ = "env {params} bash scripts/prepare.sh >/dev/null"
    cmd = cmd_templ.format(params=params_s)
    subprocess.call(cmd, shell=True)


def prepare_os(nova, params):
    allow_ssh(nova, params['security_group'])

    shed_ids = []
    for shed_group in params['schedulers_groups']:
        shed_ids.append(get_or_create_aa_group(nova, shed_group))

    create_keypair(nova,
                   params['keypair_name'],
                   params['pub_key_path'],
                   params['priv_key_path'])

    create_image(nova, params['image']['name'],
                 params['image']['url'])

    create_flavor(nova, **params['flavor'])


def get_or_create_aa_group(nova, name):
    try:
        group = nova.server_groups.find(name=name)
    except NotFound:
        group = nova.server_groups.create({'name': name,
                                           'policies': ['anti-affinity']})

    return group.id


def allow_ssh(nova, group_name):
    try:
        secgroup = nova.security_groups.find(name=group_name)
    except NotFound:
        secgroup = nova.security_groups.create(group_name,
                                               "allow ssh/ping to node")

    nova.security_group_rules.create(secgroup.id,
                                     ip_protocol="tcp",
                                     from_port="22",
                                     to_port="22",
                                     cidr="0.0.0.0/0")

    nova.security_group_rules.create(secgroup.id,
                                     ip_protocol="icmp",
                                     from_port=-1,
                                     cidr="0.0.0.0/0",
                                     to_port=-1)
    return secgroup.id


def create_image(nova, name, url):
    pass


def create_flavor(nova, name, **params):
    pass


def create_keypair(nova, name, pub_key_path, priv_key_path):
    try:
        nova.keypairs.find(name=name)
    except NotFound:
        if os.path.exists(pub_key_path):
            with open(pub_key_path) as pub_key_fd:
                return nova.keypairs.create(name, pub_key_fd.read())
        else:
            key = nova.keypairs.create(name)

            with open(priv_key_path, "w") as priv_key_fd:
                priv_key_fd.write(key.private_key)

            with open(pub_key_path, "w") as pub_key_fd:
                pub_key_fd.write(key.public_key)


def create_volume(size, name):
    cinder = c_client(*ostack_get_creds())
    vol = cinder.volumes.create(size=size, display_name=name)
    err_count = 0
    while vol.status != 'available':
        if vol.status == 'error':
            if err_count == 3:
                logger.critical("Fail to create volume")
                raise RuntimeError("Fail to create volume")
            else:
                err_count += 1
                cinder.volumes.delete(vol)
                time.sleep(1)
                vol = cinder.volumes.create(size=size, display_name=name)
                continue
        time.sleep(1)
        vol = cinder.volumes.get(vol.id)
    return vol


def wait_for_server_active(nova, server, timeout=240):
    t = time.time()
    while True:
        time.sleep(1)
        sstate = getattr(server, 'OS-EXT-STS:vm_state').lower()

        if sstate == 'active':
            return True

        if sstate == 'error':
            return False

        if time.time() - t > timeout:
            return False

        server = nova.servers.get(server)


class Allocate(object):
    pass


def get_floating_ips(nova, pool, amount):
    ip_list = nova.floating_ips.list()

    if pool is not None:
        ip_list = [ip for ip in ip_list if ip.pool == pool]

    return [ip for ip in ip_list if ip.instance_id is None][:amount]


def launch_vms(params):
    logger.debug("Starting new nodes on openstack")
    params = params.copy()
    count = params.pop('count')

    if isinstance(count, basestring):
        assert count.startswith("x")
        lst = NOVA_CONNECTION.services.list(binary='nova-compute')
        srv_count = len([srv for srv in lst if srv.status == 'enabled'])
        count = srv_count * int(count[1:])

    srv_params = "img: {image[name]}, flavor: {flavor[name]}".format(**params)
    msg_templ = "Will start {0} servers with next params: {1}"
    logger.info(msg_templ.format(count, srv_params))
    vm_creds = params.pop('creds')

    params = params.copy()

    params['img_name'] = params['image']['name']
    params['flavor_name'] = params['flavor']['name']

    del params['image']
    del params['flavor']
    del params['scheduler_group_name']
    private_key_path = params.pop('private_key_path')

    for ip, os_node in create_vms_mt(NOVA_CONNECTION, count, **params):
        conn_uri = vm_creds.format(ip=ip, private_key_path=private_key_path)
        yield Node(conn_uri, []), os_node.id


def create_vms_mt(nova, amount, keypair_name, img_name,
                  flavor_name, vol_sz=None, network_zone_name=None,
                  flt_ip_pool=None, name_templ='ceph-test-{0}',
                  scheduler_hints=None, security_group=None):

    with ThreadPoolExecutor(max_workers=16) as executor:
        if network_zone_name is not None:
            network_future = executor.submit(nova.networks.find,
                                             label=network_zone_name)
        else:
            network_future = None

        fl_future = executor.submit(nova.flavors.find, name=flavor_name)
        img_future = executor.submit(nova.images.find, name=img_name)

        if flt_ip_pool is not None:
            ips_future = executor.submit(get_floating_ips,
                                         nova, flt_ip_pool, amount)
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

        names = map(name_templ.format, range(amount))

        futures = []
        logger.debug("Requesting new vms")

        for name, flt_ip in zip(names, ips):
            params = (nova, name, keypair_name, img, fl,
                      nics, vol_sz, flt_ip, scheduler_hints,
                      flt_ip_pool, [security_group])

            futures.append(executor.submit(create_vm, *params))
        res = [future.result() for future in futures]
        logger.debug("Done spawning")
        return res


def create_vm(nova, name, keypair_name, img,
              fl, nics, vol_sz=None,
              flt_ip=False,
              scheduler_hints=None,
              pool=None,
              security_groups=None):
    for i in range(3):
        srv = nova.servers.create(name,
                                  flavor=fl,
                                  image=img,
                                  nics=nics,
                                  key_name=keypair_name,
                                  scheduler_hints=scheduler_hints,
                                  security_groups=security_groups)

        if not wait_for_server_active(nova, srv):
            msg = "Server {0} fails to start. Kill it and try again"
            logger.debug(msg.format(srv))
            nova.servers.delete(srv)

            while True:
                # print "wait till server deleted"
                all_id = set(alive_srv.id for alive_srv in nova.servers.list())
                if srv.id not in all_id:
                    break
                time.sleep(1)
        else:
            break

    if vol_sz is not None:
        # print "creating volume"
        vol = create_volume(vol_sz, name)
        # print "attach volume to server"
        nova.volumes.create_server_volume(srv.id, vol.id, None)

    if flt_ip is Allocate:
        flt_ip = nova.floating_ips.create(pool)

    if flt_ip is not None:
        # print "attaching ip to server"
        srv.add_floating_ip(flt_ip)

    return flt_ip.ip, nova.servers.get(srv.id)


def clear_nodes(nodes_ids):
    clear_all(NOVA_CONNECTION, nodes_ids, None)


def clear_all(nova, ids=None, name_templ="ceph-test-{0}"):

    def need_delete(srv):
        if name_templ is not None:
            return re.match(name_templ.format("\\d+"), srv.name) is not None
        else:
            return srv.id in ids

    deleted_srvs = set()
    for srv in nova.servers.list():
        if need_delete(srv):
            logger.debug("Deleting server {0}".format(srv.name))
            nova.servers.delete(srv)
            deleted_srvs.add(srv.id)

    count = 0
    while True:
        if count % 60 == 0:
            logger.debug("Waiting till all servers are actually deleted")
        all_id = set(srv.id for srv in nova.servers.list())
        if len(all_id.intersection(deleted_srvs)) == 0:
            break
        count += 1
        time.sleep(1)
    logger.debug("Done, deleting volumes")

    # wait till vm actually deleted

    if name_templ is not None:
        cinder = c_client(*ostack_get_creds())
        for vol in cinder.volumes.list():
            if isinstance(vol.display_name, basestring):
                if re.match(name_templ.format("\\d+"), vol.display_name):
                    if vol.status in ('available', 'error'):
                        logger.debug("Deleting volume " + vol.display_name)
                        cinder.volumes.delete(vol)

    logger.debug("Clearing done (yet some volumes may still deleting)")
