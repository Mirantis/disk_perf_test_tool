import re
import os
import time
import os.path
import logging
import subprocess

from concurrent.futures import ThreadPoolExecutor

from novaclient.exceptions import NotFound
from novaclient.client import Client as n_client
from cinderclient.v1.client import Client as c_client

import wally
from wally.discover import Node


logger = logging.getLogger("wally.vms")


STORED_OPENSTACK_CREDS = None
NOVA_CONNECTION = None
CINDER_CONNECTION = None


def is_connected():
    return NOVA_CONNECTION is not None


def ostack_get_creds():
    if STORED_OPENSTACK_CREDS is None:
        env = os.environ.get
        name = env('OS_USERNAME')
        passwd = env('OS_PASSWORD')
        tenant = env('OS_TENANT_NAME')
        auth_url = env('OS_AUTH_URL')
        return name, passwd, tenant, auth_url
    else:
        return STORED_OPENSTACK_CREDS


def nova_connect(name=None, passwd=None, tenant=None, auth_url=None):
    global NOVA_CONNECTION
    global STORED_OPENSTACK_CREDS

    if NOVA_CONNECTION is None:
        if name is None:
            name, passwd, tenant, auth_url = ostack_get_creds()
        else:
            STORED_OPENSTACK_CREDS = (name, passwd, tenant, auth_url)

        NOVA_CONNECTION = n_client('1.1', name, passwd, tenant, auth_url)
    return NOVA_CONNECTION


def cinder_connect(name=None, passwd=None, tenant=None, auth_url=None):
    global CINDER_CONNECTION
    global STORED_OPENSTACK_CREDS

    if CINDER_CONNECTION is None:
        if name is None:
            name, passwd, tenant, auth_url = ostack_get_creds()
        else:
            STORED_OPENSTACK_CREDS = (name, passwd, tenant, auth_url)
        CINDER_CONNECTION = c_client(name, passwd, tenant, auth_url)
    return CINDER_CONNECTION


def prepare_os_subpr(params, name=None, passwd=None, tenant=None,
                     auth_url=None):
    if name is None:
        name, passwd, tenant, auth_url = ostack_get_creds()

    MAX_VM_PER_NODE = 8
    serv_groups = " ".join(map(params['aa_group_name'].format,
                               range(MAX_VM_PER_NODE)))

    env = os.environ.copy()
    env.update(dict(
        OS_USERNAME=name,
        OS_PASSWORD=passwd,
        OS_TENANT_NAME=tenant,
        OS_AUTH_URL=auth_url,

        FLAVOR_NAME=params['flavor']['name'],
        FLAVOR_RAM=str(params['flavor']['ram_size']),
        FLAVOR_HDD=str(params['flavor']['hdd_size']),
        FLAVOR_CPU_COUNT=str(params['flavor']['cpu_count']),

        SERV_GROUPS=serv_groups,
        KEYPAIR_NAME=params['keypair_name'],

        SECGROUP=params['security_group'],

        IMAGE_NAME=params['image']['name'],
        KEY_FILE_NAME=params['keypair_file_private'],
        IMAGE_URL=params['image']['url'],
    ))

    spath = os.path.dirname(os.path.dirname(wally.__file__))
    spath = os.path.join(spath, 'scripts/prepare.sh')

    cmd = "bash {spath} >/dev/null".format(spath=spath)
    subprocess.check_call(cmd, shell=True, env=env)

    conn = nova_connect(name, passwd, tenant, auth_url)
    while True:
        status = conn.images.find(name='wally_ubuntu').status
        if status == 'ACTIVE':
            break
        msg = "Image {0} is still in {1} state. Waiting 10 more seconds"
        logger.info(msg.format('wally_ubuntu', status))
        time.sleep(10)


def find_vms(nova, name_prefix):
    for srv in nova.servers.list():
        if srv.name.startswith(name_prefix):
            for ips in srv.addresses.values():
                for ip in ips:
                    if ip.get("OS-EXT-IPS:type", None) == 'floating':
                        yield ip['addr'], srv.id
                        break


def pause(ids):
    def pause_vm(conn, vm_id):
        vm = conn.servers.get(vm_id)
        if vm.status == 'ACTIVE':
            vm.pause()

    conn = nova_connect()
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(pause_vm, conn, vm_id)
                   for vm_id in ids]
        for future in futures:
            future.result()


def unpause(ids, max_resume_time=10):
    def unpause(conn, vm_id):
        vm = conn.servers.get(vm_id)
        if vm.status == 'PAUSED':
            vm.unpause()

        for i in range(max_resume_time * 10):
            vm = conn.servers.get(vm_id)
            if vm.status != 'PAUSED':
                return
            time.sleep(0.1)
        raise RuntimeError("Can't unpause vm {0}".format(vm_id))

    conn = nova_connect()
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(unpause, conn, vm_id)
                   for vm_id in ids]

        for future in futures:
            future.result()


def prepare_os(nova, params):
    allow_ssh(nova, params['security_group'])

    MAX_VM_PER_NODE = 8
    serv_groups = " ".join(map(params['aa_group_name'].format,
                               range(MAX_VM_PER_NODE)))

    shed_ids = []
    for shed_group in serv_groups:
        shed_ids.append(get_or_create_aa_group(nova, shed_group))

    create_keypair(nova,
                   params['keypair_name'],
                   params['keypair_name'] + ".pub",
                   params['keypair_name'] + ".pem")

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


def create_flavor(nova, name, ram_size, hdd_size, cpu_count):
    pass


def create_keypair(nova, name, pub_key_path, priv_key_path):
    try:
        nova.keypairs.find(name=name)
        # if file not found- delete and recreate
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
    cinder = cinder_connect()
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


def wait_for_server_active(nova, server, timeout=300):
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


def launch_vms(params, already_has_count=0):
    logger.debug("Calculating new vm count")
    count = params['count']
    nova = nova_connect()
    lst = nova.services.list(binary='nova-compute')
    srv_count = len([srv for srv in lst if srv.status == 'enabled'])

    if isinstance(count, basestring):
        if count.startswith("x"):
            count = srv_count * int(count[1:])
        else:
            assert count.startswith('=')
            count = int(count[1:]) - already_has_count

    if count <= 0:
        logger.debug("Not need new vms")
        return

    logger.debug("Starting new nodes on openstack")

    assert isinstance(count, (int, long))

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
    creds = params['image']['creds']
    creds.format(ip="1.1.1.1", private_key_path="/some_path/xx")

    for ip, os_node in create_vms_mt(NOVA_CONNECTION, count, **vm_params):

        conn_uri = creds.format(ip=ip, private_key_path=private_key_path)
        yield Node(conn_uri, []), os_node.id


def get_free_server_grpoups(nova, template=None):
    for g in nova.server_groups.list():
        if g.members == []:
            if re.match(template, g.name):
                yield str(g.name)


def create_vms_mt(nova, amount, group_name, keypair_name, img_name,
                  flavor_name, vol_sz=None, network_zone_name=None,
                  flt_ip_pool=None, name_templ='wally-{id}',
                  scheduler_hints=None, security_group=None,
                  sec_group_size=None):

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

        names = []
        for i in range(amount):
            names.append(name_templ.format(group=group_name, id=i))

        futures = []
        logger.debug("Requesting new vm's")

        orig_scheduler_hints = scheduler_hints.copy()

        MAX_SHED_GROUPS = 32
        for start_idx in range(MAX_SHED_GROUPS):
            pass

        group_name_template = scheduler_hints['group'].format("\\d+")
        groups = list(get_free_server_grpoups(nova, group_name_template + "$"))
        groups.sort()

        for idx, (name, flt_ip) in enumerate(zip(names, ips), 2):

            scheduler_hints = None
            if orig_scheduler_hints is not None and sec_group_size is not None:
                if "group" in orig_scheduler_hints:
                    scheduler_hints = orig_scheduler_hints.copy()
                    scheduler_hints['group'] = groups[idx // sec_group_size]

            if scheduler_hints is None:
                scheduler_hints = orig_scheduler_hints.copy()

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

            try:
                for j in range(120):
                    srv = nova.servers.get(srv.id)
                    time.sleep(1)
                else:
                    msg = "Server {0} delete timeout".format(srv.id)
                    raise RuntimeError(msg)
            except NotFound:
                pass
        else:
            break
    else:
        raise RuntimeError("Failed to start server".format(srv.id))

    if vol_sz is not None:
        vol = create_volume(vol_sz, name)
        nova.volumes.create_server_volume(srv.id, vol.id, None)

    if flt_ip is Allocate:
        flt_ip = nova.floating_ips.create(pool)

    if flt_ip is not None:
        srv.add_floating_ip(flt_ip)

    return flt_ip.ip, nova.servers.get(srv.id)


def clear_nodes(nodes_ids):
    clear_all(NOVA_CONNECTION, nodes_ids, None)


def clear_all(nova, ids=None, name_templ=None):

    def need_delete(srv):
        if name_templ is not None:
            return re.match(name_templ.format("\\d+"), srv.name) is not None
        else:
            return srv.id in ids

    volumes_to_delete = []
    cinder = cinder_connect()
    for vol in cinder.volumes.list():
        for attachment in vol.attachments:
            if attachment['server_id'] in ids:
                volumes_to_delete.append(vol)
                break

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

    # logger.warning("Volume deletion commented out")
    for vol in volumes_to_delete:
        logger.debug("Deleting volume " + vol.display_name)
        cinder.volumes.delete(vol)

    logger.debug("Clearing done (yet some volumes may still deleting)")
