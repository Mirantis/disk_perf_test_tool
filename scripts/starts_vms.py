import re
import os
import time

import paramiko
from novaclient.client import Client as n_client
from cinderclient.v1.client import Client as c_client


def ostack_get_creds():
    env = os.environ.get
    name = env('OS_USERNAME')
    passwd = env('OS_PASSWORD')
    tenant = env('OS_TENANT_NAME')
    auth_url = env('OS_AUTH_URL')
    return name, passwd, tenant, auth_url


def nova_connect():
    return n_client('1.1', *ostack_get_creds())


def create_keypair(nova, name, key_path):
    with open(key_path) as key:
        return nova.keypairs.create(name, key.read())


def create_volume(size, name=None, volid=[0]):
    cinder = c_client(*ostack_get_creds())
    name = 'ceph-test-{0}'.format(volid[0])
    volid[0] = volid[0] + 1
    vol = cinder.volumes.create(size=size, display_name=name)
    err_count = 0
    while vol.status != 'available':
        if vol.status == 'error':
            if err_count == 3:
                print "Fail to create volume"
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
        time.sleep(5)
        sstate = getattr(server, 'OS-EXT-STS:vm_state').lower()

        if sstate == 'active':
            return True

        print "Curr state is", sstate, "waiting for active"

        if sstate == 'error':
            return False

        if time.time() - t > timeout:
            return False

        server = nova.servers.get(server)


def get_or_create_floating_ip(nova, pool, used_ip):
    ip_list = nova.floating_ips.list()

    if pool is not None:
        ip_list = [ip for ip in ip_list if ip.pool == pool]

    ip_list = [ip for ip in ip_list if ip.instance_id is None]
    ip_list = [ip for ip in ip_list if ip.ip not in used_ip]

    if len(ip_list) > 0:
        return ip_list[0]
    else:
        return nova.floating_ips.create(pool)


def create_vms(nova, amount, keypair_name, img_name,
               flavor_name, vol_sz, network_zone_name=None):

    network = nova.networks.find(label=network_zone_name)
    nics = [{'net-id': network.id}]
    fl = nova.flavors.find(name=flavor_name)
    img = nova.images.find(name=img_name)
    srvs = []
    counter = 0

    for i in range(3):
        amount_left = amount - len(srvs)

        new_srvs = []
        for i in range(amount_left):
            print "creating server"
            srv = nova.servers.create("ceph-test-{0}".format(counter),
                                      flavor=fl, image=img, nics=nics,
                                      key_name=keypair_name)
            counter += 1
            new_srvs.append(srv)
            print srv

        deleted_servers = []
        for srv in new_srvs:
            if not wait_for_server_active(nova, srv):
                print "Server", srv.name, "fails to start. Kill it and",
                print " try again"

                nova.servers.delete(srv)
                deleted_servers.append(srv)
            else:
                srvs.append(srv)

        if len(deleted_servers) != 0:
            time.sleep(5)

    if len(srvs) != amount:
        print "ERROR: can't start required amount of servers. Exit"
        raise RuntimeError("Fail to create {0} servers".format(amount))

    result = {}
    for srv in srvs:
        print "wait till server be ready"
        wait_for_server_active(nova, srv)
        print "creating volume"
        vol = create_volume(vol_sz)
        print "attach volume to server"
        nova.volumes.create_server_volume(srv.id, vol.id, None)
        print "create floating ip"
        flt_ip = get_or_create_floating_ip(nova, 'net04_ext', result.keys())
        print "attaching ip to server"
        srv.add_floating_ip(flt_ip)
        result[flt_ip.ip] = srv

    return result


def clear_all(nova):
    deleted_srvs = set()
    for srv in nova.servers.list():
        if re.match(r"ceph-test-\d+", srv.name):
            print "Deleting server", srv.name
            nova.servers.delete(srv)
            deleted_srvs.add(srv.id)

    while deleted_srvs != set():
        print "Waiting till all servers are actually deleted"
        all_id = set(srv.id for srv in nova.servers.list())
        if all_id.intersection(deleted_srvs) == set():
            print "Done, deleting volumes"
            break
        time.sleep(1)

    # wait till vm actually deleted

    cinder = c_client(*ostack_get_creds())
    for vol in cinder.volumes.list():
        if isinstance(vol.display_name, basestring):
            if re.match(r'ceph-test-\d+', vol.display_name):
                if vol.status in ('available', 'error'):
                    print "Deleting volume", vol.display_name
                    cinder.volumes.delete(vol)

    print "Clearing done (yet some volumes may still deleting)"


def wait_ssh_ready(host, user, key_file, retry_count=10, timeout=5):
    ssh = paramiko.SSHClient()
    ssh.load_host_keys('/dev/null')
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.known_hosts = None

    for i in range(retry_count):
        try:
            ssh.connect(host, username=user, key_filename=key_file,
                        look_for_keys=False)
            break
        except:
            if i == retry_count - 1:
                raise
            time.sleep(timeout)


# def prepare_host(key_file, ip, fio_path, dst_fio_path, user='cirros'):
#     print "Wait till ssh ready...."
#     wait_ssh_ready(ip, user, key_file)

#     print "Preparing host >"
#     print "    Coping fio"
#     copy_fio(key_file, ip, fio_path, user, dst_fio_path)

#     key_opts = '-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no'
#     args = (key_file, user, ip, key_opts)
#     cmd_format = "ssh {3} -i {0} {1}@{2} '{{0}}'".format(*args).format

#     def exec_on_host(cmd):
#         print "    " + cmd
#         subprocess.check_call(cmd_format(cmd), shell=True)

#     exec_on_host("sudo /usr/sbin/mkfs.ext4 /dev/vdb")
#     exec_on_host("sudo /bin/mkdir /media/ceph")
#     exec_on_host("sudo /bin/mount /dev/vdb /media/ceph")
#     exec_on_host("sudo /bin/chmod a+rwx /media/ceph")


def main():
    image_name = 'TestVM'
    flavor_name = 'ceph'
    vol_sz = 50
    network_zone_name = 'net04'
    amount = 10
    keypair_name = 'ceph-test'

    nova = nova_connect()
    clear_all(nova)

    try:
        ips = []
        params = dict(vol_sz=vol_sz)
        params['image_name'] = image_name
        params['flavor_name'] = flavor_name
        params['network_zone_name'] = network_zone_name
        params['amount'] = amount
        params['keypair_name'] = keypair_name

        for ip, host in create_vms(nova, **params).items():
            ips.append(ip)

        print "All setup done! Ips =", " ".join(ips)
        print "Starting tests"
    finally:
        clear_all(nova)

if __name__ == "__main__":
    exit(main())
