import re
import os
import time
import logging

from concurrent.futures import ThreadPoolExecutor

from novaclient.client import Client as n_client
from cinderclient.v1.client import Client as c_client


logger = logging.getLogger("io-perf-tool")


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


def create_vms_mt(nova, amount, keypair_name, img_name,
                  flavor_name, vol_sz=None, network_zone_name=None,
                  flt_ip_pool=None, name_templ='ceph-test-{0}',
                  scheduler_hints=None):

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
        else:
            ips_future = None

        if ips_future is not None:
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
                      flt_ip_pool)

            futures.append(executor.submit(create_vm, *params))
        res = [future.result() for future in futures]
        logger.debug("Done spawning")
        return res


def create_vm(nova, name, keypair_name, img,
              fl, nics, vol_sz=None,
              flt_ip=False,
              scheduler_hints=None,
              pool=None):
    for i in range(3):
        srv = nova.servers.create(name,
                                  flavor=fl, image=img, nics=nics,
                                  key_name=keypair_name,
                                  scheduler_hints=scheduler_hints)

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
        return (flt_ip.ip, srv)
    else:
        return (None, srv)


def clear_all(nova, name_templ="ceph-test-{0}"):
    deleted_srvs = set()
    for srv in nova.servers.list():
        if re.match(name_templ.format("\\d+"), srv.name):
            logger.debug("Deleting server {0}".format(srv.name))
            nova.servers.delete(srv)
            deleted_srvs.add(srv.id)

    while deleted_srvs != set():
        logger.debug("Waiting till all servers are actually deleted")
        all_id = set(srv.id for srv in nova.servers.list())
        if all_id.intersection(deleted_srvs) == set():
            logger.debug("Done, deleting volumes")
            break
        time.sleep(1)

    # wait till vm actually deleted

    cinder = c_client(*ostack_get_creds())
    for vol in cinder.volumes.list():
        if isinstance(vol.display_name, basestring):
            if re.match(name_templ.format("\\d+"), vol.display_name):
                if vol.status in ('available', 'error'):
                    print "Deleting volume", vol.display_name
                    cinder.volumes.delete(vol)

    logger.debug("Clearing done (yet some volumes may still deleting)")


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


# def main():
#     image_name = 'TestVM'
#     flavor_name = 'ceph'
#     vol_sz = 50
#     network_zone_name = 'net04'
#     amount = 10
#     keypair_name = 'ceph-test'

#     nova = nova_connect()
#     clear_all(nova)

#     try:
#         ips = []
#         params = dict(vol_sz=vol_sz)
#         params['image_name'] = image_name
#         params['flavor_name'] = flavor_name
#         params['network_zone_name'] = network_zone_name
#         params['amount'] = amount
#         params['keypair_name'] = keypair_name

#         for ip, host in create_vms(nova, **params):
#             ips.append(ip)

#         print "All setup done! Ips =", " ".join(ips)
#         print "Starting tests"
#     finally:
#         clear_all(nova)

# if __name__ == "__main__":
#     exit(main())
