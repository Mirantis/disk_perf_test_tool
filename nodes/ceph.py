""" Collect data about ceph nodes"""
import json
import logging
import subprocess


from node import Node
from disk_perf_test_tool.ssh_utils import connect


logger = logging.getLogger("io-perf-tool")


def local_execute(cmd):
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)


def ssh_execute(ssh):
    def closure(cmd):
        _, chan, _ = ssh.exec_command(cmd)
        return chan.read()
    return closure


def discover_ceph_nodes(ip):
    """ Return list of ceph's nodes ips """
    ips = {}

    if ip != 'local':
        executor = ssh_execute(connect(ip))
    else:
        executor = local_execute

    osd_ips = get_osds_ips(executor, get_osds_list(executor))
    mon_ips = get_mons_or_mds_ips(executor, "mon")
    mds_ips = get_mons_or_mds_ips(executor, "mds")

    for ip in osd_ips:
        url = "ssh://%s" % ip
        ips.setdefault(url, []).append("ceph-osd")

    for ip in mon_ips:
        url = "ssh://%s" % ip
        ips.setdefault(url, []).append("ceph-mon")

    for ip in mds_ips:
        url = "ssh://%s" % ip
        ips.setdefault(url, []).append("ceph-mds")

    return [Node(url, list(roles)) for url, roles in ips.items()]


def get_osds_list(executor):
    """ Get list of osds id"""
    return filter(None, executor("ceph osd ls").split("\n"))


def get_mons_or_mds_ips(executor, who):
    """ Return mon ip list
        :param who - "mon" or "mds" """
    if who not in ("mon", "mds"):
        raise ValueError(("'%s' in get_mons_or_mds_ips instead" +
                          "of mon/mds") % who)

    line_res = executor("ceph {0} dump".format(who)).split("\n")

    ips = set()
    for line in line_res:
        fields = line.split()

        # what does fields[1], fields[2] means?
        # make this code looks like:
        # SOME_MENINGFULL_VAR1, SOME_MENINGFULL_VAR2 = line.split()[1:3]

        if len(fields) > 2 and who in fields[2]:
            ips.add(fields[1].split(":")[0])

    return ips


def get_osds_ips(executor, osd_list):
    """ Get osd's ips
        :param osd_list - list of osd names from osd ls command"""
    ips = set()
    for osd_id in osd_list:
        out = executor("ceph osd find {0}".format(osd_id))
        ip = json.loads(out)["ip"]
        ips.add(str(ip.split(":")[0]))
    return ips
