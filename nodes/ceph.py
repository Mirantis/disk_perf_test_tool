""" Collect data about ceph nodes"""
import json

import sh
from node import Node


def discover_ceph_node():
    """ Return list of ceph's nodes ips """
    
    ips = {}
    osd_list = get_osds_list()
    osd_ips = get_osds_ips(osd_list)
    mon_ips = get_mons_or_mds_ips("mon")
    mds_ips = get_mons_or_mds_ips("mds")
    for ip in osd_ips:
        url = "ssh://%s" % ip
        if url in ips:
            ips[url].add("ceph-osd")
        else:
            ips[url] = ("ceph-osd")
    for ip in mon_ips:
        url = "ssh://%s" % ip
        if url in ips:
            ips[url].add("ceph-mon")
        else:
            ips[url] = ("ceph-mon")
    for ip in mds_ips:
        url = "ssh://%s" % ip
        if url in ips:
            ips[url].add("ceph-mds")
        else:
            ips[url] = ("ceph-mds")

    res = []
    for url, roles in ips:
        res.append(Node(ip=url, roles=list(roles)))

    return res


# internal services


class CephException(Exception):
    """ Exceptions from ceph call"""
    pass

class ParameterException(Exception):
    """ Bad parameter in function"""
    pass


def get_osds_list():
    """ Get list of osds id"""
    try:
        res = sh.ceph.osd.ls()
        osd_list = [osd_id
                    for osd_id in res.split("\n") if osd_id != '']
        return osd_list
    except sh.CommandNotFound:
        raise CephException("Ceph command not found")
    except:
        raise CephException("Ceph command 'osd ls' execution error")


def get_mons_or_mds_ips(who):
    """ Return mon ip list
        :param who - "mon" or "mds" """
    try:
        ips = set()
        if who == "mon":
            res = sh.ceph.mon.dump()
        elif who == "mds":
            res = sh.ceph.mds.dump()
        else:
            raise ParameterException("'%s' in get_mons_or_mds_ips instead of mon/mds" % who)

        line_res = res.split("\n")
        for line in line_res:
            fields = line.split()
            if len(fields) > 2 and who in fields[2]:
                ips.add(fields[1].split(":")[0])

        return ips

    except sh.CommandNotFound:
        raise CephException("Ceph command not found")
    except ParameterException as e:
        raise e
    except:
        mes = "Ceph command '%s dump' execution error" % who
        raise CephException(mes)


def get_osds_ips(osd_list):
    """ Get osd's ips 
        :param osd_list - list of osd names from osd ls command"""
    try:
        ips = set()
        for osd_id in osd_list:
            res = sh.ceph.osd.find(osd_id)
            ip = json.loads(str(res))["ip"]
            ips.add(ip.split(":")[0])
        return ips

    except sh.CommandNotFound:
        raise CephException("Ceph command not found")
    except:
        raise CephException("Ceph command 'osd find' execution error")

