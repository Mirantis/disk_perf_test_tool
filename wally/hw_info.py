import re
import logging
from typing import Dict, Iterable
import xml.etree.ElementTree as ET
from typing import List, Tuple, cast, Optional

from . import utils
from .node_utils import get_os, OSRelease
from .node_interfaces import IRPCNode


logger = logging.getLogger("wally")


def get_data(rr: str, data: str) -> str:
    match_res = re.search("(?ims)" + rr, data)
    return match_res.group(0)


class HWInfo:
    def __init__(self) -> None:
        self.hostname = None  # type: str
        self.cores = []  # type: List[Tuple[str, int]]

        # /dev/... devices
        self.disks_info = {}  # type: Dict[str, Tuple[str, int]]

        # real disks on raid controller
        self.disks_raw_info = {}  # type: Dict[str, str]

        # name => (speed, is_full_diplex, ip_addresses)
        self.net_info = {}  # type: Dict[str, Tuple[Optional[int], Optional[bool], List[str]]]

        self.ram_size = 0  # type: int
        self.sys_name = None  # type: str
        self.mb = None  # type: str
        self.raw = None  # type: str

        self.storage_controllers = []  # type: List[str]

    def get_hdd_count(self) -> Iterable[int]:
        # SATA HDD COUNT, SAS 10k HDD COUNT, SAS SSD count, PCI-E SSD count
        return []

    def get_summary(self) -> Dict[str, int]:
        cores = sum(count for _, count in self.cores)
        disks = sum(size for _, size in self.disks_info.values())

        return {'cores': cores,
                'ram': self.ram_size,
                'storage': disks,
                'disk_count': len(self.disks_info)}

    def __str__(self):
        res = []

        summ = self.get_summary()
        summary = "Simmary: {cores} cores, {ram}B RAM, {disk}B storage"
        res.append(summary.format(cores=summ['cores'],
                                  ram=utils.b2ssize(summ['ram']),
                                  disk=utils.b2ssize(summ['storage'])))
        res.append(str(self.sys_name))
        if self.mb:
            res.append("Motherboard: " + self.mb)

        if not self.ram_size:
            res.append("RAM: Failed to get RAM size")
        else:
            res.append("RAM " + utils.b2ssize(self.ram_size) + "B")

        if not self.cores:
            res.append("CPU cores: Failed to get CPU info")
        else:
            res.append("CPU cores:")
            for name, count in self.cores:
                if count > 1:
                    res.append("    {0} * {1}".format(count, name))
                else:
                    res.append("    " + name)

        if self.storage_controllers:
            res.append("Disk controllers:")
            for descr in self.storage_controllers:
                res.append("    " + descr)

        if self.disks_info:
            res.append("Storage devices:")
            for dev, (model, size) in sorted(self.disks_info.items()):
                ssize = utils.b2ssize(size) + "B"
                res.append("    {0} {1} {2}".format(dev, ssize, model))
        else:
            res.append("Storage devices's: Failed to get info")

        if self.disks_raw_info:
            res.append("Disks devices:")
            for dev, descr in sorted(self.disks_raw_info.items()):
                res.append("    {0} {1}".format(dev, descr))
        else:
            res.append("Disks devices's: Failed to get info")

        if self.net_info:
            res.append("Net adapters:")
            for name, (speed, dtype, _) in self.net_info.items():
                res.append("    {0} {2} duplex={1}".format(name, dtype, speed))
        else:
            res.append("Net adapters: Failed to get net info")

        return str(self.hostname) + ":\n" + "\n".join("    " + i for i in res)


class CephInfo:
    def __init__(self) -> None:
        pass


class SWInfo:
    def __init__(self) -> None:
        self.mtab = None  # type: str
        self.kernel_version = None  # type: str
        self.libvirt_version = None  # type: Optional[str]
        self.qemu_version = None  # type: Optional[str]
        self.OS_version = None  # type: OSRelease
        self.ceph_info = None  # type: Optional[CephInfo]


def get_ceph_services_info(node: IRPCNode) -> CephInfo:
    # TODO: use ceph-monitoring module
    return CephInfo()


def get_sw_info(node: IRPCNode) -> SWInfo:
    res = SWInfo()

    res.OS_version = get_os(node)
    res.kernel_version = node.get_file_content('/proc/version').decode('utf8').strip()
    res.mtab = node.get_file_content('/etc/mtab').decode('utf8').strip()

    try:
        res.libvirt_version = node.run("virsh -v", nolog=True).strip()
    except OSError:
        res.libvirt_version = None

    # dpkg -l ??

    try:
        res.qemu_version = node.run("qemu-system-x86_64 --version", nolog=True).strip()
    except OSError:
        res.qemu_version = None

    for role in ('ceph-osd', 'ceph-mon', 'ceph-mds'):
        if role in node.info.roles:
            res.ceph_info = get_ceph_services_info(node)
            break

    return res


def get_hw_info(node: IRPCNode) -> Optional[HWInfo]:

    try:
        lshw_out = node.run('sudo lshw -xml 2>/dev/null')
    except Exception as exc:
        logger.warning("lshw failed on node %s: %s", node.node_id, exc)
        return None

    res = HWInfo()
    res.raw = lshw_out
    lshw_et = ET.fromstring(lshw_out)

    try:
        res.hostname = cast(str, lshw_et.find("node").attrib['id'])
    except Exception:
        pass

    try:

        res.sys_name = cast(str, lshw_et.find("node/vendor").text) + " " + \
            cast(str, lshw_et.find("node/product").text)
        res.sys_name = res.sys_name.replace("(To be filled by O.E.M.)", "")
        res.sys_name = res.sys_name.replace("(To be Filled by O.E.M.)", "")
    except Exception:
        pass

    core = lshw_et.find("node/node[@id='core']")
    if core is None:
        return res

    try:
        res.mb = " ".join(cast(str, core.find(node).text)
                          for node in ['vendor', 'product', 'version'])
    except Exception:
        pass

    for cpu in core.findall("node[@class='processor']"):
        try:
            model = cast(str, cpu.find('product').text)
            threads_node = cpu.find("configuration/setting[@id='threads']")
            if threads_node is None:
                threads = 1
            else:
                threads = int(threads_node.attrib['value'])
            res.cores.append((model, threads))
        except Exception:
            pass

    res.ram_size = 0
    for mem_node in core.findall(".//node[@class='memory']"):
        descr = mem_node.find('description')
        try:
            if descr is not None and descr.text == 'System Memory':
                mem_sz = mem_node.find('size')
                if mem_sz is None:
                    for slot_node in mem_node.find("node[@class='memory']"):
                        slot_sz = slot_node.find('size')
                        if slot_sz is not None:
                            assert slot_sz.attrib['units'] == 'bytes'
                            res.ram_size += int(slot_sz.text)
                else:
                    assert mem_sz.attrib['units'] == 'bytes'
                    res.ram_size += int(mem_sz.text)
        except Exception:
            pass

    for net in core.findall(".//node[@class='network']"):
        try:
            link = net.find("configuration/setting[@id='link']")
            if link.attrib['value'] == 'yes':
                name = cast(str, net.find("logicalname").text)
                speed_node = net.find("configuration/setting[@id='speed']")

                if speed_node is None:
                    speed = None
                else:
                    speed = int(speed_node.attrib['value'])

                dup_node = net.find("configuration/setting[@id='duplex']")
                if dup_node is None:
                    dup = None
                else:
                    dup = cast(str, dup_node.attrib['value']).lower() == 'yes'

                ips = []  # type: List[str]
                res.net_info[name] = (speed, dup, ips)
        except Exception:
            pass

    for controller in core.findall(".//node[@class='storage']"):
        try:
            description = getattr(controller.find("description"), 'text', "")
            product = getattr(controller.find("product"), 'text', "")
            vendor = getattr(controller.find("vendor"), 'text', "")
            dev = getattr(controller.find("logicalname"), 'text', "")
            if dev != "":
                res.storage_controllers.append(
                    "{0}: {1} {2} {3}".format(dev, description,
                                              vendor, product))
            else:
                res.storage_controllers.append(
                    "{0} {1} {2}".format(description,
                                         vendor, product))
        except Exception:
            pass

    for disk in core.findall(".//node[@class='disk']"):
        try:
            lname_node = disk.find('logicalname')
            if lname_node is not None:
                dev = cast(str, lname_node.text).split('/')[-1]

                if dev == "" or dev[-1].isdigit():
                    continue

                sz_node = disk.find('size')
                assert sz_node.attrib['units'] == 'bytes'
                sz = int(sz_node.text)
                res.disks_info[dev] = ('', sz)
            else:
                description = disk.find('description').text
                product = disk.find('product').text
                vendor = disk.find('vendor').text
                version = disk.find('version').text
                serial = disk.find('serial').text

                full_descr = "{0} {1} {2} {3} {4}".format(
                    description, product, vendor, version, serial)

                businfo = cast(str, disk.find('businfo').text)
                res.disks_raw_info[businfo] = full_descr
        except Exception:
            pass

    return res
