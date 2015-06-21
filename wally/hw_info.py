import re
import xml.etree.ElementTree as ET

from wally import ssh_utils, utils


def get_data(rr, data):
    match_res = re.search("(?ims)" + rr, data)
    return match_res.group(0)


class HWInfo(object):
    def __init__(self):
        self.hostname = None
        self.cores = []

        # /dev/... devices
        self.disks_info = {}

        # real disks on raid controller
        self.disks_raw_info = {}

        # name => (speed, is_full_diplex, ip_addresses)
        self.net_info = {}

        self.ram_size = 0
        self.sys_name = None
        self.mb = None
        self.raw = None

        self.storage_controllers = []

    def get_HDD_count(self):
        # SATA HDD COUNT, SAS 10k HDD COUNT, SAS SSD count, PCI-E SSD count
        return []

    def get_summary(self):
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
        if self.mb is not None:
            res.append("Motherboard: " + self.mb)

        if self.ram_size == 0:
            res.append("RAM: Failed to get RAM size")
        else:
            res.append("RAM " + utils.b2ssize(self.ram_size) + "B")

        if self.cores == []:
            res.append("CPU cores: Failed to get CPU info")
        else:
            res.append("CPU cores:")
            for name, count in self.cores:
                if count > 1:
                    res.append("    {0} * {1}".format(count, name))
                else:
                    res.append("    " + name)

        if self.storage_controllers != []:
            res.append("Disk controllers:")
            for descr in self.storage_controllers:
                res.append("    " + descr)

        if self.disks_info != {}:
            res.append("Storage devices:")
            for dev, (model, size) in sorted(self.disks_info.items()):
                ssize = utils.b2ssize(size) + "B"
                res.append("    {0} {1} {2}".format(dev, ssize, model))
        else:
            res.append("Storage devices's: Failed to get info")

        if self.disks_raw_info != {}:
            res.append("Disks devices:")
            for dev, descr in sorted(self.disks_raw_info.items()):
                res.append("    {0} {1}".format(dev, descr))
        else:
            res.append("Disks devices's: Failed to get info")

        if self.net_info != {}:
            res.append("Net adapters:")
            for name, (speed, dtype, _) in self.net_info.items():
                res.append("    {0} {2} duplex={1}".format(name, dtype, speed))
        else:
            res.append("Net adapters: Failed to get net info")

        return str(self.hostname) + ":\n" + "\n".join("    " + i for i in res)


class SWInfo(object):
    def __init__(self):
        self.partitions = None
        self.kernel_version = None
        self.fio_version = None
        self.libvirt_version = None
        self.kvm_version = None
        self.qemu_version = None
        self.OS_version = None
        self.ceph_version = None


def get_sw_info(conn):
    res = SWInfo()
    res.OS_version = utils.get_os()

    with conn.open_sftp() as sftp:
        def get(fname):
            try:
                return ssh_utils.read_from_remote(sftp, fname)
            except:
                return None

        res.kernel_version = get('/proc/version')
        res.partitions = get('/etc/mtab')

    def rr(cmd):
        try:
            return ssh_utils.run_over_ssh(conn, cmd, nolog=True)
        except:
            return None

    res.libvirt_version = rr("virsh -v")
    res.qemu_version = rr("qemu-system-x86_64 --version")
    res.ceph_version = rr("ceph --version")

    return res


def get_network_info():
    pass


def get_hw_info(conn):
    res = HWInfo()
    lshw_out = ssh_utils.run_over_ssh(conn, 'sudo lshw -xml 2>/dev/null',
                                      nolog=True)

    res.raw = lshw_out
    lshw_et = ET.fromstring(lshw_out)

    try:
        res.hostname = lshw_et.find("node").attrib['id']
    except:
        pass

    try:
        res.sys_name = (lshw_et.find("node/vendor").text + " " +
                        lshw_et.find("node/product").text)
        res.sys_name = res.sys_name.replace("(To be filled by O.E.M.)", "")
        res.sys_name = res.sys_name.replace("(To be Filled by O.E.M.)", "")
    except:
        pass

    core = lshw_et.find("node/node[@id='core']")
    if core is None:
        return

    try:
        res.mb = " ".join(core.find(node).text
                          for node in ['vendor', 'product', 'version'])
    except:
        pass

    for cpu in core.findall("node[@class='processor']"):
        try:
            model = cpu.find('product').text
            threads_node = cpu.find("configuration/setting[@id='threads']")
            if threads_node is None:
                threads = 1
            else:
                threads = int(threads_node.attrib['value'])
            res.cores.append((model, threads))
        except:
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
        except:
            pass

    for net in core.findall(".//node[@class='network']"):
        try:
            link = net.find("configuration/setting[@id='link']")
            if link.attrib['value'] == 'yes':
                name = net.find("logicalname").text
                speed_node = net.find("configuration/setting[@id='speed']")

                if speed_node is None:
                    speed = None
                else:
                    speed = speed_node.attrib['value']

                dup_node = net.find("configuration/setting[@id='duplex']")
                if dup_node is None:
                    dup = None
                else:
                    dup = dup_node.attrib['value']

                res.net_info[name] = (speed, dup, [])
        except:
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
        except:
            pass

    for disk in core.findall(".//node[@class='disk']"):
        try:
            lname_node = disk.find('logicalname')
            if lname_node is not None:
                dev = lname_node.text.split('/')[-1]

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

                businfo = disk.find('businfo').text
                res.disks_raw_info[businfo] = full_descr
        except:
            pass

    return res

# import traceback
# print ET.tostring(disk)
# traceback.print_exc()

# print get_hw_info(None)

# def get_hw_info(conn):
#     res = HWInfo(None)
#     remote_run = functools.partial(ssh_utils.run_over_ssh, conn, nolog=True)

#     # some data
#     with conn.open_sftp() as sftp:
#         proc_data = ssh_utils.read_from_remote(sftp, '/proc/cpuinfo')
#         mem_data = ssh_utils.read_from_remote(sftp, '/proc/meminfo')

#     # cpu info
#     curr_core = {}
#     for line in proc_data.split("\n"):
#         if line.strip() == "":
#             if curr_core != {}:
#                 res.cores.append(curr_core)
#                 curr_core = {}
#         else:
#             param, val = line.split(":", 1)
#             curr_core[param.strip()] = val.strip()

#     if curr_core != {}:
#         res.cores.append(curr_core)

#     # RAM info
#     for line in mem_data.split("\n"):
#         if line.startswith("MemTotal"):
#             res.ram_size = int(line.split(":", 1)[1].split()[0]) * 1024
#             break

#     # HDD info
#     for dev in remote_run('ls /dev').split():
#         if dev[-1].isdigit():
#             continue

#         if dev.startswith('sd') or dev.startswith('hd'):
#             model = None
#             size = None

#             for line in remote_run('sudo hdparm -I /dev/' + dev).split("\n"):
#                 if "Model Number:" in line:
#                     model = line.split(':', 1)[1]
#                 elif "device size with M = 1024*1024" in line:
#                     size = int(line.split(':', 1)[1].split()[0])
#                     size *= 1024 ** 2

#             res.disks_info[dev] = (model, size)

#     # Network info
#     separator = '*-network'
#     net_info = remote_run('sudo lshw -class network')
#     for net_dev_info in net_info.split(separator):
#         if net_dev_info.strip().startswith("DISABLED"):
#             continue

#         if ":" not in net_dev_info:
#             continue

#         dev_params = {}
#         for line in net_dev_info.split("\n"):
#             line = line.strip()
#             if ':' in line:
#                 key, data = line.split(":", 1)
#                 dev_params[key.strip()] = data.strip()

#         if 'configuration' not in dev_params:
#             print "!!!!!", net_dev_info
#             continue

#         conf = dev_params['configuration']
#         if 'link=yes' in conf:
#             if 'speed=' in conf:
#                 speed = conf.split('speed=', 1)[1]
#                 speed = speed.strip().split()[0]
#             else:
#                 speed = None

#             if "duplex=" in conf:
#                 dtype = conf.split("duplex=", 1)[1]
#                 dtype = dtype.strip().split()[0]
#             else:
#                 dtype = None

#             res.net_info[dev_params['logical name']] = (speed, dtype)
#     return res
