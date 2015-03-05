import os.path

import psutil


def get_disk_by_mountpoint(mnt_point):
    """ Return disk of mountpoint """
    diskparts = psutil.disk_partitions()
    for item in diskparts:
        if item.mountpoint == mnt_point:
            return os.path.realpath(item.device)

    raise OSError("Can't define disk for {0!r}".format(mnt_point))


def find_mount_point(path):
    """ Find mount point by provided path """
    path = os.path.abspath(path)
    while not os.path.ismount(path):
        path = os.path.dirname(path)
    return path


class DiskInfo(object):
    def __init__(self, name, rd_cnt=0, wr_cnt=0, rd_bytes=0,
                 wr_bytes=0, rd_time=0, wr_time=0):
        self.name = name
        self.rd_cnt = rd_cnt
        self.wr_cnt = wr_cnt
        self.rd_bytes = rd_bytes
        self.wr_bytes = wr_bytes
        self.rd_time = rd_time
        self.wr_time = wr_time

    def __str__(self):
        message = 'DISK {0.name}: read count {0.rd_cnt}' + \
                  ', write count {0.wr_cnt}' + \
                  ', read bytes {0.rd_bytes}' + \
                  ', write bytes {0.wr_bytes}' + \
                  ', read time {0.rd_time}' + \
                  ', write time {0.wr_time}'
        return message.format(self)


def get_io_stats(path):
    """ Return list of CEPHDiskInfo for all disks that used by CEPH on the
        local node
    """
    stat = psutil.disk_io_counters(perdisk=True)
    disk = get_disk_by_mountpoint(find_mount_point(path))
    disk_base = os.path.basename(disk)
    print disk_base
    try:
        return stat[disk_base]
    except IndexError:
        raise OSError("Disk {0} not found in stats".format(disk))
