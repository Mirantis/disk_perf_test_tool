#!/usr/bin/env python

# Based on ps_mem.py:
# Licence: LGPLv2
# Author:  P@draigBrady.com
# Source:  http://www.pixelbeat.org/scripts/ps_mem.py
#   http://github.com/pixelb/scripts/commits/master/scripts/ps_mem.py

import errno
import os
import sys



class Proc:
    def __init__(self):
        uname = os.uname()
        if uname[0] == "FreeBSD":
            self.proc = '/compat/linux/proc'
        else:
            self.proc = '/proc'

    def path(self, *args):
        return os.path.join(self.proc, *(str(a) for a in args))

    def open(self, *args):
        try:
            return open(self.path(*args))
        except (IOError, OSError):
            val = sys.exc_info()[1]
            if (val.errno == errno.ENOENT or # kernel thread or process gone
                val.errno == errno.EPERM):
                raise LookupError
            raise


def kernel_ver():
    """ Return (major,minor,release)"""
    proc = Proc()
    kv = proc.open('sys/kernel/osrelease').readline().split(".")[:3]
    last = len(kv)
    if last == 2:
        kv.append('0')
    last -= 1
    while last > 0:
        for char in "-_":
            kv[last] = kv[last].split(char)[0]
        try:
            int(kv[last])
        except:
            kv[last] = 0
        last -= 1
    return (int(kv[0]), int(kv[1]), int(kv[2]))


#Note shared is always a subset of rss (trs is not always)
def getMemStats(pid):
    """ Return memory data of pid in format (Private, Shared) """

    PAGESIZE = os.sysconf("SC_PAGE_SIZE") / 1024 #KiB
    proc = Proc()

    Private_lines = []
    Shared_lines = []
    Pss_lines = []

    Rss = (int(proc.open(pid, 'statm').readline().split()[1])
           * PAGESIZE)

    if os.path.exists(proc.path(pid, 'smaps')): #stat
        for line in proc.open(pid, 'smaps').readlines(): #open
            # Note we checksum smaps as maps is usually but
            # not always different for separate processes.
            if line.startswith("Shared"):
                Shared_lines.append(line)
            elif line.startswith("Private"):
                Private_lines.append(line)
            elif line.startswith("Pss"):
                have_pss = 1
                Pss_lines.append(line)
        Shared = sum([int(line.split()[1]) for line in Shared_lines])
        Private = sum([int(line.split()[1]) for line in Private_lines])
        #Note Shared + Private = Rss above
        #The Rss in smaps includes video card mem etc.
        if have_pss:
            pss_adjust = 0.5 # add 0.5KiB as this avg error due to trunctation
            Pss = sum([float(line.split()[1])+pss_adjust for line in Pss_lines])
            Shared = Pss - Private
    elif (2, 6, 1) <= kernel_ver() <= (2, 6, 9):
        Shared = 0 #lots of overestimation, but what can we do?
        Private = Rss
    else:
        Shared = int(proc.open(pid, 'statm').readline().split()[2])
        Shared *= PAGESIZE
        Private = Rss - Shared
    return (Private, Shared)
