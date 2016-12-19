import os
import time
import stat
import random
import logging
import subprocess


mod_name = "fio"
__version__ = (0, 1)


logger = logging.getLogger("agent.fio")
SensorsMap = {}


def check_file_prefilled(path, used_size_mb):
    used_size = used_size_mb * 1024 ** 2
    blocks_to_check = 16

    try:
        fstats = os.stat(path)
        if stat.S_ISREG(fstats.st_mode) and fstats.st_size < used_size:
            return True
    except EnvironmentError:
        return True

    offsets = [random.randrange(used_size - 1024) for _ in range(blocks_to_check)]
    offsets.append(used_size - 1024)
    offsets.append(0)

    with open(path, 'rb') as fd:
        for offset in offsets:
            fd.seek(offset)
            if b"\x00" * 1024 == fd.read(1024):
                return True

    return False


def rpc_fill_file(fname, size, force=False, fio_path='fio'):
    if not force:
        if not check_file_prefilled(fname, size):
            return

    assert size % 4 == 0, "File size must be proportional to 4M"

    cmd_templ = "{} --name=xxx --filename={} --direct=1 --bs=4m --size={}m --rw=write"

    run_time = time.time()
    subprocess.check_output(cmd_templ.format(fio_path, fname, size), shell=True)
    run_time = time.time() - run_time

    return None if run_time < 1.0 else int(size / run_time)


def rpc_install(name, binary):
    try:
        subprocess.check_output("which {}".format(binary), shell=True)
    except:
        subprocess.check_output("apt-get install -y {}".format(name), shell=True)
