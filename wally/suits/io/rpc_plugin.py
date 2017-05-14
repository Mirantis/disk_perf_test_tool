import os
import time
import stat
import random
import logging
import subprocess


mod_name = "fio"
__version__ = (0, 1)


logger = logging.getLogger("agent.fio")


# TODO: fix this in case if file is block device
def check_file_prefilled(path, used_size_mb, blocks_to_check=16):
    used_size = used_size_mb * 1024 ** 2

    try:
        fstats = os.stat(path)
        if stat.S_ISREG(fstats.st_mode) and fstats.st_size < used_size:
            return False
    except EnvironmentError:
        return False

    offsets = [0, used_size - 1024] + [random.randrange(used_size - 1024) for _ in range(blocks_to_check)]
    logger.debug(str(offsets))
    with open(path, 'rb') as fd:
        for offset in offsets:
            fd.seek(offset)
            if b"\x00" * 1024 == fd.read(1024):
                return False

    return True


def rpc_fill_file(fname, size, force=False, fio_path='fio'):
    if not force:
        if check_file_prefilled(fname, size):
            return False, None

    assert size % 4 == 0, "File size must be proportional to 4M"

    cmd_templ = "{0} --name=xxx --filename={1} --direct=1 --bs=4m --size={2}m --rw=write"

    run_time = time.time()
    try:
        subprocess.check_output(cmd_templ.format(fio_path, fname, size), shell=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("{0!s}.\nOutput: {1}".format(exc, exc.output))
    run_time = time.time() - run_time

    prefill_bw = None if run_time < 1.0 else int(size / run_time)

    return True, prefill_bw


def rpc_install(name, binary):
    try:
        subprocess.check_output("which {0}".format(binary), shell=True)
    except:
        subprocess.check_output("apt-get install -y {0}".format(name), shell=True)
