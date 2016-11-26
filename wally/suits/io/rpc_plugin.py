import os
import time
import stat
import random
import subprocess


def rpc_run_fio(cfg):
    fio_cmd_templ = "cd {exec_folder}; {fio_path}fio --output-format=json " + \
                    "--output={out_file} --alloc-size=262144 {job_file}"

    result = {
        "name": [float],
        "lat_name": [[float]]
    }

    return result
    # fnames_before = node.run("ls -1 " + exec_folder, nolog=True)
    #
    # timeout = int(exec_time + max(300, exec_time))
    # soft_end_time = time.time() + exec_time
    # logger.error("Fio timeouted on node {}. Killing it".format(node))
    # end = time.time()
    # fnames_after = node.run("ls -1 " + exec_folder, nolog=True)
    #

def rpc_check_file_prefilled(path, used_size_mb):
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


def rpc_prefill_test_files(files, force=False, fio_path='fio'):
    cmd_templ = "{0} --name=xxx --filename={1} --direct=1" + \
                " --bs=4m --size={2}m --rw=write"

    ssize = 0
    ddtime = 0.0

    for fname, curr_sz in files.items():
        if not force:
            if not rpc_check_file_prefilled(fname, curr_sz):
                continue

        cmd = cmd_templ.format(fio_path, fname, curr_sz)
        ssize += curr_sz

        stime = time.time()
        subprocess.check_call(cmd)
        ddtime += time.time() - stime

    if ddtime > 1.0:
        return int(ssize / ddtime)

    return None


def load_fio_log_file(fname):
    with open(fname) as fd:
        it = [ln.split(',')[:2] for ln in fd]

    return [(float(off) / 1000,  # convert us to ms
            float(val.strip()) + 0.5)  # add 0.5 to compemsate average value
                                       # as fio trimm all values in log to integer
            for off, val in it]






