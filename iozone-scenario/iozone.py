import re
import os
import sys
import stat
import time
import json
import os.path
import argparse
import warnings
import subprocess


class BenchmarkOption(object):
    def __init__(self, concurence, iodepth, action, blocksize, size):
        self.iodepth = iodepth
        self.action = action
        self.blocksize = blocksize
        self.concurence = concurence
        self.size = size
        self.direct_io = False
        self.use_hight_io_priority = True
        self.sync = False


class IOZoneParser(object):
    "class to parse iozone results"

    start_tests = re.compile(r"^\s+KB\s+reclen\s+")
    resuts = re.compile(r"[\s0-9]+")
    mt_iozone_re = re.compile(r"\s+Children see throughput " +
                              r"for\s+\d+\s+(?P<cmd>.*?)\s+=\s+" +
                              r"(?P<perf>[\d.]+)\s+KB/sec")

    cmap = {'initial writers': 'write',
            'rewriters': 'rewrite',
            'initial readers': 'read',
            're-readers': 'reread',
            'random readers': 'random read',
            'random writers': 'random write'}

    string1 = "                           " + \
              "                   random  random    " + \
              "bkwd   record   stride                                   "

    string2 = "KB  reclen   write rewrite    " + \
              "read    reread    read   write    " + \
              "read  rewrite     read   fwrite frewrite   fread  freread"

    @classmethod
    def apply_parts(cls, parts, string, sep=' \t\n'):
        add_offset = 0
        for part in parts:
            _, start, stop = part
            start += add_offset
            add_offset = 0

            # condition splited to make pylint happy
            while stop + add_offset < len(string):

                # condition splited to make pylint happy
                if not (string[stop + add_offset] not in sep):
                    break

                add_offset += 1

            yield part, string[start:stop + add_offset]

    @classmethod
    def make_positions(cls):
        items = [i for i in cls.string2.split() if i]

        pos = 0
        cls.positions = []

        for item in items:
            npos = cls.string2.index(item, 0 if pos == 0 else pos + 1)
            cls.positions.append([item, pos, npos + len(item)])
            pos = npos + len(item)

        for itm, val in cls.apply_parts(cls.positions, cls.string1):
            if val.strip():
                itm[0] = val.strip() + " " + itm[0]

    @classmethod
    def parse_iozone_res(cls, res, mthreads=False):
        parsed_res = None

        sres = res.split('\n')

        if not mthreads:
            for pos, line in enumerate(sres[1:]):
                if line.strip() == cls.string2 and \
                            sres[pos].strip() == cls.string1.strip():
                    add_pos = line.index(cls.string2)
                    parsed_res = {}

                    npos = [(name, start + add_pos, stop + add_pos)
                            for name, start, stop in cls.positions]

                    for itm, res in cls.apply_parts(npos, sres[pos + 2]):
                        if res.strip() != '':
                            parsed_res[itm[0]] = int(res.strip())

                    del parsed_res['KB']
                    del parsed_res['reclen']
        else:
            parsed_res = {}
            for line in sres:
                rr = cls.mt_iozone_re.match(line)
                if rr is not None:
                    cmd = rr.group('cmd')
                    key = cls.cmap.get(cmd, cmd)
                    perf = int(float(rr.group('perf')))
                    parsed_res[key] = perf
        return parsed_res


def ssize_to_kb(ssize):
    try:
        smap = dict(k=1, K=1, M=1024, m=1024, G=1024**2, g=1024**2)
        for ext, coef in smap.items():
            if ssize.endswith(ext):
                return int(ssize[:-1]) * coef

        if int(ssize) % 1024 != 0:
            raise ValueError()

        return int(ssize) / 1024

    except (ValueError, TypeError, AttributeError):
        tmpl = "Unknow size format {0!r} (or size not multiples 1024)"
        raise ValueError(tmpl.format(ssize))


IOZoneParser.make_positions()


def do_run_iozone(params, filename, timeout, iozone_path='iozone',
                  microsecond_mode=False):

    cmd = [iozone_path]

    if params.sync:
        cmd.append('-o')

    if params.direct_io:
        cmd.append('-I')

    if microsecond_mode:
        cmd.append('-N')

    all_files = []
    threads = int(params.concurence)
    if 1 != threads:
        cmd.extend(('-t', str(threads), '-F'))
        filename = filename + "_{}"
        cmd.extend(filename % i for i in range(threads))
        all_files.extend(filename % i for i in range(threads))
    else:
        cmd.extend(('-f', filename))
        all_files.append(filename)

    bsz = 1024 if params.size > 1024 else params.size
    if params.size % bsz != 0:
        fsz = (params.size // bsz + 1) * bsz
    else:
        fsz = params.size

    for fname in all_files:
        ccmd = [iozone_path, "-f", fname, "-i", "0",
                "-s",  str(fsz), "-r", str(bsz), "-w"]
        subprocess.check_output(ccmd)

    cmd.append('-i')

    if params.action == 'write':
        cmd.append("0")
    elif params.action == 'randwrite':
        cmd.append("2")
    else:
        raise ValueError("Unknown action {0!r}".format(params.action))

    cmd.extend(('-s', str(params.size)))
    cmd.extend(('-r', str(params.blocksize)))

    raw_res = subprocess.check_output(cmd)

    try:
        parsed_res = IOZoneParser.parse_iozone_res(raw_res, threads > 1)

        res = {}

        if params.action == 'write':
            res['bw_mean'] = parsed_res['write']
        elif params.action == 'randwrite':
            res['bw_mean'] = parsed_res['random write']
        elif params.action == 'read':
            res['bw_mean'] = parsed_res['read']
        elif params.action == 'randread':
            res['bw_mean'] = parsed_res['random read']
    except:
        print raw_res
        raise

    # res['bw_dev'] = 0
    # res['bw_max'] = res["bw_mean"]
    # res['bw_min'] = res["bw_mean"]

    return res


def run_iozone(benchmark, iozone_path, tmpname, timeout=None):
    if timeout is not None:
        benchmark.size = benchmark.blocksize * 50
        res_time = do_run_iozone(benchmark, tmpname, timeout,
                                 iozone_path=iozone_path,
                                 microsecond_mode=True)

        size = (benchmark.blocksize * timeout * 1000000)
        size /= res_time["bw_mean"]
        size = (size // benchmark.blocksize + 1) * benchmark.blocksize
        benchmark.size = size

    return do_run_iozone(benchmark, tmpname, timeout,
                         iozone_path=iozone_path)


def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def install_iozone_package():
    if which('iozone'):
        return

    is_redhat = os.path.exists('/etc/centos-release')
    is_redhat = is_redhat or os.path.exists('/etc/fedora-release')
    is_redhat = is_redhat or os.path.exists('/etc/redhat-release')

    if is_redhat:
        subprocess.check_output(["yum", "install", 'iozone3'])
        return

    try:
        os_release_cont = open('/etc/os-release').read()

        is_ubuntu = "Ubuntu" in os_release_cont

        if is_ubuntu or "Debian GNU/Linux" in os_release_cont:
            subprocess.check_output(["apt-get", "install", "iozone3"])
            return
    except (IOError, OSError) as exc:
        print exc
        pass

    raise RuntimeError("Unknown host OS.")


def install_iozone_static(iozone_url, dst):
    if not os.path.isfile(dst):
        import urllib
        urllib.urlretrieve(iozone_url, dst)

    st = os.stat(dst)
    os.chmod(dst, st.st_mode | stat.S_IEXEC)


def type_size(string):
    try:
        return re.match("\d+[KGBM]?", string, re.I).group(0)
    except:
        msg = "{0!r} don't looks like size-description string".format(string)
        raise ValueError(msg)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run set of `iozone` invocations and return result")
    parser.add_argument(
        "--iodepth", metavar="IODEPTH", type=int,
        help="I/O depths to test in kb", required=True)
    parser.add_argument(
        '-a', "--action", metavar="ACTION", type=str,
        help="actions to run", required=True,
        choices=["read", "write", "randread", "randwrite"])
    parser.add_argument(
        "--blocksize", metavar="BLOCKSIZE", type=type_size,
        help="single operation block size", required=True)
    parser.add_argument(
        "--timeout", metavar="TIMEOUT", type=int,
        help="runtime of a single run", default=None)
    parser.add_argument(
        "--iosize", metavar="SIZE", type=type_size,
        help="file size", default=None)
    parser.add_argument(
        "-s", "--sync", default=False, action="store_true",
        help="exec sync after each write")
    parser.add_argument(
        "-d", "--direct-io", default=False, action="store_true",
        help="use O_DIRECT", dest='directio')
    parser.add_argument(
        "-t", "--sync-time", default=None, type=int,
        help="sleep till sime utc time", dest='sync_time')
    parser.add_argument(
        "--iozone-url", help="iozone static binary url",
        dest="iozone_url", default=None)
    parser.add_argument(
        "--test-file", help="file path to run test on",
        default=None, dest='test_file')
    parser.add_argument(
        "--iozone-path", help="iozone path",
        default=None, dest='iozone_path')
    return parser.parse_args(argv)


def main(argv):
    argv_obj = parse_args(argv)
    argv_obj.blocksize = ssize_to_kb(argv_obj.blocksize)

    if argv_obj.iosize is not None:
        argv_obj.iosize = ssize_to_kb(argv_obj.iosize)

    benchmark = BenchmarkOption(1,
                                argv_obj.iodepth,
                                argv_obj.action,
                                argv_obj.blocksize,
                                argv_obj.iosize)

    benchmark.direct_io = argv_obj.directio

    test_file_name = argv_obj.test_file
    if test_file_name is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_file_name = os.tmpnam()

    remove_iozone_bin = False
    if argv_obj.iozone_url is not None:
        if argv_obj.iozone_path is not None:
            sys.stderr.write("At most one option from --iozone-path and "
                             "--iozone-url should be provided")
            exit(1)
        iozone_binary = os.tmpnam()
        install_iozone_static(argv_obj.iozone_url, iozone_binary)
        remove_iozone_bin = True
    elif argv_obj.iozone_path is not None:
        iozone_binary = argv_obj.iozone_path
        if os.path.isfile(iozone_binary):
            if not os.access(iozone_binary, os.X_OK):
                st = os.stat(iozone_binary)
                os.chmod(iozone_binary, st.st_mode | stat.S_IEXEC)

    else:
        iozone_binary = which('iozone')

        if iozone_binary is None:
            iozone_binary = which('iozone3')

        if iozone_binary is None:
            sys.stderr.write("Can't found neither iozone not iozone3 binary"
                             "Provide --iozone-path or --iozone-url option")
            exit(1)

    try:
        if argv_obj.sync_time is not None:
            dt = argv_obj.sync_time - time.time()
            if dt > 0:
                time.sleep(dt)

        res = run_iozone(benchmark, iozone_binary, test_file_name)
        sys.stdout.write(json.dumps(res) + "\n")
    finally:
        if remove_iozone_bin:
            os.unlink(iozone_binary)

        if os.path.isfile(test_file_name):
            os.unlink(test_file_name)


# function-marker for patching, don't 'optimize' it
def INSERT_IOZONE_ARGS(x):
    return x


if __name__ == '__main__':
    # this line would be patched in case of run under rally
    # don't modify it!
    argv = INSERT_IOZONE_ARGS(sys.argv[1:])
    exit(main(argv))
