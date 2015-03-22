import sys
import argparse

from disk_perf_test_tool.utils import ssize_to_b


def make_list(x):
    if not isinstance(x, (list, tuple)):
        return [x]
    return x


def make_load(settings):

    iodepth = 1
    for conc in make_list(settings.concurrences):
        for bsize in make_list(settings.sizes):
            for oper in make_list(settings.opers):
                for cache_mode in make_list(settings.cache_modes):

                    # filter out too slow options
                    if bsize in "1k 4k" and cache_mode == "a":
                        continue

                    # filter out sync reads
                    if oper in "read randread" and cache_mode == "s":
                        continue

                    if settings.io_size is not None:
                        size_sync_opts = " --iosize " + str(settings.io_size)
                        if cache_mode == "s":
                            size_sync_opts += " -s"
                        elif cache_mode == "d":
                            size_sync_opts += " -d"
                    else:
                        if cache_mode == "s":
                            size_sync_opts = "--iosize {0} -s".format(
                                settings.sync_default_size)
                        elif cache_mode == "d":
                            if oper == 'randread':
                                assert settings.sync_default_size[0] == 'x'
                                max_f = int(settings.sync_default_size[1:])
                            else:
                                max_f = None

                            mmax_f = ssize_to_b(settings.hdd_size) / \
                                (int(conc) * ssize_to_b(bsize))

                            if max_f is None or mmax_f > max_f:
                                max_f = mmax_f

                            assert settings.direct_default_size[0] == 'x'
                            if max_f > int(settings.direct_default_size[1:]):
                                max_f = settings.direct_default_size
                            else:
                                max_f = "x{0}".format(max_f)

                            size_sync_opts = "--iosize {0} -d".format(max_f)

                        else:
                            if oper == 'randread' or oper == 'read':
                                size_sync_opts = "--iosize " + \
                                    str(settings.sync_default_size)
                            else:
                                size_sync_opts = "--iosize " + \
                                    str(settings.sync_default_size)

                    # size_sync_opts = get_file_size_opts(sync_type)

                    io_opts = "--type {0} ".format(settings.tester_type)
                    io_opts += "-a {0} ".format(oper)
                    io_opts += "--iodepth {0} ".format(iodepth)
                    io_opts += "--blocksize {0} ".format(bsize)
                    io_opts += size_sync_opts + " "
                    io_opts += "--concurrency {0}".format(conc)

                    for i in range(settings.repeats):
                        yield io_opts


def parse_opts(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs="+", required=True)
    parser.add_argument('--opers', nargs="+", required=True)
    parser.add_argument('--cache-modes', nargs="+", required=True)
    parser.add_argument('--concurrences', nargs="+", required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument("--hdd-size", default="45G")
    parser.add_argument("--tester-type", default="iozone")
    parser.add_argument("--io-size", default=None)

    parser.add_argument("--direct-default-size", default="x1000")
    parser.add_argument("--sync-default-size", default="x1000")
    parser.add_argument("--async-default-size", default="r2")

    return parser.parse_args(args[1:])


def main(args):
    opts = parse_opts(args)
    for io_opts in make_load(opts):
        print "python io.py --test-file /opt/xxx.bin " + io_opts

if __name__ == "__main__":
    exit(main(sys.argv))
