# BLOCK_SIZES = "1k 4k 64k 256k 1m"
# OPERATIONS="randwrite write randread read"
# SYNC_TYPES="s a d"
# REPEAT_COUNT="3"
# CONCURRENCES="1 8 64"

from utils import ssize_to_kb

SYNC_FACTOR = "x500"
DIRECT_FACTOR = "x10000"
ASYNC_FACTOR = "r2"


def make_list(x):
    if not isinstance(x, (list, tuple)):
        return [x]
    return x

HDD_SIZE_KB = 45 * 1000 * 1000


def make_load(sizes, opers, sync_types, concurrence,
              tester_type='iozone', repeat_count=3):

    iodepth = 1
    for conc in make_list(concurrence):
        for bsize in make_list(sizes):
            for oper in make_list(opers):
                for sync_type in make_list(sync_types):

                    # filter out too slow options
                    if bsize in "1k 4k" and sync_type == "a":
                        continue

                    # filter out sync reads
                    if oper in "read randread" and sync_type == "s":
                        continue

                    if sync_type == "s":
                        size_sync_opts = "--iosize {0} -s".format(SYNC_FACTOR)
                    elif sync_type == "d":
                        if oper == 'randread':
                            assert SYNC_FACTOR[0] == 'x'
                            max_f = int(SYNC_FACTOR[1:])
                        else:
                            max_f = None

                        mmax_f = HDD_SIZE_KB / (int(conc) * ssize_to_kb(bsize))

                        if max_f is None or mmax_f > max_f:
                            max_f = mmax_f

                        assert DIRECT_FACTOR[0] == 'x'
                        if max_f > int(DIRECT_FACTOR[1:]):
                            max_f = DIRECT_FACTOR
                        else:
                            max_f = "x{0}".format(max_f)

                        size_sync_opts = "--iosize {0} -d".format(max_f)

                    else:
                        size_sync_opts = "--iosize {0}".format(ASYNC_FACTOR)

                    # size_sync_opts = get_file_size_opts(sync_type)

                    io_opts = "--type {0} ".format(tester_type)
                    io_opts += "-a {0} ".format(oper)
                    io_opts += "--iodepth {0} ".format(iodepth)
                    io_opts += "--blocksize {0} ".format(bsize)
                    io_opts += size_sync_opts + " "
                    io_opts += "--concurrency {0}".format(conc)

                    for i in range(repeat_count):
                        yield io_opts


sizes = "4k 64k 2m".split()
opers = "randwrite write randread read".split()
sync_types = "s a d".split()
concurrence = "1 8 64".split()

for io_opts in make_load(sizes=sizes, concurrence=concurrence,
                         sync_types=sync_types, opers=opers):
    print io_opts
