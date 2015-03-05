import json
import sys

import run_test


logger = run_test.logger
tool = None


class FakeVMContext(object):
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return ["fake@fake"]

    def __exit__(self, *args, **kwargs):
        pass


def fake_start_vms(*args, **kwargs):
    return FakeVMContext


class FakeFD(object):
    def __init__(self, content):
        self.content = content
        self.channel = FakeChannel()

    def read(self):
        return self.content


class FakeChannel(object):
    def recv_exit_status(self):
        return 0


def get_fake_out(cmd):
    empty_fd = FakeFD("")
    if "pgbench" == tool:
        if "run" in cmd:
            out = FakeFD("2 1:43\n2 1:42\n4 2:77")
        else:
            out = empty_fd
    elif "iozone" == tool or "fio" == tool:
        data = {'__meta__': {
            'direct_io': 1,
            'action': 'r',
            'concurence': 1,
            'blocksize': 1,
            'sync': 's'},
                 'bw_mean': 10}
        out = FakeFD(json.dumps(data))
    else:
        raise Exception("tool not found")
    err = empty_fd
    return empty_fd, out, err


def fake_ssh_connect(*args, **kwargs):
    return FakeSSH()


class FakeSFTP(object):
    def put(self, what, where):
        logger.debug("Called sftp put with %s %s" % (what, where))

    def chmod(self, f, mode):
        logger.debug("called sftp chmod %s %s" % (mode, f))

    def close(self):
        logger.debug("called sftp close")


class FakeSSH(object):
    def exec_command(self, cmd, **kwargs):
        return get_fake_out(cmd)

    def close(self):
        pass

    def open_sftp(self):
        return FakeSFTP()


class FakePopen(object):
    def __init__(self, cmd,
                 shell=True,
                 stdout=None,
                 stderr=None,
                 stdin=None):
        print "Running subprocess command: %s" % cmd
        self.stdin, self.stdout, self.stderr = get_fake_out(cmd)

    def wait(self):
        return 0


if __name__ == '__main__':
    run_test.subprocess.Popen = FakePopen
    run_test.start_test_vms = fake_start_vms()
    run_test.ssh_runner.ssh_connect = fake_ssh_connect
    opts = run_test.parse_args(sys.argv[1:])
    tool = opts.tool_type
    exit(run_test.main(sys.argv[1:]))
