import re
import json
import time
import errno
import socket
import shutil
import logging
import os.path
import getpass
import StringIO
import threading
import subprocess

import paramiko

from agent import connect
from .ssh_utils import Local, ssh_connect, ssh_copy_file


logger = logging.getLogger("wally")


def setup_node(conn, agent_path, ip):
    agent_fname, log_fname = run_over_ssh(conn, "mktemp;echo;mktemp").strip().split()
    with conn.open_sftp() as sftp:
        ssh_copy_file(sftp, agent_path, agent_fname)

    cmd = "python {} server -d --listen-addr={}:0 --stdout-file={}"
    jdata = run_over_ssh(conn, cmd.format(agent_fname, ip, log_fname)).strip()
    run_over_ssh(conn, "rm {}".format(agent_fname))
    sett = json.loads(jdata)
    return connect(sett['addr'])


def exists(rpc, path):
    """os.path.exists for paramiko's SCP object"""
    return rpc.exists(path)


def save_to_remote(sftp, path, content):
    with sftp.open(path, "wb") as fd:
        fd.write(content)


def read_from_remote(sftp, path):
    with sftp.open(path, "rb") as fd:
        return fd.read()


def normalize_dirpath(dirpath):
    while dirpath.endswith("/"):
        dirpath = dirpath[:-1]
    return dirpath


ALL_RWX_MODE = ((1 << 9) - 1)


def ssh_mkdir(sftp, remotepath, mode=ALL_RWX_MODE, intermediate=False):
    remotepath = normalize_dirpath(remotepath)
    if intermediate:
        try:
            sftp.mkdir(remotepath, mode=mode)
        except (IOError, OSError):
            upper_dir = remotepath.rsplit("/", 1)[0]

            if upper_dir == '' or upper_dir == '/':
                raise

            ssh_mkdir(sftp, upper_dir, mode=mode, intermediate=True)
            return sftp.mkdir(remotepath, mode=mode)
    else:
        sftp.mkdir(remotepath, mode=mode)


def ssh_copy_file(sftp, localfile, remfile, preserve_perm=True):
    sftp.put(localfile, remfile)
    if preserve_perm:
        sftp.chmod(remfile, os.stat(localfile).st_mode & ALL_RWX_MODE)


def put_dir_recursively(sftp, localpath, remotepath, preserve_perm=True):
    "upload local directory to remote recursively"

    # hack for localhost connection
    if hasattr(sftp, "copytree"):
        sftp.copytree(localpath, remotepath)
        return

    assert remotepath.startswith("/"), "%s must be absolute path" % remotepath

    # normalize
    localpath = normalize_dirpath(localpath)
    remotepath = normalize_dirpath(remotepath)

    try:
        sftp.chdir(remotepath)
        localsuffix = localpath.rsplit("/", 1)[1]
        remotesuffix = remotepath.rsplit("/", 1)[1]
        if localsuffix != remotesuffix:
            remotepath = os.path.join(remotepath, localsuffix)
    except IOError:
        pass

    for root, dirs, fls in os.walk(localpath):
        prefix = os.path.commonprefix([localpath, root])
        suffix = root.split(prefix, 1)[1]
        if suffix.startswith("/"):
            suffix = suffix[1:]

        remroot = os.path.join(remotepath, suffix)

        try:
            sftp.chdir(remroot)
        except IOError:
            if preserve_perm:
                mode = os.stat(root).st_mode & ALL_RWX_MODE
            else:
                mode = ALL_RWX_MODE
            ssh_mkdir(sftp, remroot, mode=mode, intermediate=True)
            sftp.chdir(remroot)

        for f in fls:
            remfile = os.path.join(remroot, f)
            localfile = os.path.join(root, f)
            ssh_copy_file(sftp, localfile, remfile, preserve_perm)


def delete_file(conn, path):
    sftp = conn.open_sftp()
    sftp.remove(path)
    sftp.close()


def copy_paths(conn, paths):
    sftp = conn.open_sftp()
    try:
        for src, dst in paths.items():
            try:
                if os.path.isfile(src):
                    ssh_copy_file(sftp, src, dst)
                elif os.path.isdir(src):
                    put_dir_recursively(sftp, src, dst)
                else:
                    templ = "Can't copy {0!r} - " + \
                            "it neither a file not a directory"
                    raise OSError(templ.format(src))
            except Exception as exc:
                tmpl = "Scp {0!r} => {1!r} failed - {2!r}"
                raise OSError(tmpl.format(src, dst, exc))
    finally:
        sftp.close()


class ConnCreds(object):
    conn_uri_attrs = ("user", "passwd", "host", "port", "path")

    def __init__(self):
        for name in self.conn_uri_attrs:
            setattr(self, name, None)

    def __str__(self):
        return str(self.__dict__)


uri_reg_exprs = []


class URIsNamespace(object):
    class ReParts(object):
        user_rr = "[^:]*?"
        host_rr = "[^:@]*?"
        port_rr = "\\d+"
        key_file_rr = "[^:@]*"
        passwd_rr = ".*?"

    re_dct = ReParts.__dict__

    for attr_name, val in re_dct.items():
        if attr_name.endswith('_rr'):
            new_rr = "(?P<{0}>{1})".format(attr_name[:-3], val)
            setattr(ReParts, attr_name, new_rr)

    re_dct = ReParts.__dict__

    templs = [
        "^{host_rr}$",
        "^{host_rr}:{port_rr}$",
        "^{host_rr}::{key_file_rr}$",
        "^{host_rr}:{port_rr}:{key_file_rr}$",
        "^{user_rr}@{host_rr}$",
        "^{user_rr}@{host_rr}:{port_rr}$",
        "^{user_rr}@{host_rr}::{key_file_rr}$",
        "^{user_rr}@{host_rr}:{port_rr}:{key_file_rr}$",
        "^{user_rr}:{passwd_rr}@{host_rr}$",
        "^{user_rr}:{passwd_rr}@{host_rr}:{port_rr}$",
    ]

    for templ in templs:
        uri_reg_exprs.append(templ.format(**re_dct))


def parse_ssh_uri(uri):
    # user:passwd@ip_host:port
    # user:passwd@ip_host
    # user@ip_host:port
    # user@ip_host
    # ip_host:port
    # ip_host
    # user@ip_host:port:path_to_key_file
    # user@ip_host::path_to_key_file
    # ip_host:port:path_to_key_file
    # ip_host::path_to_key_file

    if uri.startswith("ssh://"):
        uri = uri[len("ssh://"):]

    res = ConnCreds()
    res.port = "22"
    res.key_file = None
    res.passwd = None
    res.user = getpass.getuser()

    for rr in uri_reg_exprs:
        rrm = re.match(rr, uri)
        if rrm is not None:
            res.__dict__.update(rrm.groupdict())
            return res

    raise ValueError("Can't parse {0!r} as ssh uri value".format(uri))


def reconnect(conn, uri, **params):
    if uri == 'local':
        return conn

    creds = parse_ssh_uri(uri)
    creds.port = int(creds.port)
    return ssh_connect(creds, reuse_conn=conn, **params)


def connect(uri, **params):
    if uri == 'local':
        res = Local()
    else:
        creds = parse_ssh_uri(uri)
        creds.port = int(creds.port)
        res = ssh_connect(creds, **params)
    return res


all_sessions_lock = threading.Lock()
all_sessions = {}


class BGSSHTask(object):
    CHECK_RETRY = 5

    def __init__(self, node, use_sudo):
        self.node = node
        self.pid = None
        self.use_sudo = use_sudo

    def start(self, orig_cmd, **params):
        uniq_name = 'test'
        cmd = "screen -S {0} -d -m {1}".format(uniq_name, orig_cmd)
        run_over_ssh(self.node.connection, cmd,
                     timeout=10, node=self.node.get_conn_id(),
                     **params)
        processes = run_over_ssh(self.node.connection, "ps aux", nolog=True)

        for iter in range(self.CHECK_RETRY):
            for proc in processes.split("\n"):
                if orig_cmd in proc and "SCREEN" not in proc:
                    self.pid = proc.split()[1]
                    break
            if self.pid is not None:
                break
            time.sleep(1)

        if self.pid is None:
            self.pid = -1

    def check_running(self):
        assert self.pid is not None
        if -1 == self.pid:
            return False
        try:
            run_over_ssh(self.node.connection,
                         "ls /proc/{0}".format(self.pid),
                         timeout=10, nolog=True)
            return True
        except OSError:
            return False

    def kill(self, soft=True, use_sudo=True):
        assert self.pid is not None
        if self.pid == -1:
            return True
        try:
            if soft:
                cmd = "kill {0}"
            else:
                cmd = "kill -9 {0}"

            if self.use_sudo:
                cmd = "sudo " + cmd

            run_over_ssh(self.node.connection,
                         cmd.format(self.pid), nolog=True)
            return True
        except OSError:
            return False

    def wait(self, soft_timeout, timeout):
        end_of_wait_time = timeout + time.time()
        soft_end_of_wait_time = soft_timeout + time.time()

        # time_till_check = random.randint(5, 10)
        time_till_check = 2

        # time_till_first_check = random.randint(2, 6)
        time_till_first_check = 2
        time.sleep(time_till_first_check)
        if not self.check_running():
            return True

        while self.check_running() and time.time() < soft_end_of_wait_time:
            # time.sleep(soft_end_of_wait_time - time.time())
            time.sleep(time_till_check)

        while end_of_wait_time > time.time():
            time.sleep(time_till_check)
            if not self.check_running():
                break
        else:
            self.kill()
            time.sleep(1)
            if self.check_running():
                self.kill(soft=False)
            return False
        return True


def run_over_ssh(conn, cmd, stdin_data=None, timeout=60,
                 nolog=False, node=None):
    "should be replaces by normal implementation, with select"

    if isinstance(conn, Local):
        if not nolog:
            logger.debug("SSH:local Exec {0!r}".format(cmd))
        proc = subprocess.Popen(cmd, shell=True,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

        stdoutdata, _ = proc.communicate(input=stdin_data)
        if proc.returncode != 0:
            templ = "SSH:{0} Cmd {1!r} failed with code {2}. Output: {3}"
            raise OSError(templ.format(node, cmd, proc.returncode, stdoutdata))

        return stdoutdata

    transport = conn.get_transport()
    session = transport.open_session()

    if node is None:
        node = ""

    with all_sessions_lock:
        all_sessions[id(session)] = session

    try:
        session.set_combine_stderr(True)

        stime = time.time()

        if not nolog:
            logger.debug("SSH:{0} Exec {1!r}".format(node, cmd))

        session.exec_command(cmd)

        if stdin_data is not None:
            session.sendall(stdin_data)

        session.settimeout(1)
        session.shutdown_write()
        output = ""

        while True:
            try:
                ndata = session.recv(1024)
                output += ndata
                if "" == ndata:
                    break
            except socket.timeout:
                pass

            if time.time() - stime > timeout:
                raise OSError(output + "\nExecution timeout")

        code = session.recv_exit_status()
    finally:
        found = False
        with all_sessions_lock:
            if id(session) in all_sessions:
                found = True
                del all_sessions[id(session)]

        if found:
            session.close()

    if code != 0:
        templ = "SSH:{0} Cmd {1!r} failed with code {2}. Output: {3}"
        raise OSError(templ.format(node, cmd, code, output))

    return output


def close_all_sessions():
    with all_sessions_lock:
        for session in all_sessions.values():
            try:
                session.sendall('\x03')
                session.close()
            except:
                pass
        all_sessions.clear()
