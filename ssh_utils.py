import re
import time
import socket
import logging
import os.path
import getpass
import threading


import paramiko


logger = logging.getLogger("io-perf-tool")


def ssh_connect(creds, retry_count=60, timeout=1):
    ssh = paramiko.SSHClient()
    ssh.load_host_keys('/dev/null')
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.known_hosts = None
    for i in range(retry_count):
        try:
            if creds.user is None:
                user = getpass.getuser()
            else:
                user = creds.user

            if creds.passwd is not None:
                ssh.connect(creds.host,
                            username=user,
                            password=creds.passwd,
                            port=creds.port,
                            allow_agent=False,
                            look_for_keys=False)
                return ssh

            if creds.key_file is not None:
                ssh.connect(creds.host,
                            username=user,
                            key_filename=creds.key_file,
                            look_for_keys=False,
                            port=creds.port)
                return ssh

            key_file = os.path.expanduser('~/.ssh/id_rsa')
            ssh.connect(creds.host,
                        username=user,
                        key_filename=key_file,
                        look_for_keys=False,
                        port=creds.port)
            return ssh
            # raise ValueError("Wrong credentials {0}".format(creds.__dict__))
        except paramiko.PasswordRequiredException:
            raise
        except socket.error:
            if i == retry_count - 1:
                raise
            time.sleep(timeout)


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
        except IOError:
            ssh_mkdir(sftp, remotepath.rsplit("/", 1)[0], mode=mode,
                      intermediate=True)
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
                    msg = templ.format(src)
                    raise OSError(msg)
            except Exception as exc:
                tmpl = "Scp {0!r} => {1!r} failed - {2!r}"
                msg = tmpl.format(src, dst, exc)
                raise OSError(msg)
    finally:
        sftp.close()


class ConnCreds(object):
    conn_uri_attrs = ("user", "passwd", "host", "port", "path")

    def __init__(self):
        for name in self.conn_uri_attrs:
            setattr(self, name, None)


uri_reg_exprs = []


class URIsNamespace(object):
    class ReParts(object):
        user_rr = "[^:]*?"
        host_rr = "[^:]*?"
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
        "^{user_rr}@{host_rr}::{key_file_rr}$",
        "^{user_rr}@{host_rr}:{port_rr}:{key_file_rr}$",
        "^{user_rr}:{passwd_rr}@@{host_rr}$",
        "^{user_rr}:{passwd_rr}@@{host_rr}:{port_rr}$",
    ]

    for templ in templs:
        uri_reg_exprs.append(templ.format(**re_dct))


def parse_ssh_uri(uri):
    # user:passwd@@ip_host:port
    # user:passwd@@ip_host
    # user@ip_host:port
    # user@ip_host
    # ip_host:port
    # ip_host
    # user@ip_host:port:path_to_key_file
    # user@ip_host::path_to_key_file
    # ip_host:port:path_to_key_file
    # ip_host::path_to_key_file

    res = ConnCreds()
    res.port = "22"
    res.key_file = None
    res.passwd = None

    for rr in uri_reg_exprs:
        rrm = re.match(rr, uri)
        if rrm is not None:
            res.__dict__.update(rrm.groupdict())
            return res

    raise ValueError("Can't parse {0!r} as ssh uri value".format(uri))


def connect(uri):
    creds = parse_ssh_uri(uri)
    creds.port = int(creds.port)
    return ssh_connect(creds)


all_sessions_lock = threading.Lock()
all_sessions = []


def run_over_ssh(conn, cmd, stdin_data=None, timeout=60, nolog=False, node=None):
    "should be replaces by normal implementation, with select"
    transport = conn.get_transport()
    session = transport.open_session()

    if node is None:
        node = ""

    with all_sessions_lock:
        all_sessions.append(session)

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
        session.close()

    if code != 0:
        templ = "SSH:{0} Cmd {1!r} failed with code {2}. Output: {3}"
        raise OSError(templ.format(node, cmd, code, output))

    return output


def close_all_sessions():
    with all_sessions_lock:
        for session in all_sessions:
            try:
                session.sendall('\x03')
                session.close()
            except:
                pass
