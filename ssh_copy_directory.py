import os.path


def normalize_dirpath(dirpath):
    while dirpath.endswith("/"):
        dirpath = dirpath[:-1]
    return dirpath


def ssh_mkdir(sftp, remotepath, mode=0777, intermediate=False):
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
        sftp.chmod(remfile, os.stat(localfile).st_mode & 0777)


def put_dir_recursively(sftp, localpath, remotepath, preserve_perm=True):
    "upload local directory to remote recursively"

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
                mode = os.stat(root).st_mode & 0777
            else:
                mode = 0777
            ssh_mkdir(sftp, remroot, mode=mode, intermediate=True)
            sftp.chdir(remroot)

        for f in fls:
            remfile = os.path.join(remroot, f)
            localfile = os.path.join(root, f)
            ssh_copy_file(sftp, localfile, remfile, preserve_perm)
