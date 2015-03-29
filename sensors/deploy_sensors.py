import time
import json
import os.path
import logging

from concurrent.futures import ThreadPoolExecutor, wait

from disk_perf_test_tool.ssh_utils import connect, copy_paths

logger = logging.getLogger('io-perf-tool')


def wait_all_ok(futures):
    return all(future.result() for future in futures)


def deploy_and_start_sensors(monitor_uri, config,
                             remote_path='/tmp/sensors',
                             connected_config=None):
    paths = {os.path.dirname(__file__): remote_path}
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []

        if connected_config is not None:
            assert config is None
            node_iter = connected_config
        else:
            node_iter = config.items()

        for uri_or_conn, config in node_iter:
            futures.append(executor.submit(deploy_and_start_sensor,
                                           paths, uri_or_conn,
                                           monitor_uri,
                                           config, remote_path))

        if not wait_all_ok(futures):
            raise RuntimeError("Sensor deployment fails on some nodes")


def deploy_and_start_sensor(paths, uri_or_conn, monitor_uri, config,
                            remote_path):
    try:
        if isinstance(uri_or_conn, basestring):
            conn = connect(uri_or_conn)
        else:
            conn = uri_or_conn

        copy_paths(conn, paths)
        sftp = conn.open_sftp()

        config_remote_path = os.path.join(remote_path, "conf.json")
        main_remote_path = os.path.join(remote_path, "main.py")

        with sftp.open(config_remote_path, "w") as fd:
            fd.write(json.dumps(config))

        cmd_templ = "python {0} -d start -u {1} {2}"
        cmd = cmd_templ.format(main_remote_path,
                               monitor_uri,
                               config_remote_path)
        conn.exec_command(cmd)
        sftp.close()

        if isinstance(uri_or_conn, basestring):
            conn.close()
    except:
        logger.exception("During deploing sensors in {0}".format(uri_or_conn))
        return False
    return True


def stop_and_remove_sensor(uri_or_conn, remote_path):
    if isinstance(uri_or_conn, basestring):
        conn = connect(uri_or_conn)
    else:
        conn = uri_or_conn

    main_remote_path = os.path.join(remote_path, "main.py")

    cmd_templ = "python {0} -d stop"
    conn.exec_command(cmd_templ.format(main_remote_path))

    # some magic
    time.sleep(0.3)

    conn.exec_command("rm -rf {0}".format(remote_path))

    if isinstance(uri_or_conn, basestring):
        conn.close()


def stop_and_remove_sensors(config, remote_path='/tmp/sensors',
                            connected_config=None):
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []

        if connected_config is not None:
            assert config is None
            conf_iter = connected_config
        else:
            conf_iter = config.items()

        for uri_or_conn, config in conf_iter:
            futures.append(executor.submit(stop_and_remove_sensor,
                                           uri_or_conn, remote_path))

        wait(futures)
