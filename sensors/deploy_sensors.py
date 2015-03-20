import time
import json
import os.path

from ssh_copy_directory import copy_paths
from ssh_runner import connect

from concurrent.futures import ThreadPoolExecutor, wait


def wait_all_ok(futures):
    return all(future.result() for future in futures)


def deploy_and_start_sensors(monitor_uri, config, remote_path='/tmp/sensors'):
    paths = {os.path.dirname(__file__): remote_path}
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []

        for uri, config in config.items():
            futures.append(executor.submit(deploy_and_start_sensor,
                                           paths, uri, monitor_uri,
                                           config, remote_path))

        if not wait_all_ok(futures):
            raise RuntimeError("Sensor deployment fails on some nodes")


def deploy_and_start_sensor(paths, uri, monitor_uri, config, remote_path):
    try:
        conn = connect(uri)
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
        conn.close()
    except:
        return False
    return True


def stop_and_remove_sensor(uri, remote_path):
    conn = connect(uri)
    main_remote_path = os.path.join(remote_path, "main.py")

    cmd_templ = "python {0} -d stop"
    conn.exec_command(cmd_templ.format(main_remote_path))

    # some magic
    time.sleep(0.3)

    conn.exec_command("rm -rf {0}".format(remote_path))

    conn.close()


def stop_and_remove_sensors(config, remote_path='/tmp/sensors'):
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []

        for uri, config in config.items():
            futures.append(executor.submit(stop_and_remove_sensor,
                                           uri, remote_path))

        wait(futures)
