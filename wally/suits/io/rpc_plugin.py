def rpc_run_fio(cfg):
    fio_cmd_templ = "cd {exec_folder}; {fio_path}fio --output-format=json " + \
                    "--output={out_file} --alloc-size=262144 {job_file}"

    # fnames_before = node.run("ls -1 " + exec_folder, nolog=True)
    #
    # timeout = int(exec_time + max(300, exec_time))
    # soft_end_time = time.time() + exec_time
    # logger.error("Fio timeouted on node {}. Killing it".format(node))
    # end = time.time()
    # fnames_after = node.run("ls -1 " + exec_folder, nolog=True)
    #

def parse_fio_result(data):
    pass
