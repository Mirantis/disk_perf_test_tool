import os
import time
import unittest


from oktest import ok, main, test


from wally import utils, ssh_utils


class AgentTest(unittest.TestCase):
    @test("test_local_executor_ls")
    def test_ls(self):
        expected = sorted(os.listdir('/'))
        ok(sorted(utils.run_locally('ls /').split())) == expected

    @test("test_local_executor_sleep1")
    def test_sleep1(self):
        t = time.time()
        with self.assertRaises(RuntimeError):
            utils.run_locally(['sleep', '20'], timeout=1)
        ok(time.time() - t) < 1.2

    @test("test_local_executor_sleep2")
    def test_sleep2(self):
        t = time.time()
        with self.assertRaises(RuntimeError):
            utils.run_locally('sleep 20', timeout=1)
        ok(time.time() - t) < 1.2

    @test("test_ssh_executor1")
    def test_ssh_executor1(self):
        id_rsa_path = os.path.expanduser('~/.ssh/id_rsa')
        ssh_url = "ssh://localhost::" + id_rsa_path
        expected = sorted(os.listdir('/'))

        conn = ssh_utils.connect(ssh_url)
        out = ssh_utils.run_over_ssh(conn, "ls /")
        ok(sorted(out.split())) == expected

if __name__ == '__main__':
    main()
