import getpass

from oktest import ok

from wally import ssh_utils, ssh


creds = "root@osd-0"


def test_ssh_url_parser():
    curr_user = getpass.getuser()
    creds = {
        "test": ssh_utils.ConnCreds("test", curr_user, port=23),
        "test:13": ssh_utils.ConnCreds("test", curr_user, port=13),
        "test::xxx.key": ssh_utils.ConnCreds("test", curr_user, port=23, key_file="xxx.key"),
        "test:123:xxx.key": ssh_utils.ConnCreds("test", curr_user, port=123, key_file="xxx.key"),
        "user@test": ssh_utils.ConnCreds("test", "user", port=23),
        "user@test:13": ssh_utils.ConnCreds("test", "user", port=13),
        "user@test::xxx.key": ssh_utils.ConnCreds("test", "user", port=23, key_file="xxx.key"),
        "user@test:123:xxx.key": ssh_utils.ConnCreds("test", "user", port=123, key_file="xxx.key"),
        "user:passwd:@test": ssh_utils.ConnCreds("test", curr_user, port=23, passwd="passwd:"),
        "user:passwd:@test:123": ssh_utils.ConnCreds("test", curr_user, port=123, passwd="passwd:"),
    }

    for uri, expected in creds.items():
        parsed = ssh_utils.parse_ssh_uri(uri)
        ok(parsed.user) == expected.user
        ok(parsed.addr.port) == expected.addr.port
        ok(parsed.addr.host) == expected.addr.host
        ok(parsed.key_file) == expected.key_file
        ok(parsed.passwd) == expected.passwd

