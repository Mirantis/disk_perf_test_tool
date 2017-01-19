import logging
import collections
from typing import Dict, Sequence, NamedTuple

from .node_interfaces import IRPCNode

logger = logging.getLogger("wally")


def log_nodes_statistic(nodes: Sequence[IRPCNode]) -> None:
    logger.info("Found {0} nodes total".format(len(nodes)))

    per_role = collections.defaultdict(int)  # type: Dict[str, int]
    for node in nodes:
        for role in node.info.roles:
            per_role[role] += 1

    for role, count in sorted(per_role.items()):
        logger.debug("Found {0} nodes with role {1}".format(count, role))



OSRelease = NamedTuple("OSRelease",
                       [("distro", str),
                        ("release", str),
                        ("arch", str)])


def get_os(node: IRPCNode) -> OSRelease:
    """return os type, release and architecture for node.
    """
    arch = node.run("arch", nolog=True).strip()

    try:
        node.run("ls -l /etc/redhat-release", nolog=True)
        return OSRelease('redhat', None, arch)
    except:
        pass

    try:
        node.run("ls -l /etc/debian_version", nolog=True)

        release = None
        for line in node.run("lsb_release -a", nolog=True).split("\n"):
            if ':' not in line:
                continue
            opt, val = line.split(":", 1)

            if opt == 'Codename':
                release = val.strip()

        return OSRelease('ubuntu', release, arch)
    except:
        pass

    raise RuntimeError("Unknown os")
