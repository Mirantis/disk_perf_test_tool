from typing import NamedTuple

IP = str
IPAddr = NamedTuple("IPAddr", [("host", IP), ("port", int)])
