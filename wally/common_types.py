import abc
from typing import NamedTuple, Dict, Any

IP = str
IPAddr = NamedTuple("IPAddr", [("host", IP), ("port", int)])
