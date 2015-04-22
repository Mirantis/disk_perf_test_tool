import sys
import time
import struct
import socket
import select
import cPickle as pickle
from urlparse import urlparse

from . import cp_transport


class Timeout(Exception):
    pass


# ------------------------------------- Serializers --------------------------


class ISensortResultsSerializer(object):
    def pack(self, data):
        pass

    def unpack(self, data):
        pass


class StructSerializer(ISensortResultsSerializer):
    class LocalConfig(object):
        def __init__(self):
            self.last_format_sent = -1
            self.initial_sent = False
            self.initial_times = 5
            self.field_order = None

    def __init__(self):
        self.configs = {}

    def pack(self, data):
        OLD_FORMAT = 5
        source_id = data["source_id"]
        config = self.configs.setdefault(source_id,
                                         StructSerializer.LocalConfig())

        if config.field_order is None or \
           not config.initial_sent or \
           time.time() - config.last_format_sent > OLD_FORMAT:
           # send|resend format
            field_order = sorted(data.keys())

            config.field_order = field_order
            config.last_format_sent = time.time()
            if not config.initial_sent:
                config.initial_times -= 1
                config.initial_sent = (config.initial_times <= 0)

            forder = "\n".join(field_order)
            flen = struct.pack("!H", len(field_order))
            return "\x00{0}\x00{1}{2}".format(source_id, flen, forder)
        else:
            # send data
            # time will be first after source_id
            vals = [data["time"]]
            for name in config.field_order:
                if name in data:
                    vals.append(data[name])
            pack_fmt = "!" + ("I" * len(vals))
            packed_data = struct.pack(pack_fmt, vals)
            return "\x01{0}\x00{1}".format(source_id, packed_data)

    def unpack(self, data):
        code = data[0]
        data = data[1:]
        source_id, _, packed_data = data.partition("\x00")
        config = self.configs.setdefault(source_id,
                                         StructSerializer.LocalConfig())
        unpacked_data = {"source_id":source_id}

        if code == "\x00":
            # fields order provided
            flen = struct.unpack("!H", packed_data[:2])
            forder = packed_data[2:].split("\n")
            if len(forder) != flen:
                return unpacked_data
            config.field_order = forder
            return unpacked_data

        else:
            # data provided
            # try to find fields_order
            if config.field_order is None:
                raise ValueError("No fields order provided"
                                 " for {0}, cannot unpack".format(source_id))

            val_size = 4
            if len(packed_data) % val_size != 0:
                raise ValueError("Bad packet received"
                                 " from {0}, cannot unpack".format(source_id))
            datalen = len(packed_data) / val_size
            pack_fmt = "!" + ("I" * datalen)
            vals = struct.unpack(pack_fmt, packed_data)

            unpacked_data['time'] = vals[0]
            i = 1
            for field in config.field_order:
                data[field] = vals[i]
                i += 1
            return data


class PickleSerializer(ISensortResultsSerializer):
    def pack(self, data):
        ndata = {}
        for key, val in data.items():
            if isinstance(val, basestring):
                ndata[key] = val
            else:
                ndata[key] = val.value
        return pickle.dumps(ndata)

    def unpack(self, data):
        return pickle.loads(data)

# ------------------------------------- Transports ---------------------------


class ITransport(object):
    def __init__(self, receiver):
        pass

    def send(self, data):
        pass

    def recv(self, timeout=None):
        pass


class StdoutTransport(ITransport):
    MIN_COL_WIDTH = 10

    def __init__(self, receiver, delta=True):
        if receiver:
            cname = self.__class__.__name__
            raise ValueError("{0} don't allows receiving".format(cname))

        self.headers = None
        self.line_format = ""
        self.prev = {}
        self.delta = delta
        self.fd = sys.stdout

    def send(self, data):
        if self.headers is None:
            self.headers = sorted(data)

            for pos, header in enumerate(self.headers):
                self.line_format += "{%s:>%s}" % (pos,
                                                  max(len(header) + 1,
                                                      self.MIN_COL_WIDTH))

            print self.line_format.format(*self.headers)

        if self.delta:
            vals = [data[header].value - self.prev.get(header, 0)
                    for header in self.headers]

            self.prev.update(dict((header, data[header].value)
                             for header in self.headers))
        else:
            vals = [data[header].value for header in self.headers]

        self.fd.write(self.line_format.format(*vals) + "\n")

    def recv(self, timeout=None):
        cname = self.__class__.__name__
        raise ValueError("{0} don't allows receiving".format(cname))


class FileTransport(StdoutTransport):
    def __init__(self, receiver, fname, delta=True):
        StdoutTransport.__init__(self, receiver, delta)
        self.fd = open(fname, "w")


class UDPTransport(ITransport):
    def __init__(self, receiver, ip, port, packer_cls):
        self.port = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if receiver:
            self.port.bind((ip, port))
            self.packer_cls = packer_cls
            self.packers = {}
        else:
            self.packer = packer_cls()
            self.dst = (ip, port)

    def send(self, data):
        raw_data = self.packer.pack(data)
        self.port.sendto(raw_data, self.dst)

    def recv(self, timeout=None):
        r, _, _ = select.select([self.port], [], [], timeout)
        if len(r) != 0:
            raw_data, addr = self.port.recvfrom(10000)
            packer = self.packers.setdefault(addr, self.packer_cls())
            return addr, packer.unpack(raw_data)
        else:
            raise Timeout()


# -------------------------- Factory function --------------------------------


def create_protocol(uri, receiver=False):
    parsed_uri = urlparse(uri)
    if parsed_uri.scheme == 'stdout':
        return StdoutTransport(receiver)
    elif parsed_uri.scheme == 'udp':
        ip, port = parsed_uri.netloc.split(":")
        return UDPTransport(receiver, ip=ip, port=int(port),
                            packer_cls=StructSerializer)
    elif parsed_uri.scheme == 'file':
        return FileTransport(receiver, parsed_uri.path)
    else:
        templ = "Can't instantiate transport from {0!r}"
        raise ValueError(templ.format(uri))
