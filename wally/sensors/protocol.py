import sys
import time
import struct
import socket
import select
import cPickle as pickle
from urlparse import urlparse


class Timeout(Exception):
    pass


class CantUnpack(Exception):
    pass


# ------------------------------------- Serializers --------------------------


class ISensortResultsSerializer(object):
    def pack(self, data):
        pass

    def unpack(self, data):
        pass


class StructSerializerSend(ISensortResultsSerializer):
    initial_times = 5
    resend_timeout = 60
    HEADERS = 'h'
    DATA = 'd'
    END_OF_HEADERS = '\x00'
    END_OF_SOURCE_ID = '\x00'
    HEADERS_SEPARATOR = '\n'

    def __init__(self):
        self.field_order = None
        self.headers_send_cycles_left = self.initial_times
        self.pack_fmt = None
        self.next_header_send_time = None

    def pack(self, data):
        data = data.copy()

        source_id = data.pop("source_id")
        vals = [int(data.pop("time").value)]

        if self.field_order is None:
            self.field_order = sorted(data.keys())
            self.pack_fmt = "!I" + "I" * len(self.field_order)

        need_resend = False
        if self.next_header_send_time is not None:
            if time.time() > self.next_header_send_time:
                need_resend = True

        if self.headers_send_cycles_left > 0 or need_resend:
            forder = self.HEADERS_SEPARATOR.join(self.field_order)
            flen = struct.pack("!H", len(self.field_order))

            result = (self.HEADERS + source_id +
                      self.END_OF_SOURCE_ID +
                      socket.gethostname() +
                      self.END_OF_SOURCE_ID +
                      flen + forder + self.END_OF_HEADERS)

            if self.headers_send_cycles_left > 0:
                self.headers_send_cycles_left -= 1

            self.next_header_send_time = time.time() + self.resend_timeout
        else:
            result = ""

        for name in self.field_order:
            vals.append(int(data[name].value))

        packed_data = self.DATA + source_id
        packed_data += self.END_OF_SOURCE_ID
        packed_data += struct.pack(self.pack_fmt, *vals)

        return result + packed_data


class StructSerializerRecv(ISensortResultsSerializer):
    def __init__(self):
        self.fields = {}
        self.formats = {}
        self.hostnames = {}

    def unpack(self, data):
        code = data[0]

        if code == StructSerializerSend.HEADERS:
            source_id, hostname, packed_data = data[1:].split(
                StructSerializerSend.END_OF_SOURCE_ID, 2)
            # fields order provided
            flen_sz = struct.calcsize("!H")
            flen = struct.unpack("!H", packed_data[:flen_sz])[0]

            headers_data, rest = packed_data[flen_sz:].split(
                StructSerializerSend.END_OF_HEADERS, 1)

            forder = headers_data.split(
                StructSerializerSend.HEADERS_SEPARATOR)

            assert len(forder) == flen, \
                "Wrong len {0} != {1}".format(len(forder), flen)

            if 'source_id' in self.fields:
                assert self.fields[source_id] == ['time'] + forder,\
                    "New field order"
            else:
                self.fields[source_id] = ['time'] + forder
                self.formats[source_id] = "!I" + "I" * flen
                self.hostnames[source_id] = hostname

            if len(rest) != 0:
                return self.unpack(rest)
            return None
        else:
            source_id, packed_data = data[1:].split(
                StructSerializerSend.END_OF_SOURCE_ID, 1)
            assert code == StructSerializerSend.DATA,\
                "Unknown code {0!r}".format(code)

            try:
                fields = self.fields[source_id]
            except KeyError:
                raise CantUnpack("No fields order provided"
                                 " for {0} yet".format(source_id))
            s_format = self.formats[source_id]

            exp_size = struct.calcsize(s_format)
            assert len(packed_data) == exp_size, \
                "Wrong data len {0} != {1}".format(len(packed_data), exp_size)

            vals = struct.unpack(s_format, packed_data)
            res = dict(zip(fields, vals))
            res['source_id'] = source_id
            res['hostname'] = self.hostnames[source_id]
            return res


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

        if receiver:
            packer_cls = StructSerializerRecv
        else:
            packer_cls = StructSerializerSend

        return UDPTransport(receiver, ip=ip, port=int(port),
                            packer_cls=packer_cls)
    elif parsed_uri.scheme == 'file':
        return FileTransport(receiver, parsed_uri.path)
    else:
        templ = "Can't instantiate transport from {0!r}"
        raise ValueError(templ.format(uri))
