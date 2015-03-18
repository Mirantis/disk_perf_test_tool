import time
import socket
import select
import cPickle as pickle
from urlparse import urlparse

import cp_transport


class Timeout(Exception):
    pass


# ------------------------------------- Serializers --------------------------


class ISensortResultsSerializer(object):
    def pack(self, data):
        pass

    def unpack(self, data):
        pass


class PickleSerializer(ISensortResultsSerializer):
    def pack(self, data):
        ndata = {key: val.value for key, val in data.items()}
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
            raise ValueError("StdoutTransport don't allows receiving")

        self.headers = None
        self.line_format = ""
        self.prev = {}
        self.delta = delta

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

            self.prev.update({header: data[header].value
                              for header in self.headers})
        else:
            vals = [data[header].value for header in self.headers]

        print self.line_format.format(*vals)

    def recv(self, timeout=None):
        raise ValueError("StdoutTransport don't allows receiving")


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


class HugeUDPTransport(ITransport, cp_transport.Sender):
    def __init__(self, receiver, ip, port):
        cp_transport.Sender.__init__(self, port=port, host=ip)
        if receiver:
            self.bind()

    def send(self, data):
        self.send_by_protocol(data)

    def recv(self, timeout=None):
        begin = time.time()

        while True:

            try:
                # return not None, if packet is ready
                ready = self.recv_by_protocol()
                # if data ready - return it
                if ready is not None:
                    return ready
                # if data not ready - check if it's time to die
                if time.time() - begin >= timeout:
                    break

            except cp_transport.Timeout:
                # no answer yet - check, if timeout end
                if time.time() - begin >= timeout:
                    break
# -------------------------- Factory function --------------------------------


def create_protocol(uri, receiver=False):
    parsed_uri = urlparse(uri)
    if parsed_uri.scheme == 'stdout':
        return StdoutTransport(receiver)
    elif parsed_uri.scheme == 'udp':
        ip, port = parsed_uri.netloc.split(":")
        return UDPTransport(receiver, ip=ip, port=int(port),
                            packer_cls=PickleSerializer)
    elif parsed_uri.scheme == 'hugeudp':
        ip, port = parsed_uri.netloc.split(":")
        return HugeUDPTransport(receiver, ip=ip, port=int(port))
    else:
        templ = "Can't instantiate transport from {0!r}"
        raise ValueError(templ.format(uri))
