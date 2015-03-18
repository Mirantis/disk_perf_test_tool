#!/usr/bin/env python
""" UDP sender class """

import socket
import urlparse

from cp_protocol import Packet
from logger import define_logger


class SenderException(Exception):
    """ Exceptions in Sender class """
    pass


class Timeout(Exception):
    """ Exceptions in Sender class """
    pass


class Sender(object):
    """ UDP sender class """

    def __init__(self, packer, url=None, port=None, host="0.0.0.0", size=256):
        """ Create connection object from input udp string or params"""

        # test input
        if url is None and port is None:
            raise SenderException("Bad initialization")
        if url is not None:
            data = urlparse.urlparse(url)
            # check schema
            if data.scheme != "udp":
                mes = "Bad protocol type: %s instead of UDP" % data.scheme
                logger.error(mes)
                raise SenderException("Bad protocol type")
            # try to get port
            try:
                int_port = int(data.port)
            except ValueError:
                logger.error("Bad UDP port")
                raise SenderException("Bad UDP port")
            # save paths
            self.sendto = (data.hostname, int_port)
            self.bindto = (data.hostname, int_port)
            # try to get size
            try:
                self.size = int(data.path.strip("/"))
            except ValueError:
                logger.error("Bad packet part size")
                raise SenderException("Bad packet part size")
        else:
            # url is None - use size and port
            self.sendto = (host, port)
            self.bindto = ("0.0.0.0", port)
            self.size = size

        self.packer = packer

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.binded = False
        self.all_data = {}
        self.send_packer = None


    def bind(self):
        """ Prepare for listening """
        self.sock.bind(self.bindto)
        self.sock.settimeout(0.5)
        self.binded = True


    def send(self, data):
        """ Send data to udp socket"""
        if self.sock.sendto(data, self.sendto) != len(data):
            mes = "Cannot send data to %s:%s" % self.sendto
            logger.error(mes)
            raise SenderException("Cannot send data")


    def send_by_protocol(self, data):
        """ Send data by Packet protocol
            data = dict"""
        if self.send_packer is None:
            self.send_packer = Packet(self.packer())
        parts = self.send_packer.create_packet_v2(data, self.size)
        for part in parts:
            self.send(part)


    def recv(self):
        """ Receive data from udp socket"""
        # check for binding
        if not self.binded:
            self.bind()
        # try to recv
        try:
            data, (remote_ip, _) = self.sock.recvfrom(self.size)
            return data, remote_ip
        except socket.timeout:
            raise Timeout()


    def recv_by_protocol(self):
        """ Receive data from udp socket by Packet protocol"""
        data, remote_ip = self.recv()

        if remote_ip not in self.all_data:
            self.all_data[remote_ip] = Packet(self.packer())

        return self.all_data[remote_ip].new_packet(data)


    def recv_with_answer(self, stop_event=None):
        """ Receive data from udp socket and send 'ok' back
            Command port = local port + 1
            Answer port = local port
            Waiting for command is blocking """
        # create command socket
        command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        command_port = self.bindto[1]+1
        command_sock.bind(("0.0.0.0", command_port))
        command_sock.settimeout(1)
        # try to recv
        while True:
            try:
                data, (remote_ip, _) = command_sock.recvfrom(self.size)
                self.send("ok")
                return data, remote_ip
            except socket.timeout:
                if stop_event is not None and stop_event.is_set():
                    # return None if we are interrupted
                    return None


    def verified_send(self, send_host, message, max_repeat=20):
        """ Send and verify it by answer not more then max_repeat
            Send port = local port + 1
            Answer port = local port
            Return True if send is verified """
        # create send socket
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        send_port = self.sendto[1]+1
        for repeat in range(0, max_repeat):
            send_sock.sendto(message, (send_host, send_port))
            try:
                data, remote_ip = self.recv()
                if remote_ip == send_host and data == "ok":
                    return True
                else:
                    logger.warning("No answer from %s, try %i", send_host, repeat)
            except Timeout:
                logger.warning("No answer from %s, try %i", send_host, repeat)

        return False



logger = define_logger(__name__)
