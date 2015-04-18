#!/usr/bin/env python
""" Protocol class """

import re
import zlib
import json
import logging
import binascii


logger = logging.getLogger("wally.sensors")


# protocol contains 2 type of packet:
# 1 - header, which contains template schema of counters
# 2 - body, which contains only values in order as in template
#       it uses msgpack (or provided packer) for optimization
#
# packet has format:
# begin_data_prefixSIZE\n\nDATAend_data_postfix
# packet part has format:
# SIZE\n\rDATA
#
# DATA use archivation


class PacketException(Exception):
    """ Exceptions from Packet"""
    pass


class Packet(object):
    """ Class proceed packet by protocol"""

    prefix = "begin_data_prefix"
    postfix = "end_data_postfix"
    header_prefix = "template"
    # other fields
    # is_begin
    # is_end
    # crc
    # data
    # data_len

    def __init__(self, packer):
        # preinit
        self.is_begin = False
        self.is_end = False
        self.crc = None
        self.data = ""
        self.data_len = None
        self.srv_template = None
        self.clt_template = None
        self.tmpl_size = 0
        self.packer = packer

    def new_packet(self, part):
        """ New packet adding """
        # proceed packet
        try:
            # get size
            local_size_s, _, part = part.partition("\n\r")
            local_size = int(local_size_s)

            # find prefix
            begin = part.find(self.prefix)
            if begin != -1:
                # divide data if something before begin and prefix
                from_i = begin + len(self.prefix)
                part = part[from_i:]
                # reset flags
                self.is_begin = True
                self.is_end = False
                self.data = ""
                # get size
                data_len_s, _, part = part.partition("\n\r")
                self.data_len = int(data_len_s)
                # get crc
                crc_s, _, part = part.partition("\n\r")
                self.crc = int(crc_s)

            # bad size?
            if local_size != self.data_len:
                raise PacketException("Part size error")

            # find postfix
            end = part.find(self.postfix)
            if end != -1:
                # divide postfix
                part = part[:end]
                self.is_end = True

            self.data += part
            # check if it is end
            if self.is_end:
                self.data = zlib.decompress(self.data)
                if self.data_len != len(self.data):
                    raise PacketException("Total size error")
                if binascii.crc32(self.data) != self.crc:
                    raise PacketException("CRC error")

                # check, if it is template
                if self.data.startswith(self.header_prefix):
                    self.srv_template = self.data
                    # template is for internal use
                    return None

                # decode values list
                vals = self.packer.unpack(self.data)
                dump = self.srv_template % tuple(vals)
                return dump
            else:
                return None

        except PacketException as e:
            # if something wrong - skip packet
            logger.warning("Packet skipped: %s", e)
            self.is_begin = False
            self.is_end = False
            return None

        except TypeError:
            # if something wrong - skip packet
            logger.warning("Packet skipped: doesn't match schema")
            self.is_begin = False
            self.is_end = False
            return None

        except:
            # if something at all wrong - skip packet
            logger.warning("Packet skipped: something is wrong")
            self.is_begin = False
            self.is_end = False
            return None

    @staticmethod
    def create_packet(data, part_size):
        """ Create packet divided by parts with part_size from data
            No compression here """
        # prepare data
        data_len = "%i\n\r" % len(data)
        header = "%s%s%s\n\r" % (Packet.prefix, data_len, binascii.crc32(data))
        compact_data = zlib.compress(data)
        packet = "%s%s%s" % (header, compact_data, Packet.postfix)

        partheader_len = len(data_len)

        beg = 0
        end = part_size - partheader_len

        result = []
        while beg < len(packet):
            block = packet[beg:beg+end]
            result.append(data_len + block)
            beg += end

        return result

    def create_packet_v2(self, data, part_size):
        """ Create packet divided by parts with part_size from data
            Compressed """
        result = []
        # create and add to result template header
        if self.srv_template is None:
            perf_string = json.dumps(data)
            self.create_answer_template(perf_string)
            template = self.header_prefix + self.srv_template
            header = Packet.create_packet(template, part_size)
            result.extend(header)

        vals = self.get_matching_value_list(data)
        body = self.packer.pack(vals)
        parts = Packet.create_packet(body, part_size)
        result.extend(parts)
        return result

    def get_matching_value_list(self, data):
        """ Get values in order server expect"""
        vals = range(0, self.tmpl_size)

        try:
            for node, groups in self.clt_template.items():
                for group, counters in groups.items():
                    for counter, index in counters.items():
                        if not isinstance(index, dict):
                            vals[index] = data[node][group][counter]
                        else:
                            for k, i in index.items():
                                vals[i] = data[node][group][counter][k]

            return vals

        except (IndexError, KeyError):
            logger = logging.getLogger(__name__)
            logger.error("Data don't match last schema")
            raise PacketException("Data don't match last schema")

    def create_answer_template(self, perf_string):
        """ Create template for server to insert counter values
            Return tuple of server and clien templates + number of replaces"""
        replacer = re.compile(": [0-9]+\.?[0-9]*")
        # replace all values by %s
        finditer = replacer.finditer(perf_string)
        # server not need know positions
        self.srv_template = ""
        # client need positions
        clt_template = ""
        beg = 0
        k = 0
        # this could be done better?
        for match in finditer:
            # define input place in server template
            self.srv_template += perf_string[beg:match.start()]
            self.srv_template += ": %s"
            # define match number in client template
            clt_template += perf_string[beg:match.start()]
            clt_template += ": %i" % k

            beg = match.end()
            k += 1

        # add tail
        self.srv_template += perf_string[beg:]
        clt_template += perf_string[beg:]

        self.tmpl_size = k
        self.clt_template = json.loads(clt_template)
