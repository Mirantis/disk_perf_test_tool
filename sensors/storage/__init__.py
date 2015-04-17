import struct


def pack(val, tp=True):
    if isinstance(val, int):
        assert 0 <= val < 2 ** 16

        if tp:
            res = 'i'
        else:
            res = ""

        res += struct.pack("!U", val)
    elif isinstance(val, dict):
        assert len(val) < 2 ** 16
        if tp:
            res = "d"
        else:
            res = ""

        res += struct.pack("!U", len(val))
        for k, v in dict.items():
            assert 0 <= k < 2 ** 16
            assert 0 <= v < 2 ** 32
            res += struct.pack("!UI", k, v)
    elif isinstance(val, str):
        assert len(val) < 256
        if tp:
            res = "s"
        else:
            res = ""
        res += chr(len(val)) + val
    else:
        raise ValueError()

    return res


def unpack(fd, tp=None):
    if tp is None:
        tp = fd.read(1)

    if tp == 'i':
        return struct.unpack("!U", fd.read(2))
    elif tp == 'd':
        res = {}
        val_len = struct.unpack("!U", fd.read(2))
        for _ in range(val_len):
            k, v = struct.unpack("!UI", fd.read(6))
            res[k] = v
        return res
    elif tp == 's':
        val_len = struct.unpack("!U", fd.read(2))
        return fd.read(val_len)

    raise ValueError()


class LocalStorage(object):
    NEW_DATA = 0
    NEW_SENSOR = 1
    NEW_SOURCE = 2

    def __init__(self, fd):
        self.fd = fd
        self.sensor_ids = {}
        self.sources_ids = {}
        self.max_source_id = 0
        self.max_sensor_id = 0

    def add_data(self, source, sensor_values):
        source_id = self.sources_ids.get(source)
        if source_id is None:
            source_id = self.max_source_id
            self.sources_ids[source] = source_id
            self.emit(self.NEW_SOURCE, source_id, source)
            self.max_source_id += 1

        new_sensor_values = {}

        for name, val in sensor_values.items():
            sensor_id = self.sensor_ids.get(name)
            if sensor_id is None:
                sensor_id = self.max_sensor_id
                self.sensor_ids[name] = sensor_id
                self.emit(self.NEW_SENSOR, sensor_id, name)
                self.max_sensor_id += 1
            new_sensor_values[sensor_id] = val

        self.emit(self.NEW_DATA, source_id, new_sensor_values)

    def emit(self, tp, v1, v2):
        self.fd.write(chr(tp) + pack(v1, False) + pack(v2))

    def readall(self):
        tp = self.fd.read(1)
        if ord(tp) == self.NEW_DATA:
            pass
        elif ord(tp) == self.NEW_SENSOR:
            pass
        elif ord(tp) == self.NEW_SOURCE:
            pass
        else:
            raise ValueError()
