import array
import threading


class SensorDatastore(object):
    def __init__(self, stime=None):
        self.lock = threading.Lock()
        self.stime = stime

        self.min_size = 60 * 60
        self.max_size = 60 * 61

        self.data = {
            'testnodes:io': array.array("B"),
            'testnodes:cpu': array.array("B"),
        }

    def get_values(self, name, start, end):
        assert end >= start

        if end == start:
            return []

        with self.lock:
            curr_arr = self.data[name]
            if self.stime is None:
                return []

            sidx = start - self.stime
            eidx = end - self.stime

            if sidx < 0 and eidx < 0:
                return [0] * (end - start)
            elif sidx < 0:
                return [0] * (-sidx) + curr_arr[:eidx]
            return curr_arr[sidx:eidx]

    def update_values(self, data_time, vals, add=False):
        with self.lock:
            if self.stime is None:
                self.stime = data_time

            for name, value in vals.items():
                curr_arr = self.data.setdefault(name, array.array("H"))
                curr_end_time = len(curr_arr) + self.stime

                dtime = data_time - curr_end_time

                if dtime > 0:
                    curr_arr.extend([0] * dtime)
                    curr_arr.append(value)
                elif dtime == 0:
                    curr_arr.append(value)
                else:
                    # dtime < 0
                    sindex = len(curr_arr) + dtime
                    if sindex > 0:
                        if add:
                            curr_arr[sindex] += value
                        else:
                            curr_arr[sindex].append(value)
