all_sensors = {}


def provides(sensor_class_name):
    def closure(func):
        assert sensor_class_name not in all_sensors
        all_sensors[sensor_class_name] = func
        return func
    return closure
