import multiprocessing


def get_barrier(count):
    val = multiprocessing.Value('i', count)
    cond = multiprocessing.Condition()

    def closure(timeout):
        with cond:
            val.value -= 1
            if val.value == 0:
                cond.notify_all()
            else:
                cond.wait(timeout)
            return val.value == 0

    return closure
