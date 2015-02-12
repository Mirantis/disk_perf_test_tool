import os
import sys
import datetime


def _log(x):
    dt_str = datetime.datetime.now().strftime("%H:%M:%S")
    pref = dt_str + " " + str(os.getpid()) + " >>>> "
    sys.stderr.write(pref + x.replace("\n", "\n" + pref) + "\n")


logger = _log


def log(x):
    logger(x)


def setlogger(nlogger):
    global logger
    logger = nlogger
