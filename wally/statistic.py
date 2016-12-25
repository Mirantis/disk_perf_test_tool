import math
import itertools
import statistics
from typing import Union, List, TypeVar, Callable, Iterable, Tuple, Any, cast, Dict

import numpy
from scipy import stats, optimize
from numpy import linalg
from numpy.polynomial.chebyshev import chebfit, chebval


from .result_classes import IStorable


Number = Union[int, float]
TNumber = TypeVar('TNumber', int, float)


DOUBLE_DELTA = 1e-8


average = statistics.mean
dev = statistics.variance


class StatProps(IStorable):
    "Statistic properties for timeserie"

    yaml_tag = 'stat'

    def __init__(self, data: List[Number]) -> None:
        self.average = None  # type: float
        self.deviation = None  # type: float
        self.confidence = None  # type: float
        self.confidence_level = None  # type: float

        self.perc_99 = None  # type: float
        self.perc_95 = None  # type: float
        self.perc_90 = None  # type: float
        self.perc_50 = None   # type: float

        self.min = None  # type: Number
        self.max = None  # type: Number

        # bin_center: bin_count
        self.histo = None  # type: Tuple[List[int], List[float]]
        self.data = data

        self.normtest = None  # type: Any

    def __str__(self) -> str:
        res = ["StatProps(num_el={}):".format(len(self.data)),
               "    distr = {0.average} ~ {0.deviation}".format(self),
               "    confidence({0.confidence_level}) = {0.confidence}".format(self),
               "    perc50={0.perc50}".format(self),
               "    perc90={0.perc90}".format(self),
               "    perc95={0.perc95}".format(self),
               "    perc95={0.perc99}".format(self),
               "    range {0.min} {0.max}".format(self),
               "    nurmtest = {0.nortest}".format(self)]
        return "\n".join(res)

    def __repr__(self) -> str:
        return str(self)

    def raw(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'StatProps':
        res = cls.__new__(cls)
        res.__dict__.update(data)
        return res


def greater_digit_pos(val: Number) -> int:
    return int(math.floor(math.log10(val))) + 1


def round_digits(val: TNumber, num_digits: int = 3) -> TNumber:
    pow = 10 ** (greater_digit_pos(val) - num_digits)
    return type(val)(int(val / pow) * pow)


def calc_stat_props(data: List[Number], confidence: float = 0.95) -> StatProps:
    "Calculate statistical properties of array of numbers"

    res = StatProps(data)

    if len(data) == 0:
        raise ValueError("Input array is empty")

    data = sorted(data)
    res.average = average(data)
    res.deviation = dev(data)
    res.max = data[-1]
    res.min = data[0]

    res.perc_50 = numpy.percentile(data, 50)
    res.perc_90 = numpy.percentile(data, 90)
    res.perc_95 = numpy.percentile(data, 95)
    res.perc_99 = numpy.percentile(data, 99)

    if len(data) >= 3:
        res.confidence = stats.sem(data) * \
                         stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    else:
        res.confidence = None

    res.histo = numpy.histogram(data, 'auto')
    res.normtest = stats.mstats.normaltest(data)
    return res


def groupby_globally(data: Iterable, key_func: Callable):
    grouped = {}  # type: ignore
    grouped_iter = itertools.groupby(data, key_func)

    for (bs, cache_tp, act, conc), curr_data_it in grouped_iter:
        key = (bs, cache_tp, act, conc)
        grouped.setdefault(key, []).extend(curr_data_it)

    return grouped


def approximate_curve(x: List[Number], y: List[float], xnew: List[Number], curved_coef: int) -> List[float]:
    """returns ynew - y values of some curve approximation"""
    return cast(List[float], chebval(xnew, chebfit(x, y, curved_coef)))


def approximate_line(x: List[Number], y: List[float], xnew: List[Number], relative_dist: bool = False) -> List[float]:
    """
    x, y - test data, xnew - dots, where we want find approximation
    if not relative_dist distance = y - newy
    returns ynew - y values of linear approximation
    """
    ox = numpy.array(x)
    oy = numpy.array(y)

    # set approximation function
    def func_line(tpl, x):
        return tpl[0] * x + tpl[1]

    def error_func_rel(tpl, x, y):
        return 1.0 - y / func_line(tpl, x)

    def error_func_abs(tpl, x, y):
        return y - func_line(tpl, x)

    # choose distance mode
    error_func = error_func_rel if relative_dist else error_func_abs

    tpl_initial = tuple(linalg.solve([[ox[0], 1.0], [ox[1], 1.0]],
                                     oy[:2]))

    # find line
    tpl_final, success = optimize.leastsq(error_func, tpl_initial[:], args=(ox, oy))

    # if error
    if success not in range(1, 5):
        raise ValueError("No line for this dots")

    # return new dots
    return func_line(tpl_final, numpy.array(xnew))


# TODO: revise next
# def difference(y, ynew):
#     """returns average and maximum relative and
#        absolute differences between y and ynew
#        result may contain None values for y = 0
#        return value - tuple:
#        [(abs dif, rel dif) * len(y)],
#        (abs average, abs max),
#        (rel average, rel max)"""
#
#     abs_dlist = []
#     rel_dlist = []
#
#     for y1, y2 in zip(y, ynew):
#         # absolute
#         abs_dlist.append(y1 - y2)
#
#         if y1 > 1E-6:
#             rel_dlist.append(abs(abs_dlist[-1] / y1))
#         else:
#             raise ZeroDivisionError("{0!r} is too small".format(y1))
#
#     da_avg = sum(abs_dlist) / len(abs_dlist)
#     dr_avg = sum(rel_dlist) / len(rel_dlist)
#
#     return (zip(abs_dlist, rel_dlist),
#             (da_avg, max(abs_dlist)), (dr_avg, max(rel_dlist))
#             )
