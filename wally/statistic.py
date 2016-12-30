import math
import array
import logging
import itertools
import statistics
from typing import List, Callable, Iterable, cast

import numpy
from scipy import stats, optimize
from numpy import linalg
from numpy.polynomial.chebyshev import chebfit, chebval


from .result_classes import NormStatProps, HistoStatProps, TimeSeries
from .utils import Number


logger = logging.getLogger("wally")
DOUBLE_DELTA = 1e-8


average = statistics.mean
dev = lambda x: math.sqrt(statistics.variance(x))


def calc_norm_stat_props(ts: TimeSeries, confidence: float = 0.95) -> NormStatProps:
    "Calculate statistical properties of array of numbers"

    data = ts.data
    res = NormStatProps(data)

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

    res.bin_populations, res.bin_edges = numpy.histogram(data, 'auto')

    try:
        res.normtest = stats.mstats.normaltest(data)
    except Exception as exc:
        logger.warning("stats.mstats.normaltest failed with error: %s", exc)

    return res


def calc_histo_stat_props(ts: TimeSeries) -> HistoStatProps:
    data = numpy.array(ts.data)
    data.shape = [len(ts.data) // ts.second_axis_size, ts.second_axis_size]

    res = HistoStatProps(ts.data, ts.second_axis_size)
    aggregated = numpy.sum(data, axis=0)

    full_sum = numpy.sum(aggregated)
    expected = [full_sum * 0.5, full_sum * 0.9, full_sum * 0.95, full_sum * 0.99]
    percentiles = []

    val_min = None
    val_max = None

    for idx, val in enumerate(aggregated):
        while expected and full_sum + val >= expected[0]:
            percentiles.append(idx)
            del expected[0]

        full_sum += val

        if val != 0:
            if val_min is None:
                val_min = idx
            val_max = idx

    res.perc_50, res.perc_90, res.perc_95, res.perc_99 = map(ts.bins_edges.__getitem__, percentiles)
    res.min = ts.bins_edges[val_min]
    res.max = ts.bins_edges[val_max]
    res.bin_populations = aggregated
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
