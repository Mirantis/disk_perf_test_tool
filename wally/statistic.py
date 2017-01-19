import math
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
MIN_VALUES_FOR_CONFIDENCE = 7


average = numpy.mean
dev = lambda x: math.sqrt(numpy.var(x, ddof=1))


def calc_norm_stat_props(ts: TimeSeries, bins_count: int, confidence: float = 0.95) -> NormStatProps:
    "Calculate statistical properties of array of numbers"

    # array.array has very basic support
    data = cast(List[int], ts.data)
    res = NormStatProps(data)  # type: ignore

    if len(data) == 0:
        raise ValueError("Input array is empty")

    data = sorted(data)
    res.average = average(data)
    res.deviation = dev(data)

    res.max = data[-1]
    res.min = data[0]

    res.perc_50, res.perc_90, res.perc_99, res.perc_99 = numpy.percentile(data, q=[50., 90., 95., 99.])

    if len(data) >= MIN_VALUES_FOR_CONFIDENCE:
        res.confidence = stats.sem(data) * \
                         stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        res.confidence_level = confidence
    else:
        res.confidence = None
        res.confidence_level = None

    res.bins_populations, bins_edges = numpy.histogram(data, bins=bins_count)
    res.bins_mids = (bins_edges[:-1] + bins_edges[1:]) / 2

    try:
        res.normtest = stats.mstats.normaltest(data)
    except Exception as exc:
        logger.warning("stats.mstats.normaltest failed with error: %s", exc)

    res.skew = stats.skew(data)
    res.kurt = stats.kurtosis(data)

    return res


def calc_histo_stat_props(ts: TimeSeries,
                          bins_edges: numpy.array,
                          bins_count: int,
                          min_valuable: float = 0.0001) -> HistoStatProps:
    data = numpy.array(ts.data, dtype='int')
    data.shape = [len(ts.data) // ts.second_axis_size, ts.second_axis_size]  # type: ignore

    res = HistoStatProps(ts.data, ts.second_axis_size)

    # summ across all series
    aggregated = numpy.sum(data, axis=0, dtype='int')
    total = numpy.sum(aggregated)

    # minimal value used for histo
    min_val_on_histo = total * min_valuable

    # percentiles levels
    expected = [total * 0.5, total * 0.9, total * 0.95, total * 0.99]
    percentiles = []

    # all indexes, where values greater than min_val_on_histo
    valuable_idxs = []

    curr_summ = 0
    non_zero = aggregated.nonzero()[0]

    # calculate percentiles and valuable_indexes
    for idx in non_zero:
        val = aggregated[idx]
        while expected and curr_summ + val >= expected[0]:
            percentiles.append(bins_edges[idx])
            del expected[0]

        curr_summ += val

        if val >= min_val_on_histo:
            valuable_idxs.append(idx)

    res.perc_50, res.perc_90, res.perc_95, res.perc_99 = percentiles

    # minimax and maximal non-zero elements
    res.min = bins_edges[aggregated[non_zero[0]]]
    res.max = bins_edges[non_zero[-1] + (1 if non_zero[-1] != len(bins_edges) else 0)]

    # minimal and maximal valueble evelemts
    val_idx_min = valuable_idxs[0]
    val_idx_max = valuable_idxs[-1]

    raw_bins_populations = aggregated[val_idx_min: val_idx_max + 1]
    raw_bins_edges = bins_edges[val_idx_min: val_idx_max + 2]
    raw_bins_mids = cast(numpy.array, (raw_bins_edges[1:] + raw_bins_edges[:-1]) / 2)

    step = (raw_bins_mids[-1] + raw_bins_mids[0]) / bins_count
    next = raw_bins_mids[0]

    # aggregate raw histogram with many bins into result histogram with bins_count bins
    cidx = 0
    bins_populations = []
    bins_mids = []

    while cidx < len(raw_bins_mids):
        next += step
        bin_population = 0

        while cidx < len(raw_bins_mids) and raw_bins_mids[cidx] <= next:
            bin_population += raw_bins_populations[cidx]
            cidx += 1

        bins_populations.append(bin_population)
        bins_mids.append(next - step / 2)

    res.bins_populations = numpy.array(bins_populations, dtype='int')
    res.bins_mids = numpy.array(bins_mids, dtype='float32')

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
