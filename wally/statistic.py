import math
import logging
import itertools
from typing import List, Callable, Iterable, cast, Tuple

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

    pcs = numpy.percentile(data, q=[1.0, 5.0, 10., 50., 90., 95., 99.])
    res.perc_1, res.perc_5, res.perc_10, res.perc_50, res.perc_90, res.perc_95, res.perc_99 = pcs

    if len(data) >= MIN_VALUES_FOR_CONFIDENCE:
        res.confidence = stats.sem(data) * \
                         stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        res.confidence_level = confidence
    else:
        res.confidence = None
        res.confidence_level = None

    res.bins_populations, res.bins_edges = numpy.histogram(data, bins=bins_count)
    res.bins_edges = res.bins_edges[:-1]

    try:
        res.normtest = stats.mstats.normaltest(data)
    except Exception as exc:
        logger.warning("stats.mstats.normaltest failed with error: %s", exc)

    res.skew = stats.skew(data)
    res.kurt = stats.kurtosis(data)

    return res


# update this code
def rebin_histogram(bins_populations: numpy.array,
                    bins_edges: numpy.array,
                    new_bins_count: int,
                    left_tail_idx: int = None,
                    right_tail_idx: int = None,
                    log_bins: bool = False) -> Tuple[numpy.array, numpy.array]:
    # rebin large histogram into smaller with new_bins bins, linearly distributes across
    # left_tail_idx:right_tail_idx range

    assert len(bins_populations.shape) == 1
    assert len(bins_edges.shape) == 1
    assert bins_edges.shape[0] == bins_populations.shape[0]

    if left_tail_idx is None:
        min_val = bins_edges[0]
    else:
        min_val = bins_edges[left_tail_idx]

    if right_tail_idx is None:
        max_val = bins_edges[-1]
    else:
        max_val = bins_edges[right_tail_idx]

    if log_bins:
        assert min_val > 1E-3
        step = (max_val / min_val) ** (1 / new_bins_count)
        new_bins_edges = min_val * (step ** numpy.arange(new_bins_count))  # type: numpy.array
    else:
        new_bins_edges = numpy.linspace(min_val, max_val, new_bins_count + 1, dtype='float')[:-1]  # type: numpy.array

    old_bins_pos = numpy.searchsorted(new_bins_edges, bins_edges, side='right')
    new_bins = numpy.zeros(new_bins_count, dtype=int)  # type: numpy.array

    # last source bin can't be split
    # TODO: need to add assert for this
    new_bins[-1] += bins_populations[-1]
    bin_sizes = bins_edges[1:] - bins_edges[:-1]

    # correct position to get bin idx from edge idx
    old_bins_pos -= 1
    old_bins_pos[old_bins_pos < 0] = 0
    new_bins_sizes = new_bins_edges[1:] - new_bins_edges[:-1]

    for population, begin, end, bsize in zip(bins_populations[:-1], old_bins_pos[:-1], old_bins_pos[1:], bin_sizes):
        if begin == end:
            new_bins[begin] += population
        else:
            density = population / bsize
            for curr_box in range(begin, end):
                cnt = min(int(new_bins_sizes[begin] * density + 0.5), population)
                new_bins[begin] += cnt
                population -= cnt

    return new_bins, new_bins_edges


def calc_histo_stat_props(ts: TimeSeries,
                          bins_edges: numpy.array,
                          rebins_count: int,
                          tail: float = 0.005) -> HistoStatProps:
    log_bins = False
    res = HistoStatProps(ts.data)

    # summ across all series
    aggregated = ts.data.sum(axis=0, dtype='int')
    total = aggregated.sum()

    # percentiles levels
    expected = list(numpy.array([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]) * total)
    cumsum = numpy.cumsum(aggregated)

    percentiles_bins = numpy.searchsorted(cumsum, expected)
    percentiles = bins_edges[percentiles_bins]
    res.perc_1, res.perc_5, res.perc_10, res.perc_50, res.perc_90, res.perc_95, res.perc_99 = percentiles

    # don't show tail ranges on histogram
    left_tail_idx, right_tail_idx = numpy.searchsorted(cumsum, [tail * total, (1 - tail) * total])

    # minimax and maximal non-zero elements
    non_zero = numpy.nonzero(aggregated)[0]
    res.min = bins_edges[aggregated[non_zero[0]]]
    res.max = bins_edges[non_zero[-1] + (1 if non_zero[-1] != len(bins_edges) else 0)]

    res.log_bins = False
    res.bins_populations, res.bins_edges = rebin_histogram(aggregated, bins_edges, rebins_count,
                                                           left_tail_idx, right_tail_idx)

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


def moving_average(data: numpy.array, window: int) -> numpy.array:
    cumsum = numpy.cumsum(data)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    return cumsum[window - 1:] / window


def moving_dev(data: numpy.array, window: int) -> numpy.array:
    cumsum = numpy.cumsum(data)
    cumsum2 = numpy.cumsum(data ** 2)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    cumsum2[window:] = cumsum2[window:] - cumsum2[:-window]
    return ((cumsum2[window - 1:] - cumsum[window - 1:] ** 2 / window) / (window - 1)) ** 0.5


def find_ouliers(data: numpy.array,
                 center_range: Tuple[int, int] = (25, 75),
                 cut_range: float = 3.0) -> numpy.array:
    v1, v2 = numpy.percentile(data, center_range)
    return numpy.abs(data - (v1 + v2) / 2) > ((v2 - v1) / 2 * cut_range)


def find_ouliers_ts(data: numpy.array,
                    windows_size: int = 30,
                    center_range: Tuple[int, int] = (25, 75),
                    cut_range: float = 3.0) -> numpy.array:
    outliers = numpy.empty(data.shape, dtype=bool)

    if len(data) < windows_size:
        outliers[:] = False
        return outliers

    begin_idx = 0
    if len(data) < windows_size * 2:
        end_idx = (len(data) % windows_size) // 2 + windows_size
    else:
        end_idx = len(data)

    while True:
        cdata = data[begin_idx: end_idx]
        outliers[begin_idx: end_idx] = find_ouliers(cdata, center_range, cut_range)
        begin_idx = end_idx

        if end_idx == len(data):
            break

        end_idx += windows_size
        if len(data) - end_idx < windows_size:
            end_idx = len(data)

    return outliers


def hist_outliers_nd(bin_populations: numpy.array,
                     bin_centers: numpy.array,
                     center_range: Tuple[int, int] = (25, 75),
                     cut_range: float = 3.0) -> Tuple[int, int]:
    assert len(bin_populations) == len(bin_centers)
    total_count = bin_populations.sum()

    perc25 = total_count / 100.0 * center_range[0]
    perc75 = total_count / 100.0 * center_range[1]

    perc25_idx, perc75_idx = numpy.searchsorted(numpy.cumsum(bin_populations), [perc25, perc75])
    middle = (bin_centers[perc75_idx] + bin_centers[perc25_idx]) / 2
    r = (bin_centers[perc75_idx] - bin_centers[perc25_idx]) / 2

    lower_bound = middle - r * cut_range
    upper_bound = middle + r * cut_range

    lower_cut_idx, upper_cut_idx = numpy.searchsorted(bin_centers, [lower_bound, upper_bound])
    return lower_cut_idx, upper_cut_idx


def hist_outliers_perc(bin_populations: numpy.array,
                       bounds_perc: Tuple[float, float] = (0.01, 0.99)) -> Tuple[int, int]:
    assert len(bin_populations.shape) == 1
    total_count = bin_populations.sum()
    lower_perc = total_count * bounds_perc[0]
    upper_perc = total_count * bounds_perc[1]
    return numpy.searchsorted(numpy.cumsum(bin_populations), [lower_perc, upper_perc])


def ts_hist_outliers_perc(bin_populations: numpy.array,
                          window_size: int = 10,
                          bounds_perc: Tuple[float, float] = (0.01, 0.99)) -> Tuple[int, int]:
    assert len(bin_populations.shape) == 2

    points = list(range(0, len(bin_populations), window_size))
    if len(bin_populations) % window_size != 0:
        points.append(points[-1] + window_size)

    ranges = []
    for begin, end in zip(points[:-1], points[1:]):
        window_hist = bin_populations[begin:end].sum(axis=0)
        ranges.append(hist_outliers_perc(window_hist, bounds_perc=bounds_perc))

    return min(i[0] for i in ranges), max(i[1] for i in ranges)


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
