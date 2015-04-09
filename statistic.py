import math
import itertools
from numpy.polynomial.chebyshev import chebfit, chebval


def med_dev(vals):
    med = sum(vals) / len(vals)
    dev = ((sum(abs(med - i) ** 2.0 for i in vals) / len(vals)) ** 0.5)
    return med, dev


def round_deviation(med_dev):
    med, dev = med_dev

    if dev < 1E-7:
        return med_dev

    dev_div = 10.0 ** (math.floor(math.log10(dev)) - 1)
    dev = int(dev / dev_div) * dev_div
    med = int(med / dev_div) * dev_div
    return (type(med_dev[0])(med),
            type(med_dev[1])(dev))


def groupby_globally(data, key_func):
    grouped = {}
    grouped_iter = itertools.groupby(data, key_func)

    for (bs, cache_tp, act, conc), curr_data_it in grouped_iter:
        key = (bs, cache_tp, act, conc)
        grouped.setdefault(key, []).extend(curr_data_it)

    return grouped


def approximate_curve(x, y, xnew, curved_coef):
    """returns ynew - y values of some curve approximation"""
    return chebval(xnew, chebfit(x, y, curved_coef))


def approximate_line(x, y, xnew, relative_dist=False):
    """returns ynew - y values of linear approximation"""


def difference(y, ynew):
    """returns average and maximum relative and
       absolute differences between y and ynew"""


def calculate_distribution_properties(data):
    """chi, etc"""


def minimal_measurement_amount(data, max_diff, req_probability):
    """
    should returns amount of measurements to get results (avg and deviation)
    with error less, that max_diff in at least req_probability% cases
    """
