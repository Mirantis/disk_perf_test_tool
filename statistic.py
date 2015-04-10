import math
import itertools
from numpy import array, linalg
from numpy.polynomial.chebyshev import chebfit, chebval
from scipy.optimize import leastsq


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
    """ x, y - test data, xnew - dots, where we want find approximation
        if not relative_dist distance = y - newy
        returns ynew - y values of linear approximation"""
    # convert to numpy.array (don't work without it)
    ox = array(x)
    oy = array(y)
    # define function for initial value
    def get_init(x, y):
        """ create initial value for better work of leastsq """
        A = [[x[i], 1.0] for i in range(0, 2)]
        b = [y[i] for i in range(0, 2)]
        return tuple(linalg.solve(A, b))
    # set approximation function
    funcLine = lambda tpl, x: tpl[0] * x + tpl[1]
    # choose distance mode
    if relative_dist:
        ErrorFunc = lambda tpl, x, y: 1.0 - y/funcLine(tpl, x)
    else:
        ErrorFunc = lambda tpl, x, y: y - funcLine(tpl, x)
    # choose initial value
    tplInitial = get_init(ox, oy)
    # find line
    tplFinal, success = leastsq(ErrorFunc, tplInitial[:], args=(ox, oy))
    # if error
    if success not in range(1, 5):
        raise ValueError("No line for this dots")
    # return new dots
    return funcLine(tplFinal, array(xnew))


def difference(y, ynew):
    """returns average and maximum relative and
       absolute differences between y and ynew
       result may contain None values for y = 0
       return value - tuple:
       [(abs dif, rel dif) * len(y)],
       (abs average, abs max),
       (rel average, rel max)"""
    da_sum = 0.0
    dr_sum = 0.0
    da_max = 0.0
    dr_max = 0.0
    dlist = []
    for y1, y2 in zip(y, ynew):
        # absolute
        da = y1 - y2
        da_sum += abs(da)
        if abs(da) > da_max:
            da_max = da
        # relative
        if y1 != 0:
            dr = abs(da / y1)
            dr_sum += dr
            if dr > dr_max:
                dr_max = dr
        else:
            dr = None
        # add to list
        dlist.append((da, dr))
    da_sum /= len(y)
    dr_sum /= len(y)
    return dlist, (da_sum, da_max), (dr_sum, dr_max)



def calculate_distribution_properties(data):
    """chi, etc"""


def minimal_measurement_amount(data, max_diff, req_probability):
    """
    should returns amount of measurements to get results (avg and deviation)
    with error less, that max_diff in at least req_probability% cases
    """
