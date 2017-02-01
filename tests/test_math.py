import numpy
from wally.statistic import rebin_histogram


def array_eq(x: numpy.array, y: numpy.array, max_diff: float = 1E-3) -> bool:
    return numpy.abs(x - y).max() <= max_diff


def test_rebin_histo():
    curr_histo = numpy.empty((100,), dtype=int)
    curr_histo[:] = 1
    edges = numpy.arange(100)
    new_histo, new_edges = rebin_histogram(curr_histo, edges, 10)

    assert new_edges.shape == (10,)
    assert new_histo.shape == (10,)
    assert new_edges.dtype.name.startswith('float')
    assert new_histo.dtype.name.startswith('int')

    assert array_eq(new_edges, numpy.arange(10) * 9.9)
    assert new_histo.sum() == curr_histo.sum()
    assert list(new_histo) == [10] * 10

    new_histo, new_edges = rebin_histogram(curr_histo, edges, 3,
                                           left_tail_idx=20,
                                           right_tail_idx=50)

    assert new_edges.shape == (3,)
    assert new_histo.shape == (3,)
    assert array_eq(new_edges, numpy.array([20, 30, 40]))
    assert new_histo.sum() == curr_histo.sum()
    assert list(new_histo) == [30, 10, 60]
