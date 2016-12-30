from typing import List


expected_lat_bins = 1216


#----------------------------  FIO HIST LOG PARSE CODE -----------------------------------------------------------------

# Copy-paste from fio/tools/hist/fiologparser_hist.py.
# Because that's impossible to understand or improve,
# you can only copy such a pearl.

def _plat_idx_to_val(idx: int , edge: float = 0.5, FIO_IO_U_PLAT_BITS: int = 6, FIO_IO_U_PLAT_VAL: int = 64) -> float:
    """ Taken from fio's stat.c for calculating the latency value of a bin
        from that bin's index.

            idx  : the value of the index into the histogram bins
            edge : fractional value in the range [0,1]** indicating how far into
            the bin we wish to compute the latency value of.

        ** edge = 0.0 and 1.0 computes the lower and upper latency bounds
           respectively of the given bin index. """

    # MSB <= (FIO_IO_U_PLAT_BITS-1), cannot be rounded off. Use
    # all bits of the sample as index
    if (idx < (FIO_IO_U_PLAT_VAL << 1)):
        return idx

    # Find the group and compute the minimum value of that group
    error_bits = (idx >> FIO_IO_U_PLAT_BITS) - 1
    base = 1 << (error_bits + FIO_IO_U_PLAT_BITS)

    # Find its bucket number of the group
    k = idx % FIO_IO_U_PLAT_VAL

    # Return the mean (if edge=0.5) of the range of the bucket
    return base + ((k + edge) * (1 << error_bits))


def plat_idx_to_val_coarse(idx: int, coarseness: int, edge: float = 0.5) -> float:
    """ Converts the given *coarse* index into a non-coarse index as used by fio
        in stat.h:plat_idx_to_val(), subsequently computing the appropriate
        latency value for that bin.
        """

    # Multiply the index by the power of 2 coarseness to get the bin
    # bin index with a max of 1536 bins (FIO_IO_U_PLAT_GROUP_NR = 24 in stat.h)
    stride = 1 << coarseness
    idx = idx * stride
    lower = _plat_idx_to_val(idx, edge=0.0)
    upper = _plat_idx_to_val(idx + stride, edge=1.0)
    return lower + (upper - lower) * edge


def get_lat_vals(columns: int = expected_lat_bins, coarseness: int = 0) -> List[float]:
    return [plat_idx_to_val_coarse(val, coarseness) for val in range(columns)]

