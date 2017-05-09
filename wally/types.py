from typing import TypeVar, List, Union

import numpy


TNumber = TypeVar('TNumber', int, float)
Number = Union[int, float]
NumVector = Union[numpy.ndarray, List[int], List[float]]
