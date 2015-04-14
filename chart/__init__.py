# Patch MARKER constant

import sys
from GChartWrapper import constants
import GChartWrapper.GChart

constants.MARKERS += 'E'
print sys.modules['GChartWrapper.GChart']
sys.modules['GChartWrapper.GChart'].MARKERS += 'E'
