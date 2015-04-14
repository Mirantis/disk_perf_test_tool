import sys

from GChartWrapper import constants

# Patch MARKER constant
constants.MARKERS += 'E'

sys.modules['GChartWrapper.GChart'].MARKERS += 'E'
