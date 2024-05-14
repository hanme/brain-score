from brainscore_vision import metric_registry

from .difference_of_correlations import DifferenceOfCorrelations
from .difference_of_fractions import DifferenceOfFractions

metric_registry['corr_diff'] = DifferenceOfCorrelations
metric_registry['frac_diff'] = DifferenceOfFractions
