from brainscore_vision import metric_registry

from .significant_match import (
    SignificantCorrelation,
    SignificantPerformanceChange, NoSignificantPerformanceChange)

metric_registry['corr_sig'] = SignificantCorrelation
metric_registry['perf_diff'] = SignificantPerformanceChange
metric_registry['perf_nodiff'] = NoSignificantPerformanceChange
