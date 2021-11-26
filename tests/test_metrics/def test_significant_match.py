import numpy as np
from brainio.assemblies import DataAssembly

from brainscore.metrics.significant_match import SignificantPerformanceChange


class TestSignificantPerformanceChange:
    def test_significant(self):
        aggregate_target = DataAssembly([1, .8], coords={'laser_on': [False, True]}, dims=['laser_on'])
        candidate_behaviors = DataAssembly([1, .9, .8, 1, .8, .5, .3, .6, .4, .5], coords={
            'laser_on': ('presentation', [False] * 5 + [True] * 5),
            'trial': ('presentation', [f'trial{trial + 1}' for trial in np.arange(10)]),
            'trial_num': ('presentation', [trial for trial in np.arange(10)])},
                                           dims=['presentation'])
        metric = SignificantPerformanceChange(
            condition_name='laser_on', condition_value1=False, condition_value2=True)
        score = metric(candidate_behaviors, aggregate_target)
        assert score == 1

    def test_insignificant(self):
        aggregate_target = DataAssembly([1, .8], coords={'laser_on': [False, True]}, dims=['laser_on'])
        candidate_behaviors = DataAssembly([1, .9, .8, 1, .8, .3, .9, .9, .8, .9], coords={
            'laser_on': ('presentation', [False] * 5 + [True] * 5),
            'trial': ('presentation', [f'trial{trial + 1}' for trial in np.arange(10)]),
            'trial_num': ('presentation', [trial for trial in np.arange(10)])},
                                           dims=['presentation'])
        metric = SignificantPerformanceChange(
            condition_name='laser_on', condition_value1=False, condition_value2=True)
        score = metric(candidate_behaviors, aggregate_target)
        assert score == 0
