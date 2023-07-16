from collections import Counter

from brainio.stimuli import StimulusSet
from brainscore.benchmarks.moeller2017 import _Moeller2017
import numpy as np
from pytest import approx


def test_make_same_different_pairs():
    num_stimuli = 192
    stimulus_set = StimulusSet({
        'stimulus_id': [f'stim{i}' for i in range(num_stimuli)],
        'object_name': ['face'] * num_stimuli,
        'object_id': [f"r{num:03d}" for num in range(num_stimuli // 6) for _ in range(6)]
    })
    num_trials = 500
    trials = _Moeller2017._make_same_different_pairs(self=None, stimulus_set=stimulus_set, num_trials=num_trials)
    # trials
    assert len(trials) == num_trials * 2
    assert len(set(trials['trial'])) == num_trials
    assert all(num == 2 for num in Counter(trials['trial']).values())
    assert all(trials.iloc[i]['trial'] == trials.iloc[i + 1]['trial'] for i in range(0, num_trials - 1, 2))
    assert set(trials['trial_cue']) == {0, 1}
    assert all(trial['trial_cue'] == num % 2 for num, (_, trial) in enumerate(trials.iterrows()))
    # sanity check that the subsequent stimulus pairs are evenly distributed between same and different trials
    pair_same_object = [stim1['object_id'] == stim2['object_id']
                        for ((_, stim1), (_, stim2)) in zip(trials[::2].iterrows(), trials[1::2].iterrows())]
    assert np.mean(pair_same_object) == approx(0.5, abs=0.08)
    # all stimuli should be used
    assert len(set(trials['stimulus_id'])) == num_stimuli
