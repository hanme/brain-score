import sys
paths = [
    "/home/mehrer/projects/perturbations/perturbation_tests/brain-score-perturbation",
    "/home/mehrer/projects/perturbations/perturbation_tests/brainio-main",
    "/home/mehrer/projects/perturbations/perturbation_tests/candidate_models-perturbation",
    "/home/mehrer/projects/perturbations/perturbation_tests/direct-causality-master",
    "/home/mehrer/projects/perturbations/perturbation_tests/model-tools-private-perturbation",
    "/home/mehrer/projects/perturbations/perturbation_tests/topotorch-master",
    "/home/mehrer/projects/perturbations/perturbation_tests/result_caching-master",
    "/home/mehrer/projects/perturbations/TDANN/TDANN",
    "/home/mehrer/projects/perturbations/TDANN/vissl",
    "/home/mehrer/projects/perturbations/TDANN/vonenet"
]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)
print(sys.path)

import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from brainio.assemblies import DataAssembly
from brainio.stimuli import StimulusSet


def collect_stimuli():
    stimulus_set = pd.read_pickle(Path(__file__).parent / 'stimuli/metadata.pkl')
    stimulus_set = stimulus_set.rename(columns={'stim_idx': 'idx'})
    assert all(Path(stimulus_set.get_stimulus(stimulus_id)).is_file() for stimulus_id in stimulus_set['stimulus_id'])
    assert set(stimulus_set['monkey']) == {'Ph', 'Sp'}
    assert set(stimulus_set['train_test']) == {'train', 'test'}
    """ max 22 train stimuli, max 40 test stimuli"""
    assert all(int(idx) <= (22 if label == 'train' else 40) for label, idx in \
               zip(stimulus_set['train_test'], stimulus_set['idx']))
    return stimulus_set


#def collect_assembly():
#    XXX


def collect_stimulation_report_rate_training():
    """ fig 1C (only data for monkey Ph given) """
    # data extracted with https://apps.automeris.io/wpd/ on 2023-08-18, points manually selected
    data = pd.read_csv(Path(__file__).parent / 'Azadi2023_fig1C.csv')

     # package into xarray
    assembly = DataAssembly(data['stimulation_report_rate'], coords={
        'monkey': ('measurement', data['monkey']),
        'session_number': ('measurement', data['session_number']),
        'condition_description': ('measurement', data['condition']),
        'summary_stats': ('measurement', data['summary_stats']),
    }, dims=['measurement'])
    assembly = assembly.unstack()
    assembly['laser_on_virus_site'] = ('condition_description', [condition == 'stimulation'
                                                      for condition in assembly['condition_description']])
    assembly['laser_on_catch'] = ('condition_description', [condition == 'catch'
                                                      for condition in assembly['condition_description']])
    assembly['nonstimulation'] = ('condition_description', [condition == 'nonstimulation'
                                                      for condition in assembly['condition_description']])
    assembly = assembly.stack(condition=['condition_description'])
    assembly = DataAssembly(assembly)
    return assembly

