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


def collect_detection_profile():
    """ fig2A & figS2 """
    # data extracted with https://apps.automeris.io/wpd/ on 2023-08-18, points manually selected
    data = pd.read_csv(Path(__file__).parent / 'Azadi2023_fig2A_figS2.csv')
    
    # package into xarray
    assembly = DataAssembly(data['performance_d_prime'], coords={
        'monkey': ('measurement', data['monkey']),
        'site': ('measurement', data['site']),
        'image_idx': ('measurement', data['image_idx']),
        'summary_stats': ('measurement', data['summary_stats']),
    }, dims=['measurement'])
    assembly = assembly.unstack()
    assembly['monkey_Ph'] = ('monkey', [monkey == 'Ph' for monkey in assembly['monkey']])
    assembly['monkey_Sp'] = ('monkey', [monkey == 'Sp' for monkey in assembly['monkey']])
    assembly = assembly.stack(condition=['monkey'])
    assembly = DataAssembly(assembly)
    return assembly


def collect_corr_between_detection_profiles():
    """ Azadi2023_fig2B_fig_S3A """
    # data extracted with https://apps.automeris.io/wpd/ on 2023-08-18, points manually selected
    data = pd.read_csv(Path(__file__).parent / 'Azadi2023_fig2B_fig_S3A.csv')
    print(data)

    # package into xarray
    assembly = DataAssembly(data['corr_coeff'], coords={
        'monkey': ('measurement', data['monkey']),
        'site': ('measurement', data['site']),
        'violin_part': ('measurement', data['violin_part']),
        'summary_stats': ('measurement', data['summary_stats']),
    }, dims=['measurement'])
    assembly = assembly.unstack()
    assembly['monkey_Ph_corr_median'] = assembly.where((assembly.monkey == 'Ph') & \
                                                  (assembly.violin_part == 'center')).corr_coeff
    assembly['monkey_Sp_corr_median'] = assembly.where((assembly.monkey == 'Sp') & \
                                                  (assembly.violin_part == 'center')).corr_coeff
    assembly = assembly.stack(condition=['monkey'])
    assembly = DataAssembly(assembly)
    return assembly


def collect_psychometric_functions_illumination_power():
    """ Azadi2023_fig2C_figS3BCD """
    # data extracted with https://apps.automeris.io/wpd/ on 2023-08-18, points manually selected
    data = pd.read_csv(Path(__file__).parent / 'Azadi2023_fig2C_figS3BCD.csv')
    print(data)

    # package into xarray
    assembly = DataAssembly(data['performance_d_prime'], coords={
        'monkey': ('measurement', data['monkey']),
        'site': ('measurement', data['site']),
        'image_id_color': ('measurement', data['image_id_color']),
        'illumination_power_mW': ('measurement', data['illumination_power_mW']),
        'summary_stats': ('measurement', data['summary_stats']),
    }, dims=['measurement'])
    assembly = assembly.unstack()
    assembly['monkey_Ph'] = ('monkey', [monkey == 'Ph' for monkey in assembly['monkey']])
    assembly['monkey_Sp'] = ('monkey', [monkey == 'Sp' for monkey in assembly['monkey']])
    assembly = assembly.stack(condition=['monkey'])
    assembly = DataAssembly(assembly)
    return assembly


def collect_psychometric_functions_image_visibility():
    """ Azadi2023_fig3_figS3E """
    # data extracted with https://apps.automeris.io/wpd/ on 2023-08-18, points manually selected
    data = pd.read_csv(Path(__file__).parent / 'Azadi2023_fig3_figS3E.csv')
    print(data)

    # package into xarray
    assembly = DataAssembly(data['performance_d_prime'], coords={
        'monkey': ('measurement', data['monkey']),
        'image_visibility_condition': ('measurement', data['image_visibility_condition']),
    }, dims=['measurement'])
    assembly = assembly.unstack()
    assembly['monkey_Ph'] = ('monkey', [monkey == 'Ph' for monkey in assembly['monkey']])
    assembly['monkey_Sp'] = ('monkey', [monkey == 'Sp' for monkey in assembly['monkey']])
    assembly = assembly.stack(condition=['monkey'])
    assembly = DataAssembly(assembly)
    return assembly
