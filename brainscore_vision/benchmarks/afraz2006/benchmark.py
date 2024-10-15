import logging

import numpy as np
import scipy.optimize
from numpy.random import RandomState
from scipy.optimize import fsolve
from tqdm import tqdm
from xarray import DataArray

import brainscore_vision
from brainio.assemblies import merge_data_arrays, walk_coords, DataAssembly
from brainscore_core import Score
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_vision.data.afraz2006 import BIBTEX
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.utils import fullname

STIMULATION_PARAMETERS = {
    # "Microstimulation consisted of bipolar current pulses of 50mA delivered at 200 Hz (refs 19, 20).
    # The stimulation pulses were biphasic, with the cathodal pulse leading. Each pulse was 0.2 ms in
    # duration with 0.1 ms between the cathodal and anodal phase. [...] Stimulating pulses were
    # delivered for 50 ms in one of three time periods following onset of the visual stimulus:
    # 0–50 ms, 50–100 ms or 100–150 ms."
    # We here focus on the 100-150ms condition.
    'current_pulse_mA': 50,
    'pulse_type': 'biphasic',
    'pulse_rate_Hz': 200,
    'pulse_duration_ms': 0.2,
    'pulse_interval_ms': 0.1,
    'stimulation_onset_ms': 100,
    'stimulation_duration_ms': 50,
}
STIMULATION_PARAMETERS = frozenset(STIMULATION_PARAMETERS.items())  # make immutable so that params cannot be changed


def Afraz2006FaceDependentShiftSignificant():
    metric = brainscore_vision.load_metric('corr_sig', x_coord='face_selectivity', ignore_nans=True)
    return _Afraz2006(metric_identifier='shift_significant', metric=metric)


def Afraz2006FaceDependentShift():
    metric = brainscore_vision.load_metric('diff_corr', correlation_variable='face_selectivity')
    return _Afraz2006(metric_identifier='face_dependent_shift', metric=metric)


class _Afraz2006(BenchmarkBase):
    def __init__(self, metric_identifier, metric):
        self._logger = logging.getLogger(fullname(self))
        self._assembly, self._fitting_stimuli = self._load_assembly()
        self._metric = metric
        super(_Afraz2006, self).__init__(
            identifier='esteky.Afraz2006-' + metric_identifier,
            ceiling=Score(1),
            version=1, parent='IT',
            bibtex=BIBTEX)

    def _load_assembly(self):
        assembly = brainscore_vision.load_dataset('Afraz2006')
        # stimuli
        train_stimuli = brainscore_vision.load_stimulus_set('Afraz2006.train')
        test_stimuli = brainscore_vision.load_stimulus_set('Afraz2006.test')
        assembly.attrs['stimulus_set'] = test_stimuli
        return assembly, train_stimuli

    def __call__(self, candidate: BrainModel):
        # record to later determine face-selectivity
        candidate.start_recording('IT', time_bins=[(50, 100)], recording_type=BrainModel.RecordingType.exact)
        recordings = candidate.look_at(self._assembly.stimulus_set)

        # "We trained two adult macaque monkeys to perform a face/non-face categorization task
        # upon viewing single images from one or the other category that were systematically degraded
        # by varying amounts of visual signal."
        # train on face/non-face categorization task
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli=self._fitting_stimuli)
        nonstimulated_behavior = candidate.look_at(self._assembly.stimulus_set)

        # "Altogether, we assessed stimulus selectivity at 348 recording sites in 86 electrode penetrations in
        # two monkeys (46 and 40 in monkeys FR and KH, respectively).
        # We conducted microstimulation experiments at 31 face-selective sites and 55 non-selective sites,
        # while the monkey performed the object categorization task.
        # Selectivity for faces was defined as having a d' value > 1."

        # "Recordings were made on an evenly spaced grid, with 1-mm intervals between penetrations over a wide region of
        # the lower bank of STS and TEa cortices (left hemisphere 14 to 21mm anterior to interauricular line in FR,
        # and right hemisphere 14 to 20mm anterior to interauricular line in KH). The recording positions were
        # determined stereotaxically by referring to magnetic resonance images acquired before the surgery.
        # Multiunit neural responses were recorded through tungsten microelectrodes (0.4–1.0MΩ). Neural selectivity of
        # neighbouring sites within ±500mm from the stimulated site along each recording track was determined as the
        # electrode was advanced. The recorded positions were separated by at least 150mm (mean, 296mm). After
        # determining the neighbourhood selectivity, the electrode tip was positioned in the middle of the recorded
        # area and remained there through the rest of the experiment. The neural selectivity in this site was verified
        # before starting the categorization task."

        # We here randomly sub-select the recordings to match the number of stimulation sites in the experiment, based
        # on the assumption that we can compare trend effects even with a random sample.
        subselect = 86
        random_state = RandomState(1)
        neuroid_ids = sorted(recordings['neuroid_id'].values)  # sort to make sure random choice is reproducible
        subselected_neuroid_ids = random_state.choice(neuroid_ids, size=subselect, replace=False)
        recordings = recordings[{'neuroid': [neuroid_id in subselected_neuroid_ids
                                             for neuroid_id in recordings['neuroid_id'].values]}]
        stimulation_locations = np.stack((recordings['recording_x'], recordings['recording_y'])).T.tolist()
        candidate_behaviors = []
        for site, location in enumerate(tqdm(stimulation_locations, desc='stimulation locations')):
            candidate.perturb(perturbation=None, target='IT')  # reset
            location = np.round(location, decimals=2)
            self._logger.debug(f"Stimulating at {location}")
            candidate.perturb(perturbation=BrainModel.Perturbation.microstimulation,
                              target='IT', perturbation_parameters={
                    **dict(STIMULATION_PARAMETERS),
                    **{'location': location},
                })
            behavior = candidate.look_at(self._assembly.stimulus_set)
            behavior = behavior.expand_dims('site')
            behavior['site_iteration'] = 'site', [site]
            behavior['site_x'] = 'site', [location[0]]
            behavior['site_y'] = 'site', [location[1]]
            behavior = type(behavior)(behavior)  # make sure site is indexed
            candidate_behaviors.append(behavior)
        candidate_behaviors = merge_data_arrays(candidate_behaviors)
        psychometric_shifts = self.characterize_psychometric_shifts(nonstimulated_behavior, candidate_behaviors)

        # face selectivities
        face_selectivities = determine_face_selectivity(recordings)
        self.attach_face_selectivities(psychometric_shifts, face_selectivities[:subselect])

        # compare
        psychometric_shifts = psychometric_shifts[{'site': [  # ignore nan values
            not np.isnan(face_selectivity) for face_selectivity in psychometric_shifts['face_selectivity'].values]}]
        score = self._metric(psychometric_shifts, self._assembly)
        score.attrs['raw'] = psychometric_shifts
        return score

    def characterize_psychometric_shifts(self, nonstimulated_behavior, behaviors):
        nonstimulated_curve = self.grouped_face_responses(nonstimulated_behavior)
        nonstimulated_logistic = self.fit_logistic(x=nonstimulated_curve['label_signal_level'],
                                                   y=nonstimulated_curve.values)
        nonstimulated_signal_midpoint = self.logistic_midpoint(nonstimulated_logistic)

        psychometric_shifts = []
        for site in behaviors['site'].values:
            # index instead of `.sel` to preserve all site coords
            behavior = behaviors[{'site': [s == site for s in behaviors['site'].values]}]
            site_coords = {coord: (dims, values) for coord, dims, values in walk_coords(behavior['site'])}
            behavior = behavior.squeeze('site')
            psychometric_curve = self.grouped_face_responses(behavior)
            try:
                site_logistic = self.fit_logistic(x=psychometric_curve['label_signal_level'],
                                                  y=psychometric_curve.values)
                site_midpoint = self.logistic_midpoint(site_logistic)
                psychometric_shift = nonstimulated_signal_midpoint - site_midpoint
            except (AssertionError, RuntimeError):  # unable to fit function / find midpoint
                psychometric_shift = 0
            psychometric_shift = DataAssembly([psychometric_shift], coords=site_coords, dims=['site'])
            psychometric_shifts.append(psychometric_shift)
        psychometric_shifts = merge_data_arrays(psychometric_shifts)
        return psychometric_shifts

    def attach_face_selectivities(self, psychometric_shifts, face_selectivities):
        assert len(psychometric_shifts) == len(face_selectivities)
        # assume same ordering
        psychometric_shifts['face_selectivity'] = 'site', face_selectivities.values

    def grouped_face_responses(self, behavior):
        np.testing.assert_array_equal(behavior['choice'], ['face', 'nonface'])
        behavior['choose_face'] = 'presentation', behavior.argmax('choice').values
        face_responses = DataAssembly(behavior.argmax('choice'), coords={
            coord: (dims, values) for coord, dims, values in walk_coords(behavior['presentation'])},
                                      dims=['presentation'])
        face_responses = 1 - face_responses  # invert so that nonface (0) to face (1)
        grouped_face_responses = face_responses.groupby('label_signal_level').mean()
        return grouped_face_responses

    def fit_logistic(self, x, y):
        params, pcov = scipy.optimize.curve_fit(logistic, x, y)
        return params

    def logistic_midpoint(self, logistic_params, midpoint=0.5, initial_guess=0):
        func = lambda x: logistic(x, *logistic_params) - midpoint
        solution = fsolve(func, initial_guess)[0]
        fit_midpoint = logistic(solution, *logistic_params)
        assert np.isclose(fit_midpoint, midpoint), f"Unable to find midpoint: " \
                                                   f"{fit_midpoint} (with parameters {logistic_params}) " \
                                                   f"is different from target midpoint {midpoint}"
        return solution


def determine_face_selectivity(recordings):
    assert (recordings >= 0).all()
    # A d' value of zero indicates indistinguishable responses to faces and non-faces.
    # Increasingly positive d' values indicate progressively better selectivity for faces.
    # Selectivity for faces was defined as having a d' value > 1.
    result = []
    for neuroid_id in tqdm(recordings['neuroid_id'].values, desc='neuron face dprime'):
        neuron = recordings.sel(neuroid_id=neuroid_id)
        neuron = neuron.squeeze()
        face_mean, face_variance = mean_var(neuron.sel(image_label='face'))
        nonface_mean, nonface_variance = mean_var(neuron.sel(image_label='nonface'))
        # face selectivity based on "more positive" firing
        dprime = (face_mean - nonface_mean) / np.sqrt((face_variance + nonface_variance) / 2)
        result.append(dprime)
    result = DataArray(result, coords={'neuroid_id': recordings['neuroid_id'].values}, dims=['neuroid_id'])
    return result


def mean_var(neuron):
    mean, var = np.mean(neuron.values), np.var(neuron.values)
    return mean, var


def logistic(x, a, b):
    return 1 / (1 + np.exp(-(a + b * x)))
