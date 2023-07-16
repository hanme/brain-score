from typing import Tuple

import numpy as np
from numpy.random import RandomState
from tqdm import tqdm

from brainio.assemblies import merge_data_arrays, DataAssembly
from brainio.stimuli import StimulusSet
from brainscore.benchmarks import BenchmarkBase
from brainscore.metrics import Metric
from brainscore.metrics.accuracy import Accuracy
from brainscore.metrics.significant_match import SignificantPerformanceChange
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad
from data_packaging.moeller2017 import collect_target_assembly

"""
Benchmarks from the Moeller et al. 2017 paper (https://www.nature.com/articles/nn.4527).
Only AM and outside-AM stimulation experiments are considered for now.
"""

BIBTEX = '''@article{article,
            author = {Moeller, Sebastian and Crapse, Trinity and Chang, Le and Tsao, Doris},
            year = {2017},
            month = {03},
            pages = {},
            title = {The effect of face patch microstimulation on perception of faces and objects},
            volume = {20},
            journal = {Nature Neuroscience},
            doi = {10.1038/nn.4527}
            }'''

STIMULATION_PARAMETERS = {
    'type': [None, BrainModel.Perturbation.microstimulation, BrainModel.Perturbation.microstimulation],
    'current_pulse_mA': [0, 300],
    'pulse_rate_Hz': 150,
    'pulse_duration_ms': 0.2,
    'pulse_interval_ms': 0.1,
    'stimulation_duration_ms': 200
}

DPRIME_THRESHOLD_SELECTIVITY = .66  # equivalent to .5 in monkey, see Lee et al. 2020
DPRIME_THRESHOLD_FACE_PATCH = .85  # equivalent to .65 in monkey, see Lee et al. 2020


class _Moeller2017(BenchmarkBase):
    def __init__(self, stimulus_class: str, perturbation_location: str, identifier: str,
                 metric: Metric, performance_measure):
        """
        Perform a same vs different identity judgement task on the given dataset
            with and without micro-stimulation in the specified location.
            Compute the behavioral performance on the given performance measure and
            compare it to the equivalent data retrieved from primate experiments.

        For each decision, only identities from the same object category are being compared.
        Within each dataset, the number of instances per category is equalized. As is the number of different
        representations (faces: expressions, object: viewing angles) per instance.

        :param stimulus_class: one of: ['Faces', 'Objects', 'Eliciting_Face_Response', 'Abstract_Faces', 'Abstract_Houses']
        :param perturbation_location: one of: ['within_facepatch', 'outside_facepatch']
        :param identifier: benchmark id
        :param metric: in: performances along multiple dimensions of 2 instances | out: Score object, evaluating similarity
        :param performance_measure: taking behavioral data, returns performance w.r.t. each dimension
        """
        super().__init__(
            identifier=identifier,
            ceiling_func=lambda: None,
            version=1, parent='IT',
            bibtex=BIBTEX)

        self._metric = metric
        self._performance_measure = performance_measure
        self._perturbation_location = perturbation_location
        # LazyLoad because `self._perturbation_coordinates` is not set until later
        self._perturbations = LazyLoad(self._set_up_perturbations)
        self._stimulus_class = stimulus_class

        self._target_assembly = collect_target_assembly(stimulus_class=stimulus_class,
                                                        perturbation_location=perturbation_location)
        self._test_set = self._make_same_different_pairs(
            self._target_assembly.stimulus_set, include_label=True)
        self._stimulus_set_face_patch = self._target_assembly.stimulus_set_face_patch
        self._training_trials = self._make_same_different_pairs(
            self._target_assembly.training_stimuli, include_label=True)

    def _make_same_different_pairs(self, stimulus_set: StimulusSet, num_trials: int = 500,
                                   include_label: bool = False) -> StimulusSet:
        # "In the main experiment (Experiment 1; 32 faces, 6 exemplars), for each trial we randomly selected one image
        # from all 192 different images as the first cue. The second cue was drawn either from all images showing a
        # different identity as the first cue (one of 186 images) or from all images showing the same identity
        # (one of 6 images)."
        # Not obvious how many stimuli were presented in an experiment though. We'll conservatively assume 500 although
        # the actual number might be closer to ~100 ("Single-unit responses [...] in response to 80 stimuli",
        # "population decoding of individual identity for a set of 128 stimuli")
        rng = RandomState(0)
        stimulus_set = stimulus_set.sort_values(by='stimulus_id')  # sort to make sure choice is reproducible
        trials, labels = [], []
        while len(trials) < num_trials:
            first_cue = stimulus_set.iloc[rng.choice(len(stimulus_set))]
            same_task = rng.choice([True, False])
            if same_task:  # choose from images of the same identity, but not the exact same one
                second_cue_choices = stimulus_set[(stimulus_set['object_id'] == first_cue['object_id']) &
                                                  (stimulus_set['stimulus_id'] != first_cue['stimulus_id'])]
            else:  # choose from images of different identities
                second_cue_choices = stimulus_set[stimulus_set['object_id'] != first_cue['object_id']]
            second_cue = second_cue_choices.iloc[rng.choice(len(second_cue_choices))]
            trials.append((first_cue, second_cue))
            labels.append('same' if same_task else 'diff')

        pair_stimuli = [stimulus for trial_pair in trials for stimulus in trial_pair]
        pair_labels = [stimulus_label for label in labels for stimulus_label in [label] * 2]
        trial_stimulus_set = StimulusSet(pair_stimuli)
        trial_stimulus_set['trial'] = [num for num in range(len(trial_stimulus_set) // 2) for _ in range(2)]
        trial_stimulus_set['trial_cue'] = [num % 2 for num in range(len(trial_stimulus_set))]
        if include_label:
            trial_stimulus_set['label'] = pair_labels
        trial_stimulus_set.stimulus_paths = stimulus_set.stimulus_paths
        trial_stimulus_set.identifier = stimulus_set.identifier + '-trials'
        return trial_stimulus_set

    def __call__(self, candidate: BrainModel):
        candidate.perturb(perturbation=None, target='IT')  # reset
        self._compute_perturbation_coordinates(candidate)
        # The Moeller et al. 2017 paper used a rather elaborate training paradigm
        # (passive fixation -> "choice" task with only the correct target -> 1 tomato vs 5 grapes classification
        # -> 5 human faces same-different -> 32 faces -> non-face objects).
        # We here simplify to just presenting
        candidate.start_task(BrainModel.Task.same_different, fitting_stimuli=self._training_trials)

        behaviors = []
        for perturbation in self._perturbations:
            candidate.perturb(perturbation=None, target='IT')  # reset
            candidate.perturb(perturbation=perturbation['type'], target='IT',
                              perturbation_parameters=perturbation['perturbation_parameters'])
            behavior = candidate.look_at(self._test_set)

            current_pulse_mA = perturbation['perturbation_parameters']['current_pulse_mA']
            behavior = behavior.expand_dims('stimulation')
            behavior['current_pulse_mA'] = 'stimulation', [current_pulse_mA]
            behavior['stimulated'] = 'stimulation', [current_pulse_mA > 0]
            behavior = type(behavior)(behavior)  # make sure current_pulse_mA is indexed
            behaviors.append(behavior)
        candidate.perturb(perturbation=None, target='IT')  # reset
        behaviors = merge_data_arrays(behaviors)
        # flatten
        behaviors = behaviors.reset_index('stimulation').reset_index('presentation') \
            .stack(condition=['stimulation', 'presentation'])
        behaviors = type(behaviors)(behaviors)

        score = self._metric(behaviors, self._target_assembly)
        return score

    def _set_up_perturbations(self):
        """
        Create a list of dictionaries, each containing the parameters for one perturbation
        :return: list of dict, each containing parameters for one perturbation
        """
        perturbation_list = []
        for stimulation, current in zip(STIMULATION_PARAMETERS['type'], STIMULATION_PARAMETERS['current_pulse_mA']):
            perturbation_dict = {'type': stimulation,
                                 'perturbation_parameters': {
                                     **{
                                         'current_pulse_mA': current,
                                         'location': self._perturbation_coordinates
                                     }, **{key: value for key, value in STIMULATION_PARAMETERS.items()
                                           if key not in ['type', 'current_pulse_mA', 'location']
                                           }
                                 }}

            perturbation_list.append(perturbation_dict)
        return perturbation_list

    def _compute_perturbation_coordinates(self, candidate: BrainModel):
        """
        Save stimulation coordinates (x,y) to self._perturbation_coordinates
        :param candidate: BrainModel
        """
        candidate.start_recording('IT', time_bins=[(50, 100)], recording_type=BrainModel.RecordingType.fMRI)
        recordings = candidate.look_at(self._stimulus_set_face_patch)  # vs _training_assembly

        # compute face selectivity
        face_selectivities_voxel = self._determine_face_selectivity(recordings)

        # Determine location
        if self._perturbation_location == 'within_facepatch':  # "Electrical stimulation of face patch AM"
            x, y = self._get_purity_center(face_selectivities_voxel)
        elif self._perturbation_location == 'outside_facepatch':
            x, y = self._sample_outside_face_patch(face_selectivities_voxel)
        else:
            raise KeyError

        self._perturbation_coordinates = (x, y)

    @staticmethod
    def _get_purity_center(selectivity_assembly: DataAssembly, radius: int = 1) -> Tuple[int, int]:
        """
        Adapted from Lee et al. 2020.
        Computes the voxel of the selectivity map with the highest purity
        :param: selectivity_assembly:
            dims: 'neuroid_id'
                    coords:
                        recording_x: voxel coordinate
                        recording_y: voxel coordinate
                  'category_name'
        :param: radius (scalar): radius in mm of the circle in which to consider units
        :return: location of highest purity
        """

        def get_purity(center_x, center_y):
            """
            Evaluates purity at a given center position, radius, and corresponding selectivity values
            """
            passing_indices = np.where(np.sqrt(np.square(x - center_x) + np.square(y - center_y)) < radius)[0]
            return 100. * np.sum(selectivity_assembly.values[passing_indices]) / passing_indices.shape[0]

        x, y = selectivity_assembly.recording_x.values, selectivity_assembly.recording_y.values
        purity = np.array(list(map(get_purity, x, y)))
        highest_purity_idx = np.argmax(purity)

        center_x, center_y = x[highest_purity_idx], y[highest_purity_idx]
        return center_x, center_y

    @staticmethod
    def _determine_face_selectivity(recordings: DataAssembly):
        """
        Determines face selectivity of each neuroid
        :param recordings: DataAssembly
        :return: DataAssembly, same as recordings where activations have been replaced with dprime values
        """

        def mean_var(neuron):
            mean, var = np.mean(neuron.values), np.var(neuron.values)
            return mean, var

        assert (recordings >= 0).all(), 'selectivities must be positive'

        selectivities = []
        for voxel_id in tqdm(recordings['voxel_id'].values, desc='neuron face dprime'):
            voxel = recordings.sel(voxel_id=voxel_id)
            voxel = voxel.squeeze()
            face_mean, face_variance = mean_var(voxel.sel(category_name='Faces'))  # image_label='face'))
            nonface_mean, nonface_variance = mean_var(
                voxel.where(voxel.category_name != 'Faces', drop=True))  # sel(image_label='nonface'))
            dprime = (face_mean - nonface_mean) / np.sqrt((face_variance + nonface_variance) / 2)
            selectivities.append(dprime)

        selectivity_array = DataAssembly(
            data=selectivities,
            dims=['voxel_id'],
            coords={'voxel_id': recordings['voxel_id'].values,
                    'recording_x': ('voxel_id', recordings['recording_x'].values),
                    'recording_y': ('voxel_id', recordings['recording_y'].values)})
        return selectivity_array

    def _sample_outside_face_patch(self, selectivity_assembly: DataAssembly, radius=2):
        """
        Sample one voxel outside of face patch
        1. make a list of voxels where neither the voxel nor its close neighbors are in a face patch
        2. randomly sample from list
        :param selectivity_assembly:
        :param radius: determining the neighborhood size in mm of each voxel that cannot be selective
        :return: x, y location of voxel outside face_patch
        """
        not_selective_voxels = selectivity_assembly[selectivity_assembly.values < DPRIME_THRESHOLD_SELECTIVITY]
        voxels = []
        for voxel in not_selective_voxels:
            recording_x, recording_y = voxel.voxel_id.item()[1:]  # because iteration removes coords somehow
            inside_radius = np.where(np.sqrt(
                np.square(not_selective_voxels['recording_x'].values - recording_x) +
                np.square(not_selective_voxels['recording_y'].values - recording_y)) < radius)[0]
            if np.all(not_selective_voxels[inside_radius].values < DPRIME_THRESHOLD_FACE_PATCH):
                voxels.append(voxel)  # safety if there is tissue where this does not hold

        rng = np.random.default_rng(seed=1)
        random_idx = rng.integers(0, len(voxels))  # choice removes meta data i.e. location
        x, y = voxels[random_idx].voxel_id.item()[1:]
        return x, y

    @staticmethod
    def _merge_behaviors(arrays):
        """
        Hack because from brainio.assemblies import merge_data_arrays gives
        an index duplicate error
        :param arrays:
        :return:
        """
        if len(arrays) > 1:
            data = np.concatenate([e.data for e in arrays])
            conditions = np.concatenate([e.condition.data for e in arrays])
            truths = np.concatenate([e.truth.data for e in arrays])
            object_names = np.concatenate([e.object_name.data for e in arrays])
            return DataAssembly(data=data, dims='condition',
                                coords={'condition': conditions, 'truth': ('condition', truths),
                                        'object_name': ('condition', object_names)})
        else:
            return DataAssembly(arrays[0])


def stimulation_same_different_significant_change(candidate_behaviors, aggregate_target):
    """
    Tests that the candidate behaviors changed in the same direction as the data after stimulation,
    separately for same and different tasks.
    :param candidate_behaviors: Per-trial behaviors (_not_ aggregate performance measures).
    :param aggregate_target: Performance numbers for the experimental observations, i.e. _not_ per-trial data.
        This will be used to determine the expected direction from stimulation (increase/decrease)
        for each of the two tasks.
    :return: A :class:`~brainscore.metrics.Score` of 1 if the candidate_behaviors significantly change in the same
        direction as the aggregate_target, for each of the same and different tasks; 0 otherwise
    """
    change_metric = SignificantPerformanceChange(condition_name='current_pulse_mA',
                                                 condition_value1=0, condition_value2=300, trial_dimension='condition')
    score_same = change_metric(candidate_behaviors.sel(task='same_id'), aggregate_target.sel(task='same_id'))
    score_different = change_metric(candidate_behaviors.sel(task='different_id'), aggregate_target.sel(task='same_id'))
    joint_score = score_same & score_different
    joint_score.attrs['score_same'] = score_same
    joint_score.attrs['score_different'] = score_different
    return joint_score


def Moeller2017Experiment1SameDecreaseDifferentIncrease():
    """
    Stimulate face patch during face identification
    32 identities; 6 expressions each
    """

    return _Moeller2017(stimulus_class='Faces',
                        perturbation_location='within_facepatch',
                        identifier='dicarlo.Moeller2017-Experiment_1',
                        metric=stimulation_same_different_significant_change,
                        performance_measure=Accuracy())


"""
TODO  very unclean
i: Stimulate outside of the face patch during face identification
ii: Stimulate face patch during object identification
28 Objects; 3 viewing angles each
"""


class Moeller2017Experiment2(BenchmarkBase):
    def __init__(self):
        metric = None  # PerformanceSimilarity()
        super().__init__(
            identifier='dicarlo.Moeller2017-Experiment_2', ceiling_func=None, version=1, parent='IT', bibtex=BIBTEX)
        self.benchmark1_outside_faces = _Moeller2017(stimulus_class='Faces', perturbation_location='outside_facepatch',
                                                     identifier='dicarlo.Moeller2017-Experiment_2i',
                                                     metric=metric, performance_measure=Accuracy())
        self.benchmark2_objects = _Moeller2017(stimulus_class='Objects', perturbation_location='within_facepatch',
                                               identifier='dicarlo.Moeller2017-Experiment_2ii',
                                               metric=metric, performance_measure=Accuracy())

    def __call__(self, candidate):
        import copy
        candidate_copy = copy.copy(candidate)  # TODO: why is this done
        return self.benchmark1_outside_faces(candidate), self.benchmark2_objects(candidate_copy)


def Moeller2017Experiment3():
    """
    Stimulate face patch during face & non-face object eliciting patch response identification
    15 black & white round objects + faces; 3 exemplars per category  (apples, citrus, teapots, alarmclocks, faces)
    """
    return _Moeller2017(stimulus_class='Eliciting_Face_Response',
                        perturbation_location='within_facepatch',
                        identifier='dicarlo.Moeller2017-Experiment_3',
                        metric=PerformanceSimilarity(),
                        performance_measure=Accuracy())


def Moeller2017Experiment4a():
    """
    Stimulate face patch during abstract face identification
    16 Face Abtractions; 4 per category (Line Drawings, Silhouettes, Cartoons, Mooney Faces)
    """
    return _Moeller2017(stimulus_class='Abstract_Faces',
                        perturbation_location='within_facepatch',
                        identifier='dicarlo.Moeller2017-Experiment_4a',
                        metric=PerformanceSimilarity(),
                        performance_measure=Accuracy())


def Moeller2017Experiment4b():
    """
    Stimulate face patch during face & abstract houses identification
    20 Abstract Houses & Faces; 4 per category (House Line Drawings, House Cartoons, House Silhouettes, Mooney Faces, Mooney Faces up-side-down)
    """
    return _Moeller2017(stimulus_class='Abstract_Houses',
                        perturbation_location='within_facepatch',
                        identifier='dicarlo.Moeller2017-Experiment_4b',
                        metric=PerformanceSimilarity(),
                        performance_measure=Accuracy())
