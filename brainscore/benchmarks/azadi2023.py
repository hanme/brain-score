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

import functools
import logging

import numpy as np
import xarray as xr
from brainio.assemblies import merge_data_arrays, DataAssembly, walk_coords, array_is_element
from numpy.random import RandomState
from tqdm import tqdm
from xarray import DataArray

from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.afraz2006 import mean_var
from brainscore.metrics import Score
from brainscore.metrics.difference_of_correlations import DifferenceOfCorrelations
from brainscore.metrics.difference_of_fractions import DifferenceOfFractions
from brainscore.metrics.significant_match import SignificantCorrelation, SignificantPerformanceChange, \
    is_significantly_different, NoSignificantPerformanceChange
from brainscore.model_interface import BrainModel
from brainscore.utils import fullname
"""
from data_packaging.azadi2023 import collect_stimuli, collect_stimulation_report_rate_training, \
    collect_detection_profile, collect_corr_between_detection_profiles, \
    collect_psychometric_functions_illumination_power, collect_psychometric_functions_image_visibility
"""
from data_packaging.azadi2023 import collect_stimuli as collect_stimuli_Azadi2023
from data_packaging.azadi2023 import collect_detection_profile
from data_packaging.afraz2015 import muscimol_delta_overall_accuracy, collect_stimuli, collect_site_deltas, \
    collect_delta_overall_accuracy

BIBTEX = """{azadi_image-dependence_2023,
            title = {Image-dependence of the detectability of optogenetic stimulation in macaque inferotemporal cortex},
            volume = {33},
            issn = {09609822},
            url = {https://linkinghub.elsevier.com/retrieve/pii/S0960982222019212},
            doi = {10.1016/j.cub.2022.12.021},
            abstract = {Artiﬁcial activation of neurons in early visual areas induces perception of simple visual ﬂashes.1,2 Accordingly, stimulation in high-level visual cortices is expected to induce perception of complex features.3,4 However, results from studies in human patients challenge this expectation. Stimulation rarely induces any detectable visual event, and never a complex one, in human subjects with closed eyes.2 Stimulation of the face-selective cortex in a human patient led to remarkable hallucinations only while the subject was looking at faces.5 In contrast, stimulations of color- and face-selective sites evoke notable hallucinations independent of the object being viewed.6 These anecdotal observations suggest that stimulation of high-level visual cortex can evoke perception of complex visual features, but these effects depend on the availability and content of visual input. In this study, we introduce a novel psychophysical task to systematically investigate characteristics of the perceptual events evoked by optogenetic stimulation of macaque inferior temporal (IT) cortex. We trained macaque monkeys to detect and report optogenetic impulses delivered to their IT cortices7–9 while holding ﬁxation on object images. In a series of experiments, we show that detection of cortical stimulation is highly dependent on the choice of images presented to the eyes and it is most difﬁcult when ﬁxating on a blank screen. These ﬁndings suggest that optogenetic stimulation of high-level visual cortex results in easily detectable distortions of the concurrent contents of vision.},
            language = {en},
            number = {3},
            urldate = {2023-07-29},
            journal = {Current Biology},
            author = {Azadi, Reza and Bohn, Simon and Lopez, Emily and Lafer-Sousa, Rosa and Wang, Karen and Eldridge, Mark A.G. and Afraz, Arash},
            month = feb,
            year = {2023},
            pages = {581--588.e4}
        }"""

OPTOGENETIC_PARAMETERS = {
    # "At each [of 16] injection site, 10 μl of virus was injected at a 0.5 μl/min rate, for a total volume of injection of 160 uL."
    # Azadi2023 used an array of 16 injection sites. Needs adjustment?
    "amount_microliter": 10,  
    "rate_microliter_per_min": 0.5,
    # "We then injected AAV5-CaMKIIa-C1V1(t/t)-EYFP (nominal titer: 8x10^12 particles/ml) into the cortex"
    "virus": "AAV5-CaMKIIa-C1V1(t/t)-EYFP",
    "infectious_units_per_ml": 8E12,
    # "a 200 ms illumination impulse was delivered to IT cortex halfway through the image presentation"
    "laser_pulse_duration_ms": 200,
    # "illumination power of 3.6 mW [monkey Ph] and 5.4 mW [monkey Sp]"
    "fiber_output_power_mW": 3.6, # monkey Ph
    #"fiber_output_power_mW_monkey_Sp": 5.4, # monkey Sp
}

# Azadi2023
def load_stimuli_Azadi2023():
    #Retrieve training and testing stimuli 
    stimuli = collect_stimuli_Azadi2023()
    monkey_Ph_training_stimuli = stimuli[(stimuli['monkey'] == 'Ph') & \
                                         (stimuli['train_test'] == 'train')]
    monkey_Ph_testing_stimuli = stimuli[(stimuli['monkey'] == 'Ph') & \
                                        (stimuli['train_test'] == 'test')]
    monkey_Sp_training_stimuli = stimuli[(stimuli['monkey'] == 'Sp') & \
                                         (stimuli['train_test'] == 'train')]
    monkey_Sp_testing_stimuli = stimuli[(stimuli['monkey'] == 'Sp') & \
                                        (stimuli['train_test'] == 'test')]
    return monkey_Ph_training_stimuli, monkey_Ph_testing_stimuli, \
        monkey_Sp_training_stimuli, monkey_Sp_testing_stimuli


# Afraz2015
def load_stimuli():
    """ Retrieve gender and selectivity (object/face) stimuli """
    stimuli = collect_stimuli()
    gender_stimuli = stimuli[stimuli['category'].isin(['male', 'female'])]
    selectivity_stimuli = stimuli[stimuli['category'].isin(['object', 'face'])]
    gender_stimuli['image_label'] = gender_stimuli['category']
    gender_stimuli.identifier = stimuli.identifier + '-gender'
    selectivity_stimuli.identifier = stimuli.identifier + '-selectivity'
    return gender_stimuli, selectivity_stimuli

    
# Afraz2015
def split_train_test(stimuli, random_state, num_training, num_testing):
    train_stimuli = stimuli.sample(n=num_training, replace=False, random_state=random_state)
    remaining_stimuli = stimuli[~stimuli['stimulus_id'].isin(train_stimuli['stimulus_id'])]
    test_stimuli = remaining_stimuli.sample(n=num_testing, replace=False, random_state=random_state)
    train_stimuli.identifier = stimuli.identifier + f'-train{num_training}'
    test_stimuli.identifier = stimuli.identifier + f'-test{num_testing}'
    return train_stimuli, test_stimuli

# [...] in the same surgery, we implanted a second Opto-Array on a similar area of 
# the IT cortex in the opposite hemisphere (control site: right hemisphere in Sp, 
# and left hemisphere in Ph) where no virus injection was performed.
# --> Monkey Ph, stimulation site: right hemisphere, control site: left hemisphere
# --> Monkey Sp, stimulation site: left hemisphere, control site: right hemisphere

# "Monkeys were trained to perform a detection task in which they were rewarded if they correctly 
# identified whether a trial did or did not contain an optogenetic stimulation impulse. [...]
# an image (scaled so the largest dimension spanned 8° for most images and 30° for four scenes 
# during training and two scenes in experiment 1) appeared on the screen for 1000 ms while the 
# animal held fixation on a central target. In half of the trials (randomly selected) 500 ms from 
# the image onset, an LED on one of the Opto-Arrays was activated for 200 ms.


class _Azadi2023Optogenetics(BenchmarkBase):
    def __init__(self, metric_identifier, metric):
        self._logger = logging.getLogger(fullname(self))

        # Load the stimuli using the provided function.
        monkey_Ph_training_stimuli, monkey_Ph_testing_stimuli, \
        _, _ = load_stimuli_Azadi2023() # only using data from monkey Ph for now
        self._fitting_stimuli = monkey_Ph_training_stimuli
        self._test_stimuli = monkey_Ph_testing_stimuli

        self._assembly = self.collect_assembly()
        self._assembly.attrs['stimulus_set'] = self._test_stimuli
        self._metric = metric

        super(_Azadi2023Optogenetics, self).__init__(
            identifier='Azadi2023.optogenetics-' + metric_identifier,
            ceiling_func=None,
            version=1, parent='IT',
            bibtex=BIBTEX)

    def collect_assembly(self):
        return collect_detection_profile()

    def __call__(self, candidate: BrainModel):        
        candidate.start_task(BrainModel.Task.passive, fitting_stimuli=None)
        candidate.start_recording('IT', time_bins=[(50, 100)],
            hemisphere=BrainModel.Hemisphere.right, recording_type=BrainModel.RecordingType.exact) # monkey Ph: right hemisphere --> stimulation site
        candidate.perturb(perturbation=None, target='IT')  # reset
        hemisphere_recordings_unperturbed = candidate.look_at(self._assembly.stimulus_set)      
        print('hemisphere_recordings_unperturbed: ', hemisphere_recordings_unperturbed)  
        print('hemisphere_recordings_unperturbed.shape: ', hemisphere_recordings_unperturbed.shape)

        location = [7.74, 5.9]
        self._logger.debug(f"Activating at {location}")
        candidate.perturb(perturbation=BrainModel.Perturbation.optogenetic_activation,
                          target='IT', perturbation_parameters={
            **{'location': location}, **OPTOGENETIC_PARAMETERS})

        hemisphere_recordings_perturbed = candidate.look_at(self._assembly.stimulus_set)
        print('hemisphere_recordings_perturbed: ', hemisphere_recordings_perturbed)
        print('hemisphere_recordings_perturbed.shape: ', hemisphere_recordings_perturbed.shape)

        return self.score_behaviors(hemisphere_recordings_unperturbed, hemisphere_recordings_perturbed)

    def score_behaviors(self, unperturbed_behavior, perturbed_behavior):
        # XXX
        return 42
        

def Azadi2023OptogeneticActivationUnperturbedPerturbed():
    def behaviors_activation_metric(unperturbed_behavior, perturbed_behavior):
        # XXX
        return unperturbed_behavior, perturbed_behavior

    return _Azadi2023Optogenetics(
        metric_identifier='unperturbed_perturbed_activation',
        metric=behaviors_activation_metric)



"""
# Afraz2015
class _Afraz2015Optogenetics(BenchmarkBase):
    def __init__(self, metric_identifier, metric):
        self._logger = logging.getLogger(fullname(self))

        gender_stimuli, self._selectivity_stimuli = load_stimuli()
        # "In practice, we first trained the animals on a fixed set of 400 images (200 males and 200 females)."
        # "Once trained, we tested the animals’ performance on freshly generated sets of 400 images to confirm
        #  that they could generalize the learning to novel stimuli"
        self._fitting_stimuli, test_stimuli = split_train_test(gender_stimuli, random_state=RandomState(42),
                                                                num_training=400, num_testing=400)
        self._num_face_detector_sites = 17  # "photosuppression at high-FD sites (n = 17 sites) [...]"

        self._assembly = self.collect_assembly()
        self._assembly.attrs['stimulus_set'] = test_stimuli
        self._metric = metric
        super(_Afraz2015Optogenetics, self).__init__(
            identifier='dicarlo.Afraz2015.optogenetics-' + metric_identifier,
            ceiling_func=None,
            version=1, parent='IT',
            bibtex=BIBTEX)

    def collect_assembly(self):
        return collect_detection_profile()

    def __call__(self, candidate: BrainModel):        
        # Afraz 2015: "In practice, we first trained the animals on a fixed set of 400 images (200 males and 200 females)."
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli=self._fitting_stimuli)
        unperturbed_behavior = candidate.look_at(self._assembly.stimulus_set)        
        candidate_behaviors = []
        recordings = []
        for hemisphere in [BrainModel.Hemisphere.left, BrainModel.Hemisphere.right]:
            # record to determine face-selectivity
            candidate.start_task(BrainModel.Task.passive, fitting_stimuli=None)
            candidate.start_recording('IT', time_bins=[(50, 100)],
                                      hemisphere=hemisphere, recording_type=BrainModel.RecordingType.exact)
            hemisphere_recordings = candidate.look_at(self._selectivity_stimuli)

            # sub-select recordings to match sites in experiment
            hemisphere_recordings = self.subselect_recordings(hemisphere_recordings)
            recordings.append(hemisphere_recordings)

            candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli=self._fitting_stimuli)
            # This benchmark ignores the parafoveal presentation of images.
            suppression_locations = np.stack((hemisphere_recordings['tissue_x'],
                                              hemisphere_recordings['tissue_y'])).T.tolist()
            for site, location in enumerate(tqdm(suppression_locations, desc='injection locations')):
                candidate.perturb(perturbation=None, target='IT')  # reset
                location = np.round(location, decimals=2)
                self._logger.debug(f"Suppressing at {location}")
                candidate.perturb(perturbation=BrainModel.Perturbation.optogenetic_suppression,
                                  target='IT', perturbation_parameters={
                        **{'location': location}, **OPTOGENETIC_PARAMETERS})
                behavior = candidate.look_at(self._assembly.stimulus_set)
                behavior = behavior.expand_dims('site')
                behavior['site_iteration'] = 'site', [site]
                behavior['site_x'] = 'site', [location[0]]
                behavior['site_y'] = 'site', [location[1]]
                behavior['hemisphere'] = 'site', [hemisphere]
                behavior = type(behavior)(behavior)  # make sure site is indexed
                candidate_behaviors.append(behavior)
        candidate.perturb(perturbation=None, target='IT')  # reset
        candidate_behaviors = merge_data_arrays(candidate_behaviors)
        recordings = merge_data_arrays(recordings)

        return self.score_behaviors(candidate_behaviors, unperturbed_behavior, recordings)

    def score_behaviors(self, candidate_behaviors, unperturbed_behavior, recordings):
        accuracies = characterize_delta_accuracies(unperturbed_behavior=unperturbed_behavior,
                                                   perturbed_behaviors=candidate_behaviors)
        # face selectivities
        selectivities = determine_selectivity(recordings)
        attach_selectivity(accuracies, selectivities, coord_name='face_detection_index_dprime')
        # compare
        score = self._metric(accuracies, self._assembly)
        return score

    def subselect_recordings(self, recordings):
        # We here randomly sub-select the recordings to match the number of stimulation sites in the experiment, based
        # on the assumption that we can compare trend effects even with a random sample.
        num_nonface_detector_sites = 40 - self._num_face_detector_sites  # "40 experimental sessions" minus face sites
        face_detector_sites, nonface_detector_sites = find_selective_sites(
            num_face_detector_sites=self._num_face_detector_sites,
            num_nonface_detector_sites=num_nonface_detector_sites,
            recordings=recordings)
        recordings = recordings[{'neuroid': [neuroid_id in (face_detector_sites + nonface_detector_sites)
                                             for neuroid_id in recordings['neuroid_id'].values]}]
        return recordings

# Afraz2015
def Afraz2015OptogeneticContraDeltaAccuracySignificant():
    metric = SignificantPerformanceChange(condition_name='laser_on', condition_value1=False, condition_value2=True)

    def filter_contra_metric(source_assembly, target_assembly):
        source_assembly = source_assembly[{'presentation': [hemisphere in ['left', np.nan]  # FIXME visual field
                                                            for hemisphere in source_assembly['hemisphere'].values]}]
        target_assembly = target_assembly.sel(visual_field='contra')
        return metric(source_assembly, target_assembly)

    return _Afraz2015OptogeneticOverallAccuracy(
        metric_identifier='delta_accuracy_significant',
        metric=filter_contra_metric)

# Afraz2015
class _Afraz2015OptogeneticOverallAccuracy(_Afraz2015Optogenetics):
    def collect_assembly(self):
        return collect_delta_overall_accuracy()

    def subselect_recordings(self, recordings):
        # We here randomly sub-select the recordings to match the number of stimulation sites in the experiment, based
        # on the assumption that we can compare trend effects even with a random sample.
        face_detector_sites, _ = find_selective_sites(
            num_face_detector_sites=self._num_face_detector_sites, num_nonface_detector_sites=0, recordings=recordings)
        recordings = recordings[{'neuroid': [neuroid_id in face_detector_sites
                                             for neuroid_id in recordings['neuroid_id'].values]}]
        return recordings

    def score_behaviors(self, candidate_behaviors, unperturbed_behavior, recordings):
        # face selectivities
        selectivities = determine_selectivity(recordings)
        attach_selectivity(candidate_behaviors, selectivities)

        # compute per condition accuracy
        unperturbed_accuracy = per_image_accuracy(unperturbed_behavior)
        site_accuracies = per_image_accuracy(candidate_behaviors)
        grouped_accuracy = self.group_accuracy(unperturbed_accuracy, site_accuracies)

        # compute score
        aggregate_target = self._assembly.sel(aggregation='center')
        score = self._metric(grouped_accuracy, aggregate_target)
        return score

    def group_accuracy(self, unperturbed_accuracy, site_accuracies):
        site_coords = site_accuracies['site']
        site_accuracies = stack_multiindex(site_accuracies, 'presentation')
        site_accuracies['laser_on'] = 'presentation', [True] * len(site_accuracies['presentation'])
        unperturbed_accuracy['laser_on'] = 'presentation', [False] * len(unperturbed_accuracy['presentation'])
        # in order to concatenate, we need the same coordinates on all data assemblies
        for coord, dims, values in walk_coords(site_coords):
            unperturbed_accuracy[coord] = 'presentation', [None] * len(unperturbed_accuracy['presentation'])
        grouped_accuracy = xr.concat([site_accuracies, DataAssembly(unperturbed_accuracy)], dim='presentation')
        return DataAssembly(grouped_accuracy)  # make sure MultiIndex is built

    def site_accuracies(self, unperturbed_behavior, perturbed_behaviors):
        unperturbed_accuracy = per_image_accuracy(unperturbed_behavior)

        site_accuracies = []
        for site in perturbed_behaviors['site'].values:
            # index instead of `.sel` to preserve all site coords
            behavior = perturbed_behaviors[{'site': [s == site for s in perturbed_behaviors['site'].values]}]
            site_coords = {coord: (dims, values) for coord, dims, values in walk_coords(behavior['site'])}
            behavior = behavior.squeeze('site', drop=True)

            site_image_accuracies = per_image_accuracy(behavior)

            site_image_accuracies = site_image_accuracies.expand_dims('site')
            for coord, (dims, values) in site_coords.items():
                site_image_accuracies[coord] = dims, values
            site_accuracies.append(DataAssembly(site_image_accuracies))
        site_accuracies = merge_data_arrays(site_accuracies)
        return unperturbed_accuracy, site_accuracies
    
# Afraz2015
def find_selective_sites(num_face_detector_sites, num_nonface_detector_sites, recordings,
                         # "face-selective units (defined as FD d′ > 1)" (SI Electrophysiology)
                         face_selectivity_threshold=1,
                         ):
    random_state = RandomState(seed=1)
    face_detector_sites, nonface_detector_sites = [], []
    while len(face_detector_sites) < num_face_detector_sites:
        neuroid_id = random_state.choice(recordings['neuroid_id'].values)
        selectivity = determine_selectivity(recordings[{'neuroid': [
            nid == neuroid_id for nid in recordings['neuroid_id'].values]}])
        if selectivity > face_selectivity_threshold:
            face_detector_sites.append(neuroid_id)
    while len(nonface_detector_sites) < num_nonface_detector_sites:
        neuroid_id = random_state.choice(recordings['neuroid_id'].values)
        selectivity = determine_selectivity(recordings[{'neuroid': [
            nid == neuroid_id for nid in recordings['neuroid_id'].values]}])
        if selectivity <= face_selectivity_threshold:
            nonface_detector_sites.append(neuroid_id)
    return face_detector_sites, nonface_detector_sites

# Afraz2015
def determine_selectivity(recordings):
    assert (recordings >= 0).all()
    # A d' value of zero indicates indistinguishable responses to faces and non-faces.
    # Increasingly positive d' values indicate progressively better selectivity for faces.
    # Selectivity for faces was defined as having a d' value > 1.
    result = []
    iterator = recordings['neuroid_id'].values
    if len(iterator) > 1:
        iterator = tqdm(iterator, desc='neuron face dprime')
    for neuroid_id in iterator:
        neuron = recordings.sel(neuroid_id=neuroid_id)
        neuron = neuron.squeeze()
        face_mean, face_variance = mean_var(neuron.sel(category='face'))
        nonface_mean, nonface_variance = mean_var(neuron.sel(category='object'))
        # face selectivity based on "more positive" firing
        dprime = (face_mean - nonface_mean) / np.sqrt((face_variance + nonface_variance) / 2)
        result.append(dprime)
    result = DataArray(result, coords={'neuroid_id': recordings['neuroid_id'].values}, dims=['neuroid_id'])
    return result

"""