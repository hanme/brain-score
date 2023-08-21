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
from data_packaging.azadi2023 import collect_stimuli, collect_stimulation_report_rate_training, \
    collect_detection_profile, collect_corr_between_detection_profiles, \
    collect_psychometric_functions_illumination_power, collect_psychometric_functions_image_visibility

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
    # XXX Azadi2023 used an array of 16 injection sites. How do we account for this here?
    "amount_microliter": 10,  
    "rate_microliter_per_min": 0.5,
    # "We then injected AAV5-CaMKIIa-C1V1(t/t)-EYFP (nominal titer: 8x10^12 particles/ml) into the cortex"
    "virus": "AAV5-CaMKIIa-C1V1(t/t)-EYFP",
    "infectious_units_per_ml": 8E12,
    # "a 200 ms illumination impulse was delivered to IT cortex halfway through the image presentation"
    "laser_pulse_duration_ms": 200,
    # "total fiber output power ∼12 mW"
    # "illumination power of 3.6 mW [monkey Ph] and 5.4 mW [monkey Sp]"
    "fiber_output_power_mW_monkey_Ph": 3.6, 
    "fiber_output_power_mW_monkey_Sp": 5.4,
}

class _Azadi2023Optogenetics(BenchmarkBase):
    def __init__(self, metric_identifier, metric):
        self._logger = logging.getLogger(fullname(self))
        monkey_Ph_training_stimuli, monkey_Ph_testing_stimuli, \
        monkey_Sp_training_stimuli, monkey_Sp_testing_stimuli = \
            load_stimuli()
        #self._fitting_stimuli, test_stimuli = split_train_test(gender_stimuli, random_state=RandomState(42),
        #                                                       num_training=400, num_testing=200)

        # [...] in the same surgery, we implanted a second Opto-Array on a similar area of 
        # the IT cortex in the opposite hemisphere (control site: right hemisphere in Sp, 
        # and left hemisphere in Ph) where no virus injection was performed.
        # --> Monkey Ph, stimulation site: right hemisphere, control site: left hemisphere
        # --> Monkey Sp, stimulation site: left hemisphere, control site: right hemisphere
        self._num_IT_detector_sites = 1
        self._num_catch_detector_sites = 1

        self._assembly = self.collect_assembly()
        self._assembly.attrs['Monkey_Ph_stimulus_set'] = monkey_Ph_testing_stimuli
        self._assembly.attrs['Monkey_Sp_stimulus_set'] = monkey_Sp_testing_stimuli
        self._metric = metric
        super(_Azadi2023Optogenetics, self).__init__(
            identifier='dicarlo.Azadi2023.optogenetics-' + metric_identifier,
            ceiling_func=None,
            version=1, parent='IT',
            bibtex=BIBTEX)
        
    def collect_assembly(self):
        return collect_detection_profile()
    
        
    def __call__(self, candidate: BrainModel):
        # "Monkeys were trained to perform a detection task in which they were rewarded if they correctly 
        # identified whether a trial did or did not contain an optogenetic stimulation impulse. [...]
        # an image (scaled so the largest dimension spanned 8% for most images and 30% for four scenes 
        # during training and two scenes in experiment 1) appeared on the screen for 1000 ms while the 
        # animal held fixation on a central target. In half of the trials (randomly selected) 500 ms from 
        # the image onset, an LED on one of the Opto-Arrays was activated for 200 ms.
        # --> not using candidate.start_task(BrainModel.Task.passive) to model passive viewing as not 
        #     relevant here. 
        # 
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli=self._fitting_stimuli)
        unperturbed_behavior = candidate.look_at(self._assembly.stimulus_set)

        """
        candidate_behaviors = []   
        candidate.start_task(BrainModel.Task.passive, fitting_stimuli=None)
        candidate.start_recording('IT', time_bins=[(50, 100)],
                                    hemisphere=hemisphere, recording_type=BrainModel.RecordingType.exact)
        hemisphere_recordings = candidate.look_at(self._selectivity_stimuli)
        """


from sklearn.metrics import mean_squared_error
metric_identifier = "mse"
metric = mean_squared_error
afraz_optogenetics = _Azadi2023Optogenetics(metric_identifier, metric)
afraz_optogenetics(candidate='')



def Azadi2023OptogeneticDetectionPerformanceSTDdiff():


    return _Azadi2023Optogenetics(
        metric_identifier='XXX',
        metric=XXX)






def load_stimuli():
    """ Retrieve training and testing stimuli """
    stimuli = collect_stimuli()
    monkey_Ph_training_stimuli = stimuli[(stimuli['monkey'] == 'Ph') & \
                                         (stimuli['train_test'] == 'train')]
    monkey_Ph_testing_stimuli = stimuli[(stimuli['monkey'] == 'Ph') & \
                                        (stimuli['train_test'] == 'test')]
    monkey_Sp_training_stimuli = stimuli[(stimuli['monkey'] == 'Sp') & \
                                         (stimuli['train_test'] == 'train')]
    monkey_Sp_testing_stimuli = stimuli[(stimuli['monkey'] == 'Sp') & \
                                        (stimuli['train_test'] == 'test')]
    assert len(monkey_Ph_training_stimuli) == 22
    assert len(monkey_Ph_testing_stimuli) == 40
    assert len(monkey_Sp_training_stimuli) == 22
    assert len(monkey_Sp_testing_stimuli) == 40
    return monkey_Ph_training_stimuli, monkey_Ph_testing_stimuli, \
        monkey_Sp_training_stimuli, monkey_Sp_testing_stimuli


#def split_train_test(stimuli, random_state, num_training, num_testing):
    
    
    #train_stimuli = stimuli.sample(n=num_training, replace=False, random_state=random_state)
    #remaining_stimuli = stimuli[~stimuli['stimulus_id'].isin(train_stimuli['stimulus_id'])]
    #test_stimuli = remaining_stimuli.sample(n=num_testing, replace=False, random_state=random_state)
    #train_stimuli.identifier = stimuli.identifier + f'-train{num_training}'
    #test_stimuli.identifier = stimuli.identifier + f'-test{num_testing}'
    #return train_stimuli, test_stimuli


this = _Azadi2023Optogenetics(metric_identifier = 'abcd', metric = 'efgh')