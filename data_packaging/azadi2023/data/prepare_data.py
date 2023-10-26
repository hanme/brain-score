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

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly

# Azadi2023 - behavioral task
# "Throughout the training phase and all the experiments, 50% of the trials were ‘no-stimulation.’ 
# The other 50% were trials in which an opto-array was activated. In ‘stimulation’ trials (40%–50% 
# of all trials depending on the experiment and monkey, see experimental conditions for details), 
# the opto-array on the virus-expressed site was activated and in ‘catch’ trials (0%–10% of all 
# trials) the optoarray on the control site was activated."
# --> 50% no-stimulation, 40-50% stimulation, 0-10% catch
monkey_Ph_test_stimuli_base = 'Azadi2023_stimuli_monkey_Ph_test_stimulus_'


assembly = BehavioralAssembly(['dog', 'dog', 'cat', 'dog', ...],
                               coords={
                                   'stimulus_id': ('presentation', 
                                                   [f"{monkey_Ph_test_stimuli_base}{i:02}" for i in range(1, 41)]),
                                   #'sample_object': ('presentation', ['dog', 'cat', 'cat', 'dog', ...]),
                                   #'distractor_object': ('presentation', ['cat', 'dog', 'dog', 'cat', ...]),
                                   # ...more meta
                                   # Note that meta from the StimulusSet will automatically be merged into the
                                   #  presentation dimension:
                                   #  https://github.com/brain-score/brainio/blob/d0ac841779fb47fa7b8bdad3341b68357c8031d9/brainio/fetch.py#L125-L132
                               },
                               dims=['presentation'])
assembly.name = 'azadi2023_monkey_Ph_test'  # give the assembly an identifier name

# make sure the assembly is what you would expect
# "[...] and 10 [..] sessions were performed with a total of 17,033 trials, and an overall performance of 
# 84.6% [...] correct (catch trials excluded) [...].
assert len(assembly['presentation']) == 17033
assert len(set(assembly['stimulus_id'].values)) == 40
assert len(set(assembly['choice'].values)) == len(set(assembly['sample_object'].values)) \
       == len(set(assembly['distractor_object'].values)) == 2

