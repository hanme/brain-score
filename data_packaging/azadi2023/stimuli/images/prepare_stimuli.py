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

from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

stimuli = []  # collect meta
stimulus_paths = {}  # collect mapping of stimulus_id to filepath
for filepath in Path("/home/mehrer/projects/perturbations/perturbation_tests/brain-score-perturbation/data_packaging/azadi2023/stimuli/images").glob('*.png'):
    stimulus_id = filepath.stem
    monkey = filepath.stem.split('_')[3]
    train_test = filepath.stem.split('_')[4]
    stim_idx = filepath.stem.split('_')[6]
    stimulus_paths[stimulus_id] = filepath
    stimuli.append({
        'stimulus_id': stimulus_id,
        'monkey': monkey,
        'train_test': train_test,
        'stim_idx': stim_idx,
        # optionally you can set 'stimulus_path_within_store' to define the filename in the packaged stimuli
    })
stimuli = StimulusSet(stimuli)
stimuli.stimulus_paths = stimulus_paths
stimuli.name = 'Azadi2023'  # give the StimulusSet an identifier name

assert len(stimuli) == 124  # make sure the StimulusSet is what you would expect

# save as pickle 
stimuli.to_pickle(Path(__file__).parent.parent / 'metadata.pkl')
#package_stimulus_set(stimuli, stimulus_set_identifier=stimuli.name)  # upload to S3