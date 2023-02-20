from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetb1_cutmixpatch_augmix_e5_342x288'] = ModelCommitment(identifier='effnetb1_cutmixpatch_augmix_e5_342x288', activations_model=get_model('effnetb1_cutmixpatch_augmix_e5_342x288'), layers=get_layers('effnetb1_cutmixpatch_augmix_e5_342x288'))
