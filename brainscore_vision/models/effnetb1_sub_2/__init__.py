from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetb1_cutmix_augmix_epoch4_324x288_manylayers'] = ModelCommitment(identifier='effnetb1_cutmix_augmix_epoch4_324x288_manylayers', activations_model=get_model('effnetb1_cutmix_augmix_epoch4_324x288_manylayers'), layers=get_layers('effnetb1_cutmix_augmix_epoch4_324x288_manylayers'))
