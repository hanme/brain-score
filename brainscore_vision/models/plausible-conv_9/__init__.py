from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-18-LC_w_sh_1_iter'] = ModelCommitment(identifier='resnet-18-LC_w_sh_1_iter', activations_model=get_model('resnet-18-LC_w_sh_1_iter'), layers=get_layers('resnet-18-LC_w_sh_1_iter'))
