from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet_blocks_seed30'] = ModelCommitment(identifier='resnet_blocks_seed30', activations_model=get_model('resnet_blocks_seed30'), layers=get_layers('resnet_blocks_seed30'))
