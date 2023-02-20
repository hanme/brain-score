from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50_380'] = ModelCommitment(identifier='resnet50_380', activations_model=get_model('resnet50_380'), layers=get_layers('resnet50_380'))
