from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['mobilenet_v3_small-LC-dyn-1x1'] = ModelCommitment(identifier='mobilenet_v3_small-LC-dyn-1x1', activations_model=get_model('mobilenet_v3_small-LC-dyn-1x1'), layers=get_layers('mobilenet_v3_small-LC-dyn-1x1'))
