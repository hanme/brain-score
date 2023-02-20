from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['convnext_small'] = ModelCommitment(identifier='convnext_small', activations_model=get_model('convnext_small'), layers=get_layers('convnext_small'))
