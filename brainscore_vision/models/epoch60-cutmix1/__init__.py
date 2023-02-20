from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['imgnfull-e60-cut1'] = ModelCommitment(identifier='imgnfull-e60-cut1', activations_model=get_model('imgnfull-e60-cut1'), layers=get_layers('imgnfull-e60-cut1'))
