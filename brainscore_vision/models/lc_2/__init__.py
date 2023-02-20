from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['0.5x_resnet-18_LC'] = ModelCommitment(identifier='0.5x_resnet-18_LC', activations_model=get_model('0.5x_resnet-18_LC'), layers=get_layers('0.5x_resnet-18_LC'))
