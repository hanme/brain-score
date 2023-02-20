from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['dorinet_cornet_z_40_V1'] = ModelCommitment(identifier='dorinet_cornet_z_40_V1', activations_model=get_model('dorinet_cornet_z_40_V1'), layers=get_layers('dorinet_cornet_z_40_V1'))
