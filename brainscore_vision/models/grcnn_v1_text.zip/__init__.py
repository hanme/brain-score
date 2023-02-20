from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['grcnn_v1_text'] = ModelCommitment(identifier='grcnn_v1_text', activations_model=get_model('grcnn_v1_text'), layers=get_layers('grcnn_v1_text'))
