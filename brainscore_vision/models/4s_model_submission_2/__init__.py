from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['sketch_model-4s-ep1-weights'] = ModelCommitment(identifier='sketch_model-4s-ep1-weights', activations_model=get_model('sketch_model-4s-ep1-weights'), layers=get_layers('sketch_model-4s-ep1-weights'))
