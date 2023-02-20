from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['This'] = ModelCommitment(identifier='This', activations_model=get_model('This'), layers=get_layers('This'))
model_registry['is'] = ModelCommitment(identifier='is', activations_model=get_model('is'), layers=get_layers('is'))
model_registry['a'] = ModelCommitment(identifier='a', activations_model=get_model('a'), layers=get_layers('a'))
model_registry['model'] = ModelCommitment(identifier='model', activations_model=get_model('model'), layers=get_layers('model'))
model_registry['for'] = ModelCommitment(identifier='for', activations_model=get_model('for'), layers=get_layers('for'))
model_registry['2023_1_28_19_50'] = ModelCommitment(identifier='2023_1_28_19_50', activations_model=get_model('2023_1_28_19_50'), layers=get_layers('2023_1_28_19_50'))
model_registry['lzh'] = ModelCommitment(identifier='lzh', activations_model=get_model('lzh'), layers=get_layers('lzh'))
