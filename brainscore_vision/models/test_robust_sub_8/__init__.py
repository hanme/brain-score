from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['tf_efficientnet_b1_ns_finetune_robust_linf8255_e1_324x288'] = ModelCommitment(identifier='tf_efficientnet_b1_ns_finetune_robust_linf8255_e1_324x288', activations_model=get_model('tf_efficientnet_b1_ns_finetune_robust_linf8255_e1_324x288'), layers=get_layers('tf_efficientnet_b1_ns_finetune_robust_linf8255_e1_324x288'))
