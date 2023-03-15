import functools
from model_tools.check_submission import check_models
import os
import torch
import torchvision.models as models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images

"""
Template module for a base model submission to brain-score
"""
current_model_names = [
    'resnet50_batch512_3steps_eps0.01-best',
    'resnet50_batch512_3steps_eps0.02-best',
    'resnet50_batch512_3steps_eps0.05-best',
    'resnet50_batch512_3steps_eps0.1-best',
    'resnet50_batch512_3steps_eps0.2-best',
    'resnet50_batch512_3steps_eps0.5-best',
    'resnet50_batch512_3steps_eps1-best',
    'resnet50_batch512_3steps_eps2-best',
    'resnet50_batch512_3steps_eps5-best',
    'resnet50_batch512_3steps_eps10-best',
    'resnet50_batch512_3steps_eps20-best',
    'resnet50_batch512_3steps_eps50-best',
    'resnet50_batch512_3steps_eps100-best'
]

def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return current_model_names


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name in current_model_names
    modelname = name.split('-')[0]
    model = models.resnet50(pretrained=False)
    url = f'https://yug-model-robustness.s3.amazonaws.com/extracted_models/{modelname}/{modelname}-checkpoint.pt.best'
    state_dict = torch.utils.model_zoo.load_url(url, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    assert name in current_model_names
    return ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
