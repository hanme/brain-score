# Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

from model_tools.check_submission import check_models
import numpy as np
import torch
from torch import nn
import functools
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from model_tools.activations.pytorch import load_preprocess_images
from brainscore import score_model
from torch.nn.functional import conv1d
from torch.nn.functional import conv2d
from torch.nn.functional import conv3d
from torch.nn.functional import conv_transpose1d
from torch.nn.functional import conv_transpose2d
from torch.nn.functional import conv_transpose3d

"""
Template module for a base model submission to brain-score
"""

class ShiftedReLU(nn.Module):

    def __init__(self, shrink=0):
        if shrink < 0:
            raise ValueError("Shrink must be >= 0")
        super().__init__()
        self.ReLU = nn.ReLU()
        self.shrink = shrink

    def forward(self, input):
        return self.ReLU(input - self.shrink)

class ConvSparseLayer(nn.Module):
    """
    An implementation of a Convolutional Sparse Layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, shrink=0.05, lam=0.1, activation_lr=1e-1,
                 activation_iter=30, rectifier=True, convo_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_dim = convo_dim
        self.activation_lr = activation_lr
        self.activation_iter = activation_iter

        self.filters = nn.Parameter(torch.rand((out_channels, in_channels) +
                                               self.conv_dim * (kernel_size,)),
                                    requires_grad=True)
        torch.nn.init.xavier_uniform_(self.filters)
        self.normalize_weights()

        self.shrink = shrink
        if rectifier:
            self.threshold = ShiftedReLU(shrink)
        else:
            self.threshold = nn.Softshrink(shrink)

        if self.conv_dim == 1:
            self.convo = conv1d
            self.deconvo = conv_transpose1d
        elif self.conv_dim == 2:
            self.convo = conv2d
            self.deconvo = conv_transpose2d
        elif self.conv_dim == 3:
            self.convo = conv3d
            self.deconvo = conv_transpose3d
        else:
            raise ValueError("Conv_dim must be 1, 2, or 3")

        self.recon_loss = torch.nn.MSELoss(reduction='mean')
        self.lam = lam

    def normalize_weights(self):
        with torch.no_grad():
            norms = torch.norm(self.filters.reshape(
                self.out_channels, self.in_channels, -1), dim=2, keepdim=True)
            norms = torch.max(norms, 1e-12*torch.ones_like(norms)).view(
                (self.out_channels, self.in_channels) +
                len(self.filters.shape[2:])*(1,)).expand(self.filters.shape)
            self.filters.div_(norms)

    def reconstructions(self, activations):
        return self.deconvo(activations, self.filters, padding=self.padding,
                            stride=self.stride)

    def loss(self, images, activations):
        reconstructions = self.reconstructions(activations)
        loss = 0.5 * self.recon_loss(images, reconstructions)
        loss += self.lam * torch.mean(torch.sum(torch.abs(
            activations.reshape(activations.shape[0], -1)), dim=1))
        return loss

    def u_grad(self, u, images):
        acts = self.threshold(u)
        #acts = acts.cuda()
        recon = self.deconvo(acts, self.filters, padding=self.padding,
                             stride=self.stride)
        #recon2 = recon[:,:,:224,:224]
        e = images - recon
        #u = u.cuda()
        du = -u
        #e = e.cuda()
        con = self.convo(e, self.filters,padding=self.padding, stride=self.stride)
        #con = con[:,:,:224,:224]
        du = du + con
        du += acts
        return du

    def activations(self, images):
        with torch.no_grad():
            u = nn.Parameter(torch.zeros((images.shape[0], self.out_channels) +
                                         images.shape[2:]))
            optimizer = torch.optim.AdamW([u], lr=self.activation_lr)
            for i in range(self.activation_iter):
                tmp = -self.u_grad(u,images)
                #u = u.cuda()
                u.grad = tmp
                optimizer.step()

        return self.threshold(u)

    def forward(self, images):
        return self.activations(images)


#mat = scipy.io.loadmat('eightfilt.mat')
#mat = scipy.io.loadmat('newfilters5k_nomom.mat')
#dic = torch.from_numpy((mat['weight_vals'].astype(np.float32)))
#dic = dic.permute(3,2,1,0)
#dic = dic.float() /1


#ALEXNET MODIFIED
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = ConvSparseLayer(in_channels=3, out_channels=4, kernel_size=3)
        # self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
        #self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4)
        #self.conv1.weight.data = dic
        self.relu1 = torch.nn.ReLU()
        linear_input_size = np.power((224 - 3 + 2 * 0) / 1 + 1, 2) * 2
        self.linear = torch.nn.Linear(int(linear_input_size), 1000)
        self.relu2 = torch.nn.ReLU()  # can't get named ReLU output otherwise

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu2(x)
        return x


# init the model and the preprocessing:
preprocessing = functools.partial(load_preprocess_images, image_size=224)

# get an activations model from the Pytorch Wrapper
activations_model = PytorchWrapper(identifier='sparse-layer-model', model=MyModel(), preprocessing=preprocessing)

# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier='sparse-layer-model', activations_model=activations_model,
                        # specify layers to consider
                        layers=['conv1', 'relu1', 'relu2'])


# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    return ['sparse-layer-model']


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'sparse-layer-model'

    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = 224
    return wrapper


# get_layers method to tell the code what layers to consider. If you are submitting a custom
# model, then you will most likley need to change this method's return values.
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

    # quick check to make sure the model is the correct one:
    assert name == 'sparse-layer-model'

    # returns the layers you want to consider
    return  ['conv1', 'relu1', 'relu2']

# Bibtex Method. For submitting a custom model, you can either put your own Bibtex if your
# model has been published, or leave the empty return value if there is no publication to refer to.
def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """

    # from pytorch.py:
    return ''

# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)