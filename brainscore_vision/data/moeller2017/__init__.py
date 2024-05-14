from brainscore_vision import data_registry
from .packaging import collect_target_assembly

BIBTEX = '''@article{article,
            author = {Moeller, Sebastian and Crapse, Trinity and Chang, Le and Tsao, Doris},
            year = {2017},
            month = {03},
            pages = {},
            title = {The effect of face patch microstimulation on perception of faces and objects},
            volume = {20},
            journal = {Nature Neuroscience},
            doi = {10.1038/nn.4527}
            }'''

data_registry['Moeller2017.experiment1'] = lambda: collect_target_assembly(
    stimulus_class='Faces', perturbation_location='within_facepatch')
