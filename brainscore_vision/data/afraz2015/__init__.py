from brainscore_vision import data_registry, stimulus_set_registry
from .packaging import muscimol_delta_overall_accuracy, collect_site_deltas, collect_delta_overall_accuracy, \
    collect_stimuli

BIBTEX = """@article {Afraz6730,
            author = {Afraz, Arash and Boyden, Edward S. and DiCarlo, James J.},
            title = {Optogenetic and pharmacological suppression of spatial clusters of face neurons reveal their causal role in face gender discrimination},
            volume = {112},
            number = {21},
            pages = {6730--6735},
            year = {2015},
            doi = {10.1073/pnas.1423328112},
            publisher = {National Academy of Sciences},
            abstract = {There exist subregions of the primate brain that contain neurons that respond more to images of faces over other objects. These subregions are thought to support face-detection and discrimination behaviors. Although the role of these areas in telling faces from other objects is supported by direct evidence, their causal role in distinguishing faces from each other lacks direct experimental evidence. Using optogenetics, here we reveal their causal role in face-discrimination behavior and provide a mechanistic explanation for the process. This study is the first documentation of behavioral effects of optogenetic intervention in primate object-recognition behavior. The methods developed here facilitate the usage of the technical advantages of optogenetics for future studies of high-level vision.Neurons that respond more to images of faces over nonface objects were identified in the inferior temporal (IT) cortex of primates three decades ago. Although it is hypothesized that perceptual discrimination between faces depends on the neural activity of IT subregions enriched with {\textquotedblleft}face neurons,{\textquotedblright} such a causal link has not been directly established. Here, using optogenetic and pharmacological methods, we reversibly suppressed the neural activity in small subregions of IT cortex of macaque monkeys performing a facial gender-discrimination task. Each type of intervention independently demonstrated that suppression of IT subregions enriched in face neurons induced a contralateral deficit in face gender-discrimination behavior. The same neural suppression of other IT subregions produced no detectable change in behavior. These results establish a causal link between the neural activity in IT face neuron subregions and face gender-discrimination behavior. Also, the demonstration that brief neural suppression of specific spatial subregions of IT induces behavioral effects opens the door for applying the technical advantages of optogenetics to a systematic attack on the causal relationship between IT cortex and high-level visual perception.},
            issn = {0027-8424},
            URL = {https://www.pnas.org/content/112/21/6730},
            eprint = {https://www.pnas.org/content/112/21/6730.full.pdf},
            journal = {Proceedings of the National Academy of Sciences}
        }"""

data_registry['Afraz2015.muscimol_overall'] = muscimol_delta_overall_accuracy
data_registry['Afraz2015.optogenetics_sites'] = collect_site_deltas
data_registry['Afraz2015.optogenetics_overall'] = collect_delta_overall_accuracy
stimulus_set_registry['Afraz2015'] = collect_stimuli
