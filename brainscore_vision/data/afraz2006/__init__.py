from brainscore_vision import data_registry, stimulus_set_registry
from .packaging import collect_assembly, train_test_stimuli

BIBTEX = """@article{Afraz2006,
                abstract = {The inferior temporal cortex (IT) of primates is thought to be the final visual area in the ventral stream of cortical areas responsible for object recognition. Consistent with this hypothesis, single IT neurons respond selectively to highly complex visual stimuli such as faces. However, a direct causal link between the activity of face-selective neurons and face perception has not been demonstrated. In the present study of macaque monkeys, we artificially activated small clusters of IT neurons by means of electrical microstimulation while the monkeys performed a categorization task, judging whether noisy visual images belonged to 'face' or 'non-face' categories. Here we show that microstimulation of face-selective sites, but not other sites, strongly biased the monkeys' decisions towards the face category. The magnitude of the effect depended upon the degree of face selectivity of the stimulation site, the size of the stimulated cluster of face-selective neurons, and the exact timing of microstimulation. Our results establish a causal relationship between the activity of face-selective neurons and face perception.},
                author = {Afraz, Seyed Reza and Kiani, Roozbeh and Esteky, Hossein},
                doi = {10.1038/nature04982},
                isbn = {1476-4687 (Electronic) 0028-0836 (Linking)},
                issn = {14764687},
                journal = {Nature},
                month = {aug},
                number = {7103},
                pages = {692--695},
                pmid = {16878143},
                publisher = {Nature Publishing Group},
                title = {{Microstimulation of inferotemporal cortex influences face categorization}},
                url = {http://www.nature.com/articles/nature04982},
                volume = {442},
                year = {2006}
                }"""

data_registry['Afraz2006'] = collect_assembly

train_stimuli, test_stimuli = train_test_stimuli()
stimulus_set_registry['Afraz2006.train'] = train_stimuli
stimulus_set_registry['Afraz2006.test'] = test_stimuli
