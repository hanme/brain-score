import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics.image_level_behavior import I2n
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad

BIBTEX = """@article {Rajalingham240614,
                author = {Rajalingham, Rishi and Issa, Elias B. and Bashivan, Pouya and Kar, Kohitij and Schmidt, Kailyn and DiCarlo, James J.},
                title = {Large-scale, high-resolution comparison of the core visual object recognition behavior of humans, monkeys, and state-of-the-art deep artificial neural networks},
                elocation-id = {240614},
                year = {2018},
                doi = {10.1101/240614},
                publisher = {Cold Spring Harbor Laboratory},
                abstract = {Primates{\textemdash}including humans{\textemdash}can typically recognize objects in visual images at a glance even in the face of naturally occurring identity-preserving image transformations (e.g. changes in viewpoint). A primary neuroscience goal is to uncover neuron-level mechanistic models that quantitatively explain this behavior by predicting primate performance for each and every image. Here, we applied this stringent behavioral prediction test to the leading mechanistic models of primate vision (specifically, deep, convolutional, artificial neural networks; ANNs) by directly comparing their behavioral signatures against those of humans and rhesus macaque monkeys. Using high-throughput data collection systems for human and monkey psychophysics, we collected over one million behavioral trials for 2400 images over 276 binary object discrimination tasks. Consistent with previous work, we observed that state-of-the-art deep, feed-forward convolutional ANNs trained for visual categorization (termed DCNNIC models) accurately predicted primate patterns of object-level confusion. However, when we examined behavioral performance for individual images within each object discrimination task, we found that all tested DCNNIC models were significantly non-predictive of primate performance, and that this prediction failure was not accounted for by simple image attributes, nor rescued by simple model modifications. These results show that current DCNNIC models cannot account for the image-level behavioral patterns of primates, and that new ANN models are needed to more precisely capture the neural mechanisms underlying primate object vision. To this end, large-scale, high-resolution primate behavioral benchmarks{\textemdash}such as those obtained here{\textemdash}could serve as direct guides for discovering such models.SIGNIFICANCE STATEMENT Recently, specific feed-forward deep convolutional artificial neural networks (ANNs) models have dramatically advanced our quantitative understanding of the neural mechanisms underlying primate core object recognition. In this work, we tested the limits of those ANNs by systematically comparing the behavioral responses of these models with the behavioral responses of humans and monkeys, at the resolution of individual images. Using these high-resolution metrics, we found that all tested ANN models significantly diverged from primate behavior. Going forward, these high-resolution, large-scale primate behavioral benchmarks could serve as direct guides for discovering better ANN models of the primate visual system.},
                URL = {https://www.biorxiv.org/content/early/2018/02/12/240614},
                eprint = {https://www.biorxiv.org/content/early/2018/02/12/240614.full.pdf},
                journal = {bioRxiv}
            }"""


class DicarloRajalingham2018I2n(BenchmarkBase):
    def __init__(self):
        self._metric = I2n()
        self._fitting_stimuli = brainscore.get_stimulus_set('dicarlo.objectome.public')
        self._assembly = LazyLoad(lambda: load_assembly('private'))
        self._visual_degrees = 8
        self._number_of_trials = 2
        super(DicarloRajalingham2018I2n, self).__init__(
            identifier='dicarlo.Rajalingham2018-i2n', version=2,
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
            parent='behavior',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        fitting_stimuli = place_on_screen(self._fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        probabilities = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        score = self._metric(probabilities, self._assembly)
        score = self._metric.ceil_score(score, self.ceiling)
        return score


def load_assembly(access='private'):
    assembly = brainscore.get_assembly(f'dicarlo.Rajalingham2018.{access}')
    assembly['correct'] = assembly['choice'] == assembly['sample_obj']
    return assembly
