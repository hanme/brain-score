import functools
import logging

import numpy as np
import xarray as xr
from brainio.assemblies import merge_data_arrays, DataAssembly, walk_coords, array_is_element
from numpy.random import RandomState
from tqdm import tqdm
from xarray import DataArray

from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.afraz2006 import mean_var
from brainscore.metrics import Score
from brainscore.metrics.difference_of_correlations import DifferenceOfCorrelations
from brainscore.metrics.difference_of_fractions import DifferenceOfFractions
from brainscore.metrics.significant_match import SignificantCorrelation, SignificantPerformanceChange, \
    is_significantly_different, NoSignificantPerformanceChange
from brainscore.model_interface import BrainModel
from brainscore.utils import fullname
#from data_packaging.azadi2023 import XXX

BIBTEX = """{azadi_image-dependence_2023,
            title = {Image-dependence of the detectability of optogenetic stimulation in macaque inferotemporal cortex},
            volume = {33},
            issn = {09609822},
            url = {https://linkinghub.elsevier.com/retrieve/pii/S0960982222019212},
            doi = {10.1016/j.cub.2022.12.021},
            abstract = {Artiﬁcial activation of neurons in early visual areas induces perception of simple visual ﬂashes.1,2 Accordingly, stimulation in high-level visual cortices is expected to induce perception of complex features.3,4 However, results from studies in human patients challenge this expectation. Stimulation rarely induces any detectable visual event, and never a complex one, in human subjects with closed eyes.2 Stimulation of the face-selective cortex in a human patient led to remarkable hallucinations only while the subject was looking at faces.5 In contrast, stimulations of color- and face-selective sites evoke notable hallucinations independent of the object being viewed.6 These anecdotal observations suggest that stimulation of high-level visual cortex can evoke perception of complex visual features, but these effects depend on the availability and content of visual input. In this study, we introduce a novel psychophysical task to systematically investigate characteristics of the perceptual events evoked by optogenetic stimulation of macaque inferior temporal (IT) cortex. We trained macaque monkeys to detect and report optogenetic impulses delivered to their IT cortices7–9 while holding ﬁxation on object images. In a series of experiments, we show that detection of cortical stimulation is highly dependent on the choice of images presented to the eyes and it is most difﬁcult when ﬁxating on a blank screen. These ﬁndings suggest that optogenetic stimulation of high-level visual cortex results in easily detectable distortions of the concurrent contents of vision.},
            language = {en},
            number = {3},
            urldate = {2023-07-29},
            journal = {Current Biology},
            author = {Azadi, Reza and Bohn, Simon and Lopez, Emily and Lafer-Sousa, Rosa and Wang, Karen and Eldridge, Mark A.G. and Afraz, Arash},
            month = feb,
            year = {2023},
            pages = {581--588.e4}
        }"""

OPTOGENETIC_PARAMETERS = {
    # "At each [of 16] injection site, 10 μl of virus was injected at a 0.5 μl/min rate, for a total volume of injection of 160 uL."
    "amount_microliter": 10,
    "rate_microliter_per_min": 0.5,
    # "We then injected AAV5-CaMKIIa-C1V1(t/t)-EYFP (nominal titer: 8x10^12 particles/ml) into the cortex"
    "virus": "AAV5-CaMKIIa-C1V1(t/t)-EYFP",
    "infectious_units_per_ml": 8E12,
    # "a 200 ms illumination impulse was delivered to IT cortex halfway through the image presentation"
    "laser_pulse_duration_ms": 200,
    # "total fiber output power ∼12 mW"
    # "illumination power of 3.6 mW [monkey Ph] and 5.4 mW [monkey Sp]"
    "fiber_output_power_mW_monkey_Ph": 3.6, 
    "fiber_output_power_mW_monkey_Sp": 5.4,
}

class _Azadi2023Optogenetics(BenchmarkBase):
    def __init__(self, metric_identifier, metric):
        self._logger = logging.getLogger(fullname(self))
        gender_stimuli, self._selectivity_stimuli = load_stimuli()
        # ...