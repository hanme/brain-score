from brainscore_vision import benchmark_registry

from .benchmark import Moeller2017Experiment1

benchmark_registry['Moeller2017.experiment1-same_decrease_different_increase'] = Moeller2017Experiment1
