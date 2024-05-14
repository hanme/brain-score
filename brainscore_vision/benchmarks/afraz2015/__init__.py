from brainscore_vision import benchmark_registry

from .benchmark import (
    Afraz2015OptogeneticContraDeltaAccuracySignificant,
    Afraz2015OptogeneticIpsiDeltaAccuracyInsignificant,
    Afraz2015OptogeneticDeltaAccuracyCorrelated,
    Afraz2015MuscimolDeltaAccuracySignificant)

benchmark_registry[
    'Afraz2015.optogenetics-contra_accuracy_significant'] = Afraz2015OptogeneticContraDeltaAccuracySignificant
benchmark_registry[
    'Afraz2015.optogenetics-ipsi_accuracy_insignificant'] = Afraz2015OptogeneticIpsiDeltaAccuracyInsignificant
benchmark_registry[
    'Afraz2015.optogenetics-delta_accuracy_correlated'] = Afraz2015OptogeneticDeltaAccuracyCorrelated
benchmark_registry[
    'Afraz2015.muscimol-delta_accuracy_significant'] = Afraz2015MuscimolDeltaAccuracySignificant
