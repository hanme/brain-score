from brainscore_vision import benchmark_registry

from .benchmark import (
    Rajalingham2019GlobalDeficitsSignificant,
    Rajalingham2019LateralDeficitDifference,
    Rajalingham2019SpatialCorrelationSignificant)

benchmark_registry['Rajalingham2019-global_deficits_significant'] = Rajalingham2019GlobalDeficitsSignificant
benchmark_registry['Rajalingham2019-lateral_deficit_difference'] = Rajalingham2019LateralDeficitDifference
benchmark_registry['Rajalingham2019-spatial_correlation_significant'] = Rajalingham2019SpatialCorrelationSignificant
