from string import ascii_lowercase as alphabet

import numpy as np
from brainio.assemblies import NeuroidAssembly, DataAssembly
from pytest import approx

from brainscore_vision.metrics.ceiling import NoCeiling, InternalConsistency, SplitHalfConsistency


class TestNoCeiling:
    def test(self):
        ceiling = NoCeiling()
        ceiling_score = ceiling()
        assert ceiling_score == 1


class TestInternalConsistency:
    def test_dummy_data(self):
        data = NeuroidAssembly(np.tile(np.arange(10)[:, np.newaxis], [5, 10]),
                               coords={'stimulus_id': ('presentation', np.tile(list(alphabet)[:10], 5)),
                                       'image_meta': ('presentation', np.tile(list(alphabet)[:10], 5)),
                                       'repetition': ('presentation', np.repeat(np.arange(5), 10)),
                                       'neuroid_id': ('neuroid', np.arange(10)),
                                       'neuroid_meta': ('neuroid', np.arange(10))},
                               dims=['presentation', 'neuroid'])
        ceiler = InternalConsistency()
        ceiling = ceiler(data)
        assert ceiling.sel(aggregation='center') == 1


class TestSplitHalfConsistency:
    def test(self):
        data = NeuroidAssembly(np.tile(np.arange(10)[:, np.newaxis], [5, 10]),
                               coords={'stimulus_id': ('presentation', np.tile(list(alphabet)[:10], 5)),
                                       'image_meta': ('presentation', np.tile(list(alphabet)[:10], 5)),
                                       'repetition': ('presentation', np.tile(np.arange(5), 10)),
                                       'neuroid_id': ('neuroid', np.arange(10)),
                                       'neuroid_meta': ('neuroid', np.arange(10))},
                               dims=['presentation', 'neuroid'])
        ceiler = SplitHalfConsistency()
        ceiling = ceiler(data, data)
        assert all(ceiling == DataAssembly([approx(1)] * 10,
                                           coords={'neuroid_id': ('neuroid', np.arange(10)),
                                                   'neuroid_meta': ('neuroid', np.arange(10))},
                                           dims=['neuroid']))
