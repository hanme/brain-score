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
from data_packaging.azadi2023 import 