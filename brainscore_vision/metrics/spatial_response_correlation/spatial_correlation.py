import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

import brainscore_vision
from brainscore_vision.benchmark_helpers.neural_common import average_repetition
from brainscore_vision.metric_helpers.transformations import CrossValidation
from pandas import DataFrame
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr
from xarray import DataArray

from brainio.assemblies import walk_coords, NeuroidAssembly, merge_data_arrays
from brainscore_core import Score, Metric


def inv_ks_similarity(p, q):
    """
    Inverted ks similarity -> resulting in a score within [0,1], 1 being a perfect match
    """
    import scipy.stats
    return 1 - scipy.stats.ks_2samp(p, q)[0]


class SpatialCorrelationSimilarity(Metric):
    """
    Computes the similarity of two given distributions using a given similarity_function.
    """

    def __init__(self, similarity_function,
                 bin_size_mm: float, num_bootstrap_samples: int, num_sample_arrays: int):
        """
        :param similarity_function: similarity_function to be applied to each bin
            which in turn are created based on a given bin size and the independent variable of the distributions.
            E.g. `inv_ks_similarity`
        :param bin_size_mm: size per bin in mm
        :param num_bootstrap_samples: how many electrode pairs to sample from the data
        :param num_sample_arrays: number of simulated Utah arrays sampled from candidate model tissue
        """
        self.similarity_function = similarity_function
        self.bin_size = bin_size_mm
        self.num_bootstrap_samples = num_bootstrap_samples
        self.num_sample_arrs = num_sample_arrays

    def __call__(self, candidate_assembly: NeuroidAssembly, target_assembly: NeuroidAssembly) -> Score:
        """
        :param candidate_assembly: neural recordings from candidate model
        :param target_assembly: neural recordings from target system.
            Expected to include repetitions to compute electrode ceilings
        :return: a Score representing how similar the two assemblies are with respect to their spatial
            response-correlation.
        """
        # characterize response-correlation for each assembly
        array_size_mm = (np.ptp(target_assembly['tissue_x'].values),
                         np.ptp(target_assembly['tissue_y'].values))
        candidate_statistic = self.sample_global_tissue_statistic(candidate_assembly, array_size_mm=array_size_mm)
        target_statistic = self.compute_global_tissue_statistic_target(target_assembly)
        return self.compare_statistics(candidate_statistic, target_statistic)

    def compare_statistics(self, candidate_statistic: DataArray, target_statistic: DataArray) -> Score:
        # score all bins
        self._bin_min = np.min(target_statistic.distances)
        self._bin_max = np.max(target_statistic.distances)
        bin_scores = []
        for bin_number, (target_mask, candidate_mask) in enumerate(
                self._bin_masks(candidate_statistic, target_statistic)):
            enough_data = target_mask.size > 0 and candidate_mask.size > 0  # both non-zero
            if not enough_data:  # ignore bins with insufficient number of data
                continue
            similarity = self.similarity_function(target_statistic.values[target_mask],
                                                  candidate_statistic.values[candidate_mask])
            similarity = Score([similarity], coords={'bin': [bin_number]}, dims=['bin'])
            bin_scores.append(similarity)
        # aggregate
        bin_scores = merge_data_arrays(bin_scores)
        score = self._aggregate_scores(bin_scores)
        score.attrs['candidate_statistic'] = candidate_statistic
        score.attrs['target_statistic'] = target_statistic
        return score

    def compute_global_tissue_statistic_target(self, assembly: NeuroidAssembly) -> DataArray:
        """
        :return: DataArray with values = correlations; coordinates: distances, source, array
        """
        consistency = brainscore_vision.load_ceiling('internal_consistency')
        neuroid_reliability = consistency(assembly.transpose('presentation', 'neuroid'))

        averaged_assembly = average_repetition(assembly)
        target_statistic_list = []
        for animal in sorted(list(set(averaged_assembly.neuroid.animal.data))):
            for electrode_array in sorted(list(set(averaged_assembly.neuroid.arr.data))):
                sub_assembly = averaged_assembly.sel(animal=animal, arr=electrode_array)
                bootstrap_samples_sub_assembly = int(
                    self.num_bootstrap_samples * (sub_assembly.neuroid.size / averaged_assembly.neuroid.size))

                distances, correlations = self.sample_response_corr_vs_dist(sub_assembly,
                                                                            bootstrap_samples_sub_assembly,
                                                                            neuroid_reliability)
                sub_assembly_statistic = self.to_xarray(correlations, distances, source=animal,
                                                        electrode_array=electrode_array)
                target_statistic_list.append(sub_assembly_statistic)

        target_statistic = xr.concat(target_statistic_list, dim='meta')
        return target_statistic

    def sample_global_tissue_statistic(
            self, candidate_assembly, array_size_mm: Tuple[np.ndarray, np.ndarray]) -> DataArray:
        """
        Simulates placement of multiple arrays in tissue and computes repsonse correlation as a function of distance on
        each of them
        :param array_size_mm: physical size of Utah array in mm
        :param candidate_assembly: NeuroidAssembly
        :return: DataArray with values = correlations; coordinates: distances, source, array
        """
        candidate_statistic_list = []
        bootstrap_samples_per_array = int(self.num_bootstrap_samples / self.num_sample_arrs)
        array_locations = self.sample_array_locations(candidate_assembly.neuroid, array_size_mm=array_size_mm)
        for i, window in enumerate(array_locations):
            distances, correlations = self.sample_response_corr_vs_dist(candidate_assembly[window],
                                                                        bootstrap_samples_per_array)

            array_statistic = self.to_xarray(correlations, distances, electrode_array=str(i))
            candidate_statistic_list.append(array_statistic)

        candidate_statistic = xr.concat(candidate_statistic_list, dim='meta')
        return candidate_statistic

    def sample_array_locations(self, neuroid, array_size_mm: Tuple[np.ndarray, np.ndarray], seed=0):
        """
        Generator: Sample Utah array-like portions from artificial model tissue and generate masks
        :param neuroid: NeuroidAssembly.neuroid, has to contain tissue_x, tissue_y coords
        :param array_size_mm: physical size of Utah array in mm
        :param seed: random seed
        :return: list of masks in neuroid dimension of assembly, usage: assembly[mask] -> neuroids within one array
        """
        bound_max_x, bound_max_y = np.max([neuroid.tissue_x.data, neuroid.tissue_y.data], axis=1) - array_size_mm
        rng = np.random.default_rng(seed=seed)

        lower_corner = np.column_stack((rng.choice(neuroid.tissue_x.data[neuroid.tissue_x.data <= bound_max_x],
                                                   size=self.num_sample_arrs),
                                        rng.choice(neuroid.tissue_y.data[neuroid.tissue_y.data <= bound_max_y],
                                                   size=self.num_sample_arrs)))
        upper_corner = lower_corner + array_size_mm

        # create index masks of neuroids within sample windows
        for i in range(self.num_sample_arrs):
            yield np.logical_and.reduce([neuroid.tissue_x.data <= upper_corner[i, 0],
                                         neuroid.tissue_x.data >= lower_corner[i, 0],
                                         neuroid.tissue_y.data <= upper_corner[i, 1],
                                         neuroid.tissue_y.data >= lower_corner[i, 1]])

    def sample_response_corr_vs_dist(self, assembly, num_samples, neuroid_reliability=None, seed=0):
        """
        1. Samples random pairs from the assembly
        2. Computes distances for all pairs
        3. Computes the response correlation between items of each pair
        (4. Ceils the response correlations by ceiling each neuroid | if neuroid_reliability not None)
        :param assembly: NeuroidAssembly without stimulus repetitions
        :param num_samples: how many random pair you want to be sampled out of the data
        :param neuroid_reliability: if not None: expecting Score object containing reliability estimates of all neuroids
        :param seed: random seed
        :return: [distance, pairwise_correlation_of_neuroids], pairwise correlations can be ceiled
        """
        rng = np.random.default_rng(seed=seed)
        neuroid_pairs = rng.integers(0, assembly.shape[0], (2, num_samples))

        pairwise_distances_all = self.pairwise_distances(assembly)
        pairwise_distance_samples = pairwise_distances_all[(*neuroid_pairs,)]

        response_samples = assembly.data[neuroid_pairs]
        response_correlation_samples = self.corrcoef_rowwise(*response_samples)

        if neuroid_reliability is not None:
            pairwise_neuroid_reliability_all = self.create_pairwise_neuroid_reliability_mat(neuroid_reliability)
            pairwise_neuroid_reliability_samples = pairwise_neuroid_reliability_all[(*neuroid_pairs,)]

            response_correlation_samples = response_correlation_samples / pairwise_neuroid_reliability_samples

        # properly removing nan values
        pairwise_distance_samples = pairwise_distance_samples[~np.isnan(response_correlation_samples)]
        response_correlation_samples = response_correlation_samples[~np.isnan(response_correlation_samples)]

        return np.vstack((pairwise_distance_samples, response_correlation_samples))

    def corrcoef_rowwise(self, a, b):
        # https://stackoverflow.com/questions/41700840/correlation-of-2-time-dependent-multidimensional-signals-signal-vectors
        a_ma = a - a.mean(1)[:, None]
        b_mb = b - b.mean(1)[:, None]
        ssa = np.einsum('ij,ij->i', a_ma, a_ma)  # var A
        ssb = np.einsum('ij,ij->i', b_mb, b_mb)  # var B
        return np.einsum('ij,ij->i', a_ma, b_mb) / np.sqrt(ssa * ssb)  # cov/sqrt(varA*varB)

    def pairwise_distances(self, assembly):
        """
        Convenience function creating a simple lookup table for pairwise distances
        :param assembly: NeuroidAssembly
        :return: square matrix where each entry is the distance between the neuroids at the corresponding indices
        """
        locations = np.stack([assembly.neuroid.tissue_x.data, assembly.neuroid.tissue_y.data]).T

        return squareform(pdist(locations, metric='euclidean'))

    def create_pairwise_neuroid_reliability_mat(self, neuroid_reliability):
        """
        Convenience function creating a simple lookup table for combined reliabilities of neuroid pairs
        :param neuroid_reliability: expects Score object where neuroid_reliability.raw holds [cross validation subset,
            reliability per neuroid]
        :return: square matrix where each entry_ij = sqrt(reliability_i * reliability_j)
        """
        reliability_per_neuroid = np.mean(neuroid_reliability.raw.data, axis=0)
        c_mat = np.zeros((reliability_per_neuroid.size, reliability_per_neuroid.size))
        for i, ci in enumerate(reliability_per_neuroid):
            for j, cj in enumerate(reliability_per_neuroid):
                c_mat[i, j] = np.sqrt(ci * cj)

        return c_mat

    def to_xarray(self, correlations, distances, source='model', electrode_array=None):
        """
        :param correlations: list of data values
        :param distances: list of distance values, each distance value has to correspond to one data value
        :param source: name of monkey
        :param electrode_array: name of recording array
        """
        xarray_statistic = DataArray(
            data=correlations,
            dims=["meta"],
            coords={
                'meta': pd.MultiIndex.from_product([distances, [source], [electrode_array]],
                                                   names=('distances', 'source', 'array'))
            }
        )

        return xarray_statistic

    def _bin_masks(self, candidate_statistic: DataArray, target_statistic: DataArray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generator: Yields masks indexing which elements are within each bin.

        :yield: a triplet where the two elements are masks for a bin over the target and candidate respectively
        """
        bin_step = int(self._bin_max * (1 / self.bin_size) + 1) * 2
        for lower_bound_mm in np.linspace(self._bin_min, self._bin_max, bin_step):
            target_mask = np.where(np.logical_and(target_statistic.distances >= lower_bound_mm,
                                                  target_statistic.distances < lower_bound_mm + self.bin_size))[0]
            candidate_mask = np.where(np.logical_and(candidate_statistic.distances >= lower_bound_mm,
                                                     candidate_statistic.distances < lower_bound_mm + self.bin_size))[0]
            yield target_mask, candidate_mask

    def _aggregate_scores(self, scores: Score, over: str = 'bin') -> Score:
        """
        Aggregates scores into an aggregate Score where `center = mean(scores)` and `error = mad(scores)`
        :param scores: scores over bins
        """
        center = scores.median(dim=over)
        error = abs((scores - scores.median(dim=over))).median(dim=over)  # mean absolute deviation
        aggregate_score = Score([center, error], coords={'aggregation': ['center', 'error']}, dims=('aggregation',))
        aggregate_score.attrs[Score.RAW_VALUES_KEY] = scores

        return aggregate_score


class SpatialCharacterizationMetric:
    def __init__(self, similarity_metric, characterization):
        self._characterization = characterization
        self._similarity_metric = similarity_metric

    def __call__(self, behaviors, target):
        dprime_assembly_all = self._characterization.characterize(behaviors)
        dprime_assembly = self._characterization.subselect_tasks(dprime_assembly_all, target)
        candidate_assembly = dprime_assembly.transpose('injected', 'site', 'task')  # match target assembly shape
        candidate_statistic = self.compute_response_deficit_distance_candidate(candidate_assembly)
        target = target.mean('bootstrap')
        target_statistic = self.compute_response_deficit_distance_target(target)

        score = self._similarity_metric(candidate_statistic, target_statistic)
        score.attrs['candidate_behaviors'] = behaviors
        # score.attrs['candidate_statistic'] = candidate_statistic  FIXME temporary comment
        # score.attrs['candidate_assembly'] = candidate_assembly  FIXME
        return score

    def compute_response_deficit_distance_target(self, dprime_assembly):
        statistics_list = []
        for monkey in set(dprime_assembly.monkey.data):
            sub_assembly = dprime_assembly.sel(monkey=monkey)
            distances, correlations = self._compute_response_deficit_distance(sub_assembly)
            mask = np.triu_indices(sub_assembly.site.size)

            stats = self.to_xarray(correlations[mask], distances[mask], source=monkey)
            statistics_list.append(stats)

        return xr.concat(statistics_list, dim='meta')

    def compute_response_deficit_distance_candidate(self, dprime_assembly):
        distances, correlations = self._compute_response_deficit_distance(dprime_assembly)
        mask = np.triu_indices(dprime_assembly.site.size)
        spatial_deficits = self.to_xarray(correlations[mask], distances[mask])
        spatial_deficits = spatial_deficits.dropna('meta')  # drop cross-hemisphere distances
        return spatial_deficits

    def _compute_response_deficit_distance(self, dprime_assembly):
        """
        :param dprime_assembly: assembly of behavioral performance
        :return: square matrices with correlation and distance values; each matrix elem == value between site_i, site_j
        """
        distances = self.pairwise_distances(dprime_assembly)

        behavioral_differences = self._characterization.compute_differences(dprime_assembly)
        # dealing with nan values while correlating; not np.ma.corrcoef: https://github.com/numpy/numpy/issues/15601
        correlations = DataFrame(behavioral_differences.data).T.corr().values
        correlations[np.isnan(distances)] = None

        return distances, correlations

    @staticmethod
    def to_xarray(correlations, distances, source='model', array=None):
        """
        :param correlations: list of data values
        :param distances: list of distance values, each distance value has to correspond to one data value
        :param source: name of monkey
        :param array: name of recording array
        """
        xarray_statistic = DataArray(
            data=correlations,
            dims=["meta"],
            coords={
                'meta': pd.MultiIndex.from_product([distances, [source], [array]],
                                                   names=('distance', 'source', 'array'))
            }
        )
        return xarray_statistic

    @staticmethod
    def pairwise_distances(dprime_assembly):
        locations = np.stack([dprime_assembly.site.site_x.data,
                              dprime_assembly.site.site_y.data,
                              dprime_assembly.site.site_z.data]).T
        distances = pdist(locations, metric='euclidean')
        distances = squareform(distances)
        # ignore distances from different hemispheres
        # TODO: at some point, we should actually include these but we need to fix xyz locations in the model
        if hasattr(dprime_assembly, 'hemisphere') and dprime_assembly['hemisphere'].values.size > 1:
            same_hemispheres = np.repeat([dprime_assembly['hemisphere'].values], len(distances), axis=0) == \
                               np.repeat(dprime_assembly['hemisphere'].values, len(distances)) \
                                   .reshape(len(distances), len(distances))
            distances[~same_hemispheres] = None
        return distances

    @property
    def ceiling(self):
        split1, split2 = self._target_assembly.sel(split=0), self._target_assembly.sel(split=1)
        split1_diffs = split1.sel(silenced=False) - split1.sel(silenced=True)
        split2_diffs = split2.sel(silenced=False) - split2.sel(silenced=True)
        split_correlation, p = pearsonr(split1_diffs.values.flatten(), split2_diffs.values.flatten())
        return Score([split_correlation], coords={'aggregation': ['center']}, dims=['aggregation'])

    @classmethod
    def apply_site(cls, source_assembly, site_target_assembly):
        site_target_assembly = site_target_assembly.squeeze('site')
        np.testing.assert_array_equal(source_assembly.sortby('task_number')['task_left'].values,
                                      site_target_assembly.sortby('task_number')['task_left'].values)
        np.testing.assert_array_equal(source_assembly.sortby('task_number')['task_right'].values,
                                      site_target_assembly.sortby('task_number')['task_right'].values)

        # filter non-nan task measurements from target
        nonnan_tasks = site_target_assembly['task'][~site_target_assembly.isnull()]
        if len(nonnan_tasks) < len(site_target_assembly):
            warnings.warn(f"Ignoring tasks {site_target_assembly['task'][~site_target_assembly.isnull()].values}")
        site_target_assembly = site_target_assembly.sel(task=nonnan_tasks)
        source_assembly = source_assembly.sel(task=nonnan_tasks.values)

        # try to predict from model
        task_split = CrossValidation(split_coord='task_number', stratification_coord=None,
                                     kfold=True, splits=len(site_target_assembly['task']))
        task_scores = task_split(source_assembly, site_target_assembly, apply=cls.apply_task)
        task_scores = task_scores.raw
        correlation, p = pearsonr(task_scores.sel(type='source'), task_scores.sel(type='target'))
        score = Score([correlation, p], coords={'statistic': ['r', 'p']}, dims=['statistic'])
        score.attrs['predictions'] = task_scores.sel(type='source')
        score.attrs['target'] = task_scores.sel(type='target')
        return score

    @staticmethod
    def apply_task(source_train, target_train, source_test, target_test):
        """
        finds the best-matching site in the source train assembly to predict the task effects in the test target.
        :param source_train: source assembly for mapping with t tasks and n sites
        :param target_train: target assembly for mapping with t tasks
        :param source_test: source assembly for testing with 1 task and n sites
        :param target_test: target assembly for testing with 1 task
        :return: a pair
        """
        # deal with xarray bug
        source_train, source_test = deal_with_xarray_bug(source_train), deal_with_xarray_bug(source_test)
        # map: find site in assembly1 that best matches mapping tasks
        correlations = {}
        for site in source_train['site'].values:
            source_site = source_train.sel(site=site)
            np.testing.assert_array_equal(source_site['task'].values, target_train['task'].values)
            correlation, p = pearsonr(source_site, target_train)
            correlations[site] = correlation
        best_site = [site for site, correlation in correlations.items() if correlation == max(correlations.values())]
        best_site = best_site[0]  # choose first one if there are multiple
        # test: predictivity of held-out task.
        # We can only collect the single prediction here and then correlate in outside loop
        source_test = source_test.sel(site=best_site)
        np.testing.assert_array_equal(source_test['task'].values, target_test['task'].values)
        pair = type(target_test)([source_test.values[0], target_test.values[0]],
                                 coords={  # 'task': source_test['task'].values,
                                     'type': ['source', 'target']},
                                 dims=['type'])  # , 'task'
        return pair


def deal_with_xarray_bug(assembly):
    if hasattr(assembly, 'site_level_0'):
        return type(assembly)(assembly.values, coords={
            coord: (dim, values) for coord, dim, values in walk_coords(assembly) if coord != 'site_level_0'},
                              dims=assembly.dims)
