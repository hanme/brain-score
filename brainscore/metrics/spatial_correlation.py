import warnings

import numpy as np
import pandas as pd
import xarray as xr
from pandas import DataFrame
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr
from xarray import DataArray

from brainio.assemblies import walk_coords
from brainscore.metrics import Score, Metric
from brainscore.metrics.transformations import CrossValidation


def _aggregate(scores):
    """
    Aggregates list of values into Score object
    :param scores: list of values assumed to be scores
    :return: Score object | where score['center'] = mean(scores) and score['error'] = std(scores)
    """
    center = np.median(scores)
    error = np.median(np.absolute(scores - np.median(scores)))  # MAD
    aggregate_score = Score([center, error], coords={'aggregation': ['center', 'error']}, dims=('aggregation',))
    aggregate_score.attrs[Score.RAW_VALUES_KEY] = scores

    return aggregate_score


class SpatialCorrelationSimilarity(Metric):
    """
    Computes the similarity of two given distributions using a given similarity_function. The similarity_function is
    applied to each bin which in turn are created based on a given bin size and the independent variable of the
    distributions
    """

    def __init__(self, similarity_function, bin_size_mm):
        """
        :param similarity_function: similarity_function to be applied to each bin
        :param bin_size_mm: size per bin in mm | one fixed size, utilize Score.RAW_VALUES_KEY to change weighting
        """
        self.similarity_function = similarity_function
        self.bin_size = bin_size_mm

    def __call__(self, candidate_statistic, target_statistic):
        """
        :param candidate_statistic: list of 2 lists, [0] distances -> binning over this, [1] correlation per distance value
        :param target_statistic: list of 2 lists, [0] distances -> binning over this, [1] correlation per distance value
        """
        self.target_statistic = target_statistic
        self.candidate_statistic = candidate_statistic
        self._bin_min = np.min(self.target_statistic.distances)
        self._bin_max = np.max(self.target_statistic.distances)

        bin_scores = []
        for in_bin_t, in_bin_c, enough_data in self._bin_masks():
            if enough_data:
                bin_scores.append(self.similarity_function(self.target_statistic.values[in_bin_t],
                                                           self.candidate_statistic.values[in_bin_c]))

        return _aggregate(bin_scores)

    def _bin_masks(self):
        """
        Generator: Yields masks indexing which elements are within each bin.
        :yield: mask(target, current_bin), mask(candidate, current_bin), enough data in the bins to do further computations
        """
        for lower_bound_mm in np.linspace(self._bin_min, self._bin_max,
                                          int(self._bin_max * (1 / self.bin_size) + 1) * 2):
            t = np.where(np.logical_and(self.target_statistic.distances >= lower_bound_mm,
                                        self.target_statistic.distances < lower_bound_mm + self.bin_size))[0]
            c = np.where(np.logical_and(self.candidate_statistic.distances >= lower_bound_mm,
                                        self.candidate_statistic.distances < lower_bound_mm + self.bin_size))[0]
            enough_data = t.size > 0 and c.size > 0  # random threshold

            yield t, c, enough_data


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
        score.attrs['candidate_statistic'] = candidate_statistic
        score.attrs['candidate_assembly'] = candidate_assembly
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
        :param values: list of data values
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
