import itertools
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from brainio.assemblies import merge_data_arrays, walk_coords
from brainscore.metrics import Metric
from brainscore.metrics.regression import ridge_regression, pearsonr_correlation
from brainscore.utils import fullname


class DeficitPrediction(Metric):
    def __init__(self):
        super(DeficitPrediction, self).__init__()
        self._correlation = pearsonr_correlation(xarray_kwargs=dict(
            correlation_coord='task_number', group_coord='site_iteration', drop_target_nans=True))
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, assembly1, assembly2):
        """
        :param assembly1: a tuple with the first element representing the control behavior in the format of
            `presentation: p, choice: c` and the second element representing inactivations behaviors in
            `presentation: p, choice: c, site: n`
        :param assembly2: a processed assembly in the format of `injected :2, task: c * (c-1), site: m`
        :return: a Score
        """

        # compare per site, then median over sites, to avoid biases alone driving correlation
        prediction_pairs = self.cross_validate(assembly1, assembly2)
        source = prediction_pairs.sel(type='source')
        target = prediction_pairs.sel(type='target')
        score = self._correlation(source, target)
        score.attrs['predictions'] = source
        score.attrs['target'] = target
        return score

    def cross_validate(self, assembly1_differences, assembly2_differences):
        raise NotImplementedError()

    def fit_predict(self, source_train, target_train, source_test, target_test):
        # map: regress from source to target
        regression = ridge_regression(
            xarray_kwargs=dict(expected_dims=('task', 'site'),
                               neuroid_dim='site',
                               neuroid_coord='site_iteration',
                               stimulus_coord='task'))
        regression.fit(source_train, target_train)
        # test: predictivity of held-out task
        # We can only collect the single prediction here and then correlate in outside loop
        prediction_test = regression.predict(source_test)
        prediction_test = prediction_test.transpose(*target_test.dims)
        np.testing.assert_array_equal(prediction_test['task'].values, prediction_test['task'].values)
        np.testing.assert_array_equal(prediction_test.shape, target_test.shape)
        pair = type(target_test)([prediction_test, target_test],
                                 coords={**{'type': ['source', 'target']},
                                         **{coord: (dims, values) for coord, dims, values in
                                            walk_coords(target_test)}},
                                 dims=('type',) + target_test.dims)
        return pair


class DeficitPredictionTask(DeficitPrediction):
    def cross_validate(self, assembly1_differences, assembly2_differences):
        sites = assembly2_differences['site_iteration'].values
        tasks = assembly2_differences['task_number'].values
        prediction_pairs = []
        for target_test_site, target_test_task in tqdm(
                itertools.product(sites, tasks), desc='site+task kfold', total=len(sites) * len(tasks)):
            # test assembly is 1 task, 1 site
            target_test = assembly2_differences[{
                'site': [site == target_test_site for site in assembly2_differences['site_iteration'].values],
                'task': [task == target_test_task for task in assembly2_differences['task_number'].values]}]
            if len(target_test) < 1:
                continue  # not all tasks were run on all sites
            # train are the other tasks on the same site
            target_train = assembly2_differences[{
                'site': [site == target_test_site for site in assembly2_differences['site_iteration'].values],
                'task': [task != target_test_task for task in assembly2_differences['task_number'].values]}]
            # source test assembly is same 1 task, all sites
            source_test = assembly1_differences[{
                'task': [task == target_test_task for task in assembly1_differences['task_number'].values]}]
            # source train assembly are other tasks, all sites
            source_train = assembly1_differences[{
                'task': [task != target_test_task for task in assembly1_differences['task_number'].values]}]

            # filter non-nan task measurements from target
            nonnan_tasks = target_train['task'][~target_train.squeeze('site').isnull()].values
            target_train = target_train.sel(task=nonnan_tasks)
            source_train = source_train.sel(task=nonnan_tasks)

            pair = self.fit_predict(source_train, target_train, source_test, target_test)
            prediction_pairs.append(pair)
        prediction_pairs = merge_data_arrays(prediction_pairs)
        return prediction_pairs


class DeficitPredictionObject(DeficitPrediction):
    # the setup here is to hold out an entire object. For instance, if 'Bear' is in a task in test, then no task in
    # train may include Bear. However, it can still include other objects from test, for instance test tasks could be
    # Bear-Elephant and Bear-Chair, then train tasks could still include Elephant-Dog and Plane-Chair
    def cross_validate(self, assembly1_differences, assembly2_differences):
        sites = assembly2_differences['site_iteration'].values
        objects = np.concatenate((assembly2_differences['task_left'], assembly2_differences['task_right']))
        objects = list(sorted(set(objects)))
        prediction_pairs = []
        visited_site_tasks = defaultdict(list)
        for target_test_site, target_test_object in tqdm(
                itertools.product(sites, objects), desc='site+object kfold', total=len(sites) * len(objects)):

            # test assembly are tasks with 1 object left or right, 1 site
            test_tasks = [(task_number, task_left, task_right) for task_number, task_left, task_right in zip(
                *[assembly2_differences[coord].values for coord in ['task_number', 'task_left', 'task_right']])
                          if (task_left == target_test_object or task_right == target_test_object)]
            # only evaluate each task once per site (cannot merge otherwise)
            unvisited_test_tasks = {task_number: (task_left, task_right) for task_number, task_left, task_right
                                    in test_tasks if task_number not in visited_site_tasks[target_test_site]}
            visited_site_tasks[target_test_site] += unvisited_test_tasks
            target_test = assembly2_differences[{
                'site': [site == target_test_site for site in assembly2_differences['site_iteration'].values],
                'task': [task_number in unvisited_test_tasks
                         for task_number in assembly2_differences['task_number'].values]}]
            # source test assembly are same unvisited tasks from 1 object, all sites
            source_test = assembly1_differences[{
                'task': [task in unvisited_test_tasks for task in assembly1_differences['task_number'].values]}]
            nonnan_test = target_test['task'][~target_test.squeeze('site').isnull()].values
            # filter non-nan task measurements from target
            target_test, source_test = target_test.sel(task=nonnan_test), source_test.sel(task=nonnan_test)
            if np.prod(target_test.shape) < 1:
                continue  # have already run tasks on previous objects
            # train are tasks from other objects on the same site
            target_train = assembly2_differences[{
                'site': [site == target_test_site for site in assembly2_differences['site_iteration'].values],
                'task': [task not in test_tasks for task in assembly2_differences['task_number'].values]}]
            # source train assembly are tasks from other objects, all sites
            source_train = assembly1_differences[{
                'task': [task not in test_tasks for task in assembly1_differences['task_number'].values]}]
            # filter non-nan task measurements from target
            nonnan_train = target_train['task'][~target_train.squeeze('site').isnull()].values
            target_train, source_train = target_train.sel(task=nonnan_train), source_train.sel(task=nonnan_train)

            pair = self.fit_predict(source_train, target_train, source_test, target_test)
            prediction_pairs.append(pair)
        prediction_pairs = merge_data_arrays(prediction_pairs)
        return prediction_pairs
