import copy
import functools
import itertools
import logging
from pathlib import Path

import numpy as np
from numpy.random import RandomState
from scipy.optimize import minimize
from tqdm import tqdm

import brainscore
from brainio.assemblies import merge_data_arrays, walk_coords, DataAssembly, array_is_element
from brainscore.benchmarks import BenchmarkBase
from brainscore.metrics import Score
from brainscore.metrics.behavior_differences import DeltaPredictionTask, DeltaPredictionObject
from brainscore.metrics.difference_of_correlations import DifferenceOfCorrelations
from brainscore.metrics.image_level_behavior import _o2
from brainscore.metrics.inter_individual_stats_ceiling import InterIndividualStatisticsCeiling
from brainscore.metrics.regression import pearsonr_correlation
from brainscore.metrics.significant_match import SignificantCorrelation, SignificantPerformanceChange
from brainscore.metrics.spatial_correlation import SpatialCorrelationSimilarity, SpatialCharacterizationMetric
from brainscore.model_interface import BrainModel
from brainscore.utils import fullname
from data_packaging.rajalingham2019 import collect_assembly

TASK_LOOKUP = {
    'dog': 'Dog',
    # 'face0': '',
    # 'table4': '',
    'bear': 'Bear',
    # 'apple': '',
    'elephant': 'Elephant',
    'airplane3': 'Plane',
    # 'turtle': '',
    # 'car_alfa': '',
    'chair0': 'Chair'
}

BIBTEX = """@article{RAJALINGHAM2019493,
                title = {Reversible Inactivation of Different Millimeter-Scale Regions of Primate IT Results in Different Patterns of Core Object Recognition Deficits},
                journal = {Neuron},
                volume = {102},
                number = {2},
                pages = {493-505.e5},
                year = {2019},
                issn = {0896-6273},
                doi = {https://doi.org/10.1016/j.neuron.2019.02.001},
                url = {https://www.sciencedirect.com/science/article/pii/S0896627319301102},
                author = {Rishi Rajalingham and James J. DiCarlo},
                keywords = {object recognition, neural perturbation, inactivation, vision, primate, inferior temporal cortex},
                abstract = {Extensive research suggests that the inferior temporal (IT) population supports visual object recognition behavior. However, causal evidence for this hypothesis has been equivocal, particularly beyond the specific case of face-selective subregions of IT. Here, we directly tested this hypothesis by pharmacologically inactivating individual, millimeter-scale subregions of IT while monkeys performed several core object recognition subtasks, interleaved trial-by trial. First, we observed that IT inactivation resulted in reliable contralateral-biased subtask-selective behavioral deficits. Moreover, inactivating different IT subregions resulted in different patterns of subtask deficits, predicted by each subregion’s neuronal object discriminability. Finally, the similarity between different inactivation effects was tightly related to the anatomical distance between corresponding inactivation sites. Taken together, these results provide direct evidence that the IT cortex causally supports general core object recognition and that the underlying IT coding dimensions are topographically organized.}
                }"""

# "Each inactivation session began with a single focal microinjection of 1ml of muscimol
# (5mg/mL, Sigma Aldrich) at a slow rate (100nl/min) via a 30-gauge stainless-steel cannula at
# the targeted site in ventral IT."
MUSCIMOL_PARAMETERS = {
    'amount_microliter': 1
}


class _Rajalingham2019(BenchmarkBase):
    def __init__(self, identifier, metric, characterize=None, ceiling_func=None,
                 num_sites_per_hemisphere=10, num_experiment_bootstraps=10):
        self._target_assembly = collect_assembly()
        self._training_stimuli = brainscore.get_stimulus_set('dicarlo.hvm')
        self._training_stimuli['image_label'] = self._training_stimuli['object_name']
        # use only those images where it's the same object (label)
        self._training_stimuli = self._training_stimuli[self._training_stimuli['object_name'].isin(
            self._target_assembly.stimulus_set['object_name'])]
        self._test_stimuli = self._target_assembly.stimulus_set

        self._num_sites_per_hemisphere = num_sites_per_hemisphere
        self._num_experiment_bootstraps = num_experiment_bootstraps
        self._characterize = characterize or (lambda a1, a2: (a1, a2))
        self._metric = metric
        self._logger = logging.getLogger(fullname(self))
        super(_Rajalingham2019, self).__init__(
            identifier=identifier,
            ceiling_func=ceiling_func,
            version=1, parent='IT',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        # TODO: Both animals were previously trained on other images of other objects, and were proficient in
        #  discriminating among over 35 arbitrarily sampled basic-level object categories
        training_stimuli = self._training_stimuli
        candidate.start_task(task=BrainModel.Task.probabilities, fitting_stimuli=training_stimuli)

        # "[...] inactivation sessions were interleaved over days with control behavioral sessions.
        # Thus, each inactivation experiment consisted of three behavioral sessions:
        # the baseline or pre-control session (1 day prior to injection),
        # the inactivation session,
        # and the recovery or post-control session (2 days after injection)"
        # --> we here front-load one control session and then run many inactivation sessions
        # control
        unperturbed_behavior = self._perform_task_unperturbed(candidate)

        # silencing sessions
        bootstrap_scores = []
        for bootstrap in range(self._num_experiment_bootstraps):
            behaviors = [unperturbed_behavior]
            # "We varied the location of microinjections to randomly sample the ventral surface of IT
            # (from approximately + 8mm AP to approx + 20mm AP)."
            # stay between [0, 10] since that is the extent of the tissue

            random_state = RandomState(1)
            injection_locations = random_state.uniform(low=0, high=10, size=(self._num_sites_per_hemisphere * 2, 2))
            injection_hemispheres = (['left'] * self._num_sites_per_hemisphere) + \
                                    (['right'] * self._num_sites_per_hemisphere)
            for site, (hemisphere, injection_location) in enumerate(zip(injection_hemispheres, injection_locations)):
                perturbation_parameters = {**MUSCIMOL_PARAMETERS,
                                           **{'location': injection_location, 'hemisphere': hemisphere}}
                # TODO: we need to change the match-to-sample task paradigm here in order to account for lateralization.
                #  In particular, they treat contra trials as those trials where
                #  "images in which the center of the target object was contralateral to the injection hemisphere"
                perturbed_behavior = self._perform_task_perturbed(candidate,
                                                                  perturbation_parameters=perturbation_parameters,
                                                                  site_number=site)
                behaviors.append(perturbed_behavior)

            behaviors = merge_data_arrays(behaviors)
            behaviors = self.align_task_names(behaviors)

            source, target = self._characterize(behaviors, self._target_assembly)
            score = self._metric(source, target)

            score = score.expand_dims('bootstrap')
            score['bootstrap'] = [bootstrap]
            for attr_key, attr_value in score.attrs.items():
                score.attrs[attr_key] = attr_value.expand_dims('bootstrap')
                score.attrs[attr_key]['bootstrap'] = [bootstrap]
            bootstrap_scores.append(score)

        # average score over bootstraps
        merged_scores = merge_data_arrays(bootstrap_scores)
        score = merged_scores.mean('bootstrap')
        for attr_key in bootstrap_scores[0].attrs:
            merged_values = merge_data_arrays([score.attrs[attr_key] for score in bootstrap_scores])
            score.attrs[attr_key] = merged_values
        return score

    def _perform_task_unperturbed(self, candidate: BrainModel):
        candidate.perturb(perturbation=None, target='IT')  # reset
        behavior = candidate.look_at(self._test_stimuli, number_of_trials=None)
        behavior = behavior.expand_dims('injected')
        behavior['injected'] = [False]

        return behavior

    def _perform_task_perturbed(self, candidate: BrainModel, perturbation_parameters, site_number):
        candidate.perturb(perturbation=None, target='IT')  # reset
        candidate.perturb(perturbation=BrainModel.Perturbation.muscimol,
                          target='IT',
                          perturbation_parameters=perturbation_parameters)
        behavior = candidate.look_at(self._test_stimuli)

        behavior = behavior.expand_dims('injected').expand_dims('site')
        behavior['injected'] = [True]
        behavior['site_iteration'] = 'site', [site_number]
        behavior['hemisphere'] = 'site', [perturbation_parameters['hemisphere']]
        behavior['site_x'] = 'site', [perturbation_parameters['location'][0]]
        behavior['site_y'] = 'site', [perturbation_parameters['location'][1]]
        behavior['site_z'] = 'site', [0]
        behavior = type(behavior)(behavior)  # make sure site and injected are indexed

        return behavior

    @staticmethod
    def align_task_names(behaviors):
        behaviors = type(behaviors)(behaviors.values, coords={
            coord: (dims, values if coord not in ['object_name', 'truth', 'image_label', 'choice']
            else [TASK_LOOKUP[name] if name in TASK_LOOKUP else name for name in behaviors[coord].values])
            for coord, dims, values in walk_coords(behaviors)},
                                    dims=behaviors.dims)
        return behaviors

    @staticmethod
    def sample_grid_points(low, high, num_x, num_y):
        assert len(low) == len(high) == 2
        grid_x, grid_y = np.meshgrid(np.linspace(low[0], high[0], num_x),
                                     np.linspace(low[1], high[1], num_y))
        return np.stack((grid_x.flatten(), grid_y.flatten()), axis=1)  # , np.zeros(num_x * num_y) for empty z dimension

    def sample_points(self, low, high, num):
        assert len(low) == len(high) == 2
        random_state = RandomState(0)
        points_x = random_state.uniform(low=low[0], high=high[0], size=num)
        points_y = random_state.uniform(low=low[1], high=high[1], size=num)
        return np.stack((points_x, points_y), axis=1)


def Rajalingham2019GlobalDeficitsSignificant():
    characterization = CharacterizeDeltas()
    metric = SignificantPerformanceChange(condition_name='injected',
                                          condition_value1=False, condition_value2=True,
                                          trial_dimension='task_site_injected')

    def filter_global_metric(source_assembly, target_assembly):
        dprime_assembly_all = characterization.characterize(source_assembly)
        dprime_assembly = characterization.subselect_tasks(dprime_assembly_all, target_assembly)
        flat_source_assembly = flatten_assembly_dims(dprime_assembly, dim_coords=dict(
            task='task_number', site='site_iteration', injected='injected_'))

        target_assembly = target_assembly.sel(visual_field='all')
        target_assembly = target_assembly.mean('bootstrap')
        aggregate_target = target_assembly.mean('task').mean('site')  # aggregate for metric

        return metric(flat_source_assembly, aggregate_target)

    return _Rajalingham2019(identifier='dicarlo.Rajalingham2019-global_deficits_significant',
                            metric=filter_global_metric)


def Rajalingham2019LateralDeficitDifference():
    characterization = CharacterizeDeltas()
    metric = SignificantPerformanceChange(condition_name='visual_field',
                                          condition_value1='ipsi', condition_value2='contra',
                                          trial_dimension='task_site')

    def filter_visual_fields_metric(source_assembly, target_assembly):
        dprime_assembly_all = characterization.characterize(source_assembly)
        dprime_assembly = characterization.subselect_tasks(dprime_assembly_all, target_assembly)
        source_deficit_assembly = characterization.compute_differences(dprime_assembly)
        hemisphere_visualfield = {'left': 'ipsi', 'right': 'contra'}  # FIXME
        source_deficit_assembly['visual_field'] = 'site', [hemisphere_visualfield[hemisphere] for hemisphere in
                                                           source_deficit_assembly['hemisphere'].values]
        flat_source_deficit_assembly = flatten_assembly_dims(source_deficit_assembly, dim_coords=dict(
            task='task_number', site='site_iteration'))

        target_assembly = target_assembly.mean('bootstrap')
        target_assembly = target_assembly[{'visual_field': [visual_field in ['ipsi', 'contra'] for visual_field in
                                                            target_assembly['visual_field'].values]}]
        target_deficit_assembly = characterization.compute_differences(target_assembly)
        target_deficit_assembly = target_deficit_assembly.mean('task').mean('site')  # aggregate for metric
        return metric(flat_source_deficit_assembly, target_deficit_assembly)

    return _Rajalingham2019(identifier='Rajalingham2019-lateral_deficit_difference',
                            metric=filter_visual_fields_metric)


def Rajalingham2019SpatialCorrelationSignificant():
    metric = SpatialCharacterizationMetric(similarity_metric=SignificantCorrelation(x_coord='distance'),
                                           characterization=CharacterizeDeltas())

    def filter_global_metric(source_assembly, target_assembly):
        target_assembly = target_assembly.sel(visual_field='all')
        return metric(source_assembly, target_assembly)

    return _Rajalingham2019(identifier='dicarlo.Rajalingham2019-spatial_correlation_significant',
                            metric=filter_global_metric)


def Rajalingham2019SpatialCorrelationSimilarity():
    metric = SpatialCharacterizationMetric(similarity_metric=DifferenceOfCorrelations(correlation_variable='distance'),
                                           characterization=CharacterizeDeltas())

    def filter_global_metric(source_assembly, target_assembly):
        target_assembly = target_assembly.sel(visual_field='all')
        return metric(source_assembly, target_assembly)

    return _Rajalingham2019(identifier='dicarlo.Rajalingham2019-spatial_correlation_similarity',
                            metric=filter_global_metric)


def Rajalingham2019DeltaPredictionTask():
    metric = DeltaPredictionTask()

    def filter_global_metric(source_assembly, target_assembly):
        target_assembly = target_assembly.sel(visual_field='all')
        return metric(source_assembly, target_assembly)

    return _Rajalingham2019(identifier='dicarlo.Rajalingham2019-deficit_prediction_task',
                            # num_sites=100,  # TODO
                            characterize=CharacterizeDeltas(),
                            metric=filter_global_metric,
                            ceiling_func=None,  # TODO
                            num_experiment_bootstraps=3,  # fixme
                            )


def Rajalingham2019DeltaPredictionObject():
    metric = DeltaPredictionObject()

    def filter_global_metric(source_assembly, target_assembly):
        target_assembly = target_assembly.sel(visual_field='all')
        return metric(source_assembly, target_assembly)

    return _Rajalingham2019(identifier='dicarlo.Rajalingham2019-deficit_prediction_object',
                            # num_sites=100,  # TODO
                            characterize=CharacterizeDeltas(),
                            metric=filter_global_metric,
                            ceiling_func=None,  # TODO
                            num_experiment_bootstraps=3,  # fixme
                            )


def DicarloRajalingham2019SpatialDeficitsQuantified():
    def inv_ks_similarity(p, q):
        """
        Inverted ks similarity -> resulting in a score within [0,1], 1 being a perfect match
        """
        import scipy.stats
        return 1 - scipy.stats.ks_2samp(p, q)[0]

    similarity_metric = SpatialCorrelationSimilarity(similarity_function=inv_ks_similarity,
                                                     bin_size_mm=.8)  # arbitrary bin size
    metric = SpatialCharacterizationMetric(similarity_metric=similarity_metric, characterization=CharacterizeDeltas())
    metric._similarity_metric = similarity_metric
    benchmark = _Rajalingham2019(identifier='dicarlo.Rajalingham2019.IT-spatial_deficit_similarity_quantified',
                                 metric=metric)

    # TODO really messy solution | only works after benchmark metric has been called once
    benchmark._ceiling_func = lambda: InterIndividualStatisticsCeiling(similarity_metric)(
        benchmark._metric._similarity_metric.target_statistic)
    return benchmark


def flatten_assembly_dims(assembly, dim_coords):
    # flatten dimensions: first convert MultiIndices to single index each, then stack and re-introduce MultiIndex
    single_index_source = assembly.reset_index(list(dim_coords))
    single_index_source = single_index_source.set_index(dim_coords)
    joint_dim_name = '_'.join(dim_coords)
    flat_source_deficit_assembly = single_index_source.stack({joint_dim_name: list(dim_coords)})
    flat_source_deficit_assembly = type(flat_source_deficit_assembly)(flat_source_deficit_assembly)  # re-index
    return flat_source_deficit_assembly


class CharacterizeDeltas:
    def __call__(self, assembly1, assembly2):
        """
        :param assembly1: a tuple with the first element representing the control behavior in the format of
            `presentation: p, choice: c` and the second element representing inactivations behaviors in
            `presentation: p, choice: c, site: n`
        :param assembly2: a processed assembly in the format of `injected :2, task: c * (c-1), site: m`
        :return the two characterized difference assemblies in the form `task x site`
        """
        assembly1_characterized = self.characterize(assembly1)
        assembly1_tasks = self.subselect_tasks(assembly1_characterized, assembly2)
        assembly1_differences = self.compute_differences(assembly1_tasks)
        assembly2_differences = self.compute_differences(assembly2)
        assembly2_differences = assembly2_differences.mean('bootstrap')
        return assembly1_differences, assembly2_differences

    def characterize(self, assembly):
        """ compute per-task performance from `presentation x choice` assembly """
        # xarray can't do multi-dimensional grouping, do things manually
        o2s = []
        adjacent_values = assembly['injected'].values, assembly['site'].values
        for injected, site in tqdm(itertools.product(*adjacent_values), desc='characterize',
                                   total=np.prod([len(values) for values in adjacent_values])):
            current_assembly = assembly.sel(injected=injected, site=site)
            o2 = _o2(current_assembly)
            o2 = o2.expand_dims('injected').expand_dims('site')
            o2['injected'] = [injected]
            for (coord, _, _), value in zip(walk_coords(assembly['site']), site):
                o2[coord] = 'site', [value]
            o2 = DataAssembly(o2)  # ensure multi-index on site
            o2s.append(o2)
        o2s = merge_data_arrays(o2s)  # this only takes ~1s, ok
        return o2s

    def subselect_tasks(self, assembly, reference_assembly):
        tasks_left, tasks_right = reference_assembly['task_left'].values, reference_assembly['task_right'].values
        task_values = [assembly.sel(task_left=task_left, task_right=task_right).values
                       for task_left, task_right in zip(tasks_left, tasks_right)]
        task_values = type(assembly)(task_values, coords=
        {**{
            'task_number': ('task', reference_assembly['task_number'].values),
            'task_left': ('task', tasks_left),
            'task_right': ('task', tasks_right),
        }, **{coord: (dims, values) for coord, dims, values in walk_coords(assembly)
              if not any(array_is_element(dims, dim) for dim in ['task_left', 'task_right'])}
         }, dims=['task'] + [dim for dim in assembly.dims if
                             dim not in ['task_left', 'task_right']])
        return task_values

    def compute_differences(self, behaviors):
        """
        :param behaviors: an assembly with a dimension `injected` and values `[True, False]`
        :return: the difference between these two conditions (injected - control)
        """
        return behaviors.sel(injected=True) - behaviors.sel(injected=False)


class Rajalingham2019DeltaPredictionSpace(_Rajalingham2019):
    def __init__(self,
                 *args,
                 identifier='dicarlo.Rajalingham2019-deficit_prediction_space',
                 # num_sites=100, # TODO
                 num_epochs=100,
                 **kwargs):
        super(Rajalingham2019DeltaPredictionSpace, self).__init__(
            *args, identifier=identifier, metric=None, **kwargs)
        self._num_epochs = num_epochs
        self._characterization = CharacterizeDeltas()
        self._correlation = pearsonr_correlation(xarray_kwargs=dict(
            correlation_coord='task_number', group_coord='site_iteration', drop_target_nans=True))
        self._random_state = RandomState(0)

    def __call__(self, candidate: BrainModel) -> Score:
        target_assembly = self._target_assembly.sel(visual_field='all')
        target_assembly = self._characterization.compute_differences(target_assembly)
        target_assembly = target_assembly.mean('bootstrap')

        candidate.start_task(task=BrainModel.Task.probabilities, fitting_stimuli=self._training_stimuli)
        unperturbed_source_behavior = self._perform_task_unperturbed(candidate)

        subject_scores = []
        for subject in target_assembly['monkey'].values:
            subject_assembly = target_assembly.sel(monkey=subject)
            subject_score = self.adapt_tissues(target_assembly=subject_assembly, candidate=candidate,
                                               unperturbed_source_behavior=unperturbed_source_behavior)
            subject_score = subject_score.expand_dims('subject')
            subject_score['subject'] = [subject]
            subject_scores.append(subject_score)
        subject_scores = Score.merge(subject_scores)
        score = subject_scores.mean('subject')
        return score

    def adapt_tissues(self, target_assembly, candidate, unperturbed_source_behavior):
        # iterate over all sites to hold out
        scores = []
        for heldout_site in target_assembly['site_iteration'].values:
            test_target = target_assembly[{'site': [site == heldout_site
                                                    for site in target_assembly['site_iteration'].values]}]
            train_target = target_assembly[{'site': [site != heldout_site
                                                     for site in target_assembly['site_iteration'].values]}]
            # initialization
            a = self._random_state.random(size=(3, 3))
            offset = self._random_state.random(size=3)
            perturbation_parameters = copy.deepcopy(MUSCIMOL_PARAMETERS)

            # adapt
            errors_list = []
            parameters_list = []
            progress_bar = tqdm(total=self._num_epochs, desc="epochs")
            minimize_function = functools.partial(self.minimize_function,
                                                  train_target_assembly=train_target,
                                                  perturbation_parameters=perturbation_parameters,
                                                  candidate=candidate,
                                                  unperturbed_source_behavior=unperturbed_source_behavior,
                                                  errors_list=errors_list,
                                                  parameters_list=parameters_list,
                                                  num_epochs=self._num_epochs,
                                                  progress_bar=progress_bar,
                                                  )
            x0 = np.concatenate((a, [offset]))
            try:
                optimization_result = minimize(minimize_function, x0=np.reshape(x0, -1),
                                               options=dict(maxiter=self._num_epochs))
            except StopIteration:
                print("\nSTAHP\n")
            progress_bar.close()
            # fit_parameters = optimization_result.x
            fit_parameters = parameters_list[-1]
            a, offset = self.unravel_parameters(fit_parameters)

            # evaluate
            source_behavior = self.corresponding_source_deltas(candidate, target_site=test_target, a=a, offset=offset,
                                                               perturbation_parameters=perturbation_parameters)
            source_deltas = self.compute_source_deltas(unperturbed_source_behavior, source_behavior,
                                                       reference_assembly=test_target)
            score = self._correlation(source_deltas, test_target)
            score.attrs['predictions'] = source_deltas
            score.attrs['target'] = test_target
            scores.append(score)

            # plot TEMPORARY fixme
            errors_list = np.array(errors_list)
            print(errors_list)
            np.save(str(Path(__file__).parent / 'errors_list.npy'), errors_list)

            from matplotlib import pyplot
            fig, ax = pyplot.subplots()
            losses = errors_list
            window_size = 10
            running_average = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
            ax.scatter(np.arange(len(losses)), losses)
            ax.plot(np.arange(len(running_average)), running_average)
            savepath = f'/braintree/home/msch/direct-causality/plots/Rajalingham2019/' \
                       f'space_prediction-site{heldout_site}.png'
            fig.savefig(savepath)
            print(f"Saved to {savepath}")
            fig.show()
        scores = Score.merge(*scores)
        return scores

    def minimize_function(self, fit_parameters, train_target_assembly, perturbation_parameters,
                          candidate, unperturbed_source_behavior, num_epochs, errors_list, parameters_list,
                          progress_bar):
        parameters_list.append(fit_parameters)
        a, offset = self.unravel_parameters(fit_parameters)
        site_iterations = train_target_assembly['site_iteration'].values
        errors = []
        for site_iteration in site_iterations:
            target_site = train_target_assembly[{'site': [site == site_iteration
                                                          for site in train_target_assembly['site_iteration']]}]
            source_behavior = self.corresponding_source_deltas(candidate, target_site=target_site, a=a, offset=offset,
                                                               perturbation_parameters=perturbation_parameters)
            source_deltas = self.compute_source_deltas(unperturbed_source_behavior, source_behavior,
                                                       reference_assembly=train_target_assembly)
            # Note that some target site task deltas might be nan at this point,
            # but this is already ignored by the error function, so we do not need to drop these values
            site_error = self.mean_squared_error(source_deltas.squeeze('site'), target_site.squeeze('site')).values
            errors.append(site_error)
        error = np.mean(errors)
        progress_bar.update(1)
        errors_list.append(error)
        print(len(errors_list), error)
        if len(errors_list) >= num_epochs:
            raise StopIteration()
        return error

    def corresponding_source_deltas(self, candidate, target_site, a, offset, perturbation_parameters):
        xyz_target = np.array([target_site[f'site_{coordinate}'] for coordinate in ['x', 'y', 'z']]).squeeze()
        xyz_source = LinearTransform()(x=xyz_target, a=a, offset=offset)
        perturbation_parameters['location'] = xyz_source[:2]  # ignore z -- todo move this to the model
        perturbation_parameters['hemisphere'] = None  # todo target_site['hemisphere']
        source_behavior = self._perform_task_perturbed(
            candidate, perturbation_parameters=perturbation_parameters, site_number=target_site['site_number'].item())
        return source_behavior

    def unravel_parameters(self, fit_parameters):
        fit_parameters = np.reshape(fit_parameters, (4, 3))
        a, offset = fit_parameters[:3, :], fit_parameters[-1, :]
        return a, offset

    def compute_source_deltas(self, unperturbed_behavior, perturbed_behavior, reference_assembly):
        merged_behaviors = merge_data_arrays([unperturbed_behavior, perturbed_behavior])
        merged_behaviors = self.align_task_names(merged_behaviors)
        characterized = self._characterization.characterize(merged_behaviors)
        characterized = self._characterization.subselect_tasks(
            assembly=characterized, reference_assembly=reference_assembly)
        deltas = self._characterization.compute_differences(characterized)
        return deltas

    def mean_squared_error(self, predicted, truth):
        assert all(predicted['task'] == truth['task'])
        return np.mean((truth - predicted) ** 2)


class LinearTransform:
    def __call__(self, x: np.ndarray, a: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """
        :param x: 1*3 array [x, y, z]
        :param a: 3x3 array
        :param offset: 1*3 array [x, y, z]
        :return: 1*3 array [x, y, z]
        """
        return np.dot(a, x) + offset

    def inverse(self, y: np.ndarray, a: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """
        :param y: 1*3 array [x, y, z]
        :param a: 3x3 array
        :param offset: 1*3 array [x, y, z]
        :return: 1*3 array [x, y, z]
        """
        return np.divide((y - offset), a)
