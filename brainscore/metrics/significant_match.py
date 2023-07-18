import numpy as np
import pingouin as pg
from brainio.assemblies import DataAssembly
from scipy.stats import pearsonr, fisher_exact

from brainscore.metrics import Score, Metric


class ContingencyMatch(Metric):
    """
    Tests that two contingency tables match, i.e.
    Uses Fisher's exact test for statistical significance.
    """

    def __init__(self, category1_name: str, category1_value1: object, category1_value2: object,
                 category2_name: str, category2_value1: object, category2_value2: object,
                 significance_threshold=0.05):
        """
        :param category1_name, category2_name: The coordinate name corresponding to category 1/2, e.g. `stimulated`
        :param category1_value1, category2_value1: The initial value of the category, e.g. `False`
        :param category1_value2, category2_value2: The alternative value of the category, e.g. `True`
        :param significance_threshold: At which probability threshold to consider effects significant
        """
        super(ContingencyMatch, self).__init__()
        self.category1_value1 = category1_value1
        self.category1_value2 = category1_value2
        self.category2_name = category2_name
        self.category2_value1 = category2_value1
        self.category2_value2 = category2_value2
        self.category1_name = category1_name
        self.significance_threshold = significance_threshold

    def __call__(self, source, target):
        """
        :param source: Aggregate performance numbers of two dimensions, `category1_name` x `category2_name`,
            and two values each `category1_value1, category1_value2`, and `category2_value1, category2_value2`.
            This will be used to determine the expected direction from the condition change (increase/decrease).
        :param target: Same format as `source`.
        :return: A :class:`~brainscore.metrics.Score` of 1 if the candidate_behaviors significantly change in the same
            direction as the aggregate_target; 0 otherwise
        """
        source, target = self.align(source), self.align(target)
        _, p_target = fisher_exact(target)
        expect_significant = p_target < self.significance_threshold
        _, p_source = fisher_exact(source)
        source_significant = p_source < self.significance_threshold
        return Score([expect_significant == source_significant],
                     coords={'aggregation': ['center']}, dims=['aggregation'])

        # first figure out which direction the target went
        expected_same_direction = aggregate_target.sel(**{self.condition_name: self.condition_value2}) - \
                                  aggregate_target.sel(**{self.condition_name: self.condition_value1})
        # test if the source change is significant

    def align(self, assembly):
        # assume for now that everything is already in order instead of sorting ourselves
        np.testing.assert_array_equal(assembly.dims, [self.category1_name, self.category2_name])
        np.testing.assert_array_equal(assembly[self.category1_name], [self.category1_value1, self.category1_value2])
        np.testing.assert_array_equal(assembly[self.category2_name], [self.category2_value1, self.category2_value2])
        return assembly


class SignificantPerformanceChange(Metric):
    """
    Tests that the candidate behaviors changed in the same direction as the target.
    Significance tests are run with ANOVA.
    """

    def __init__(self, condition_name: str, condition_value1: object, condition_value2: object,
                 significance_threshold=0.05, trial_dimension='presentation'):
        """
        :param condition_name: The coordinate name corresponding to the condition for which performance is tested,
            e.g. `laser_on`
        :param condition_value1: The initial value of the condition, e.g. `False`
        :param condition_value2: The value of the condition after changing it, e.g. `True`
        :param significance_threshold: At which probability threshold to consider effects significant
        :param trial_dimension: Which dimension holds accuracy trials in the source assembly
        """
        super(SignificantPerformanceChange, self).__init__()
        self.condition_name = condition_name
        self.condition_value1 = condition_value1
        self.condition_value2 = condition_value2
        self.significance_threshold = significance_threshold
        self.trial_dimension = trial_dimension

    def __call__(self, source, aggregate_target):
        """
        :param source: Per-trial behaviors (_not_ aggregate performance measures),
            in the `self.trial_dimension` dimension.
        :param aggregate_target: Performance numbers for the experimental observations, i.e. _not_ per-trial data.
            This will be used to determine the expected direction from the condition change (increase/decrease).
            Note that since these are aggregate numbers, a significant change is assumed for the target.
        :return: A :class:`~brainscore.metrics.Score` of 1 if the candidate_behaviors significantly change in the same
            direction as the aggregate_target; 0 otherwise
        """
        # first figure out which direction the target went
        target_accuracy1 = aggregate_target.sel(**{self.condition_name: self.condition_value1}).squeeze()
        target_accuracy2 = aggregate_target.sel(**{self.condition_name: self.condition_value2}).squeeze()
        expected_direction = target_accuracy2 - target_accuracy1
        # test if the source change is significant
        difference_significant = is_significantly_different(DataAssembly(source), between=self.condition_name)
        accuracy1 = self.compute_accuracy(source, condition_value=self.condition_value1)
        accuracy2 = self.compute_accuracy(source, condition_value=self.condition_value2)
        source_difference = accuracy2 - accuracy1
        # match?
        same_direction = np.sign(source_difference) == np.sign(expected_direction)
        score = same_direction and difference_significant
        score = Score([score], coords={'aggregation': ['center']}, dims=['aggregation'])
        score.attrs['accuracy1'] = accuracy1
        score.attrs['accuracy2'] = accuracy2
        score.attrs['delta_accuracy'] = source_difference
        return score

    def compute_accuracy(self, assembly, condition_value):
        condition_assembly = assembly.sel(**{self.condition_name: condition_value})
        correct = condition_assembly == condition_assembly['label']
        accuracy = correct.mean(self.trial_dimension)
        return accuracy


class NoSignificantPerformanceChange(Metric):
    """
    Tests that the candidate behaviors did not change.
    Significance tests are run with ANOVA.
    """

    def __init__(self, condition_name: str, condition_value1: object, condition_value2: object,
                 significance_threshold=0.05):
        """
        :param condition_name: The coordinate name corresponding to the condition for which performance is tested,
            e.g. `laser_on`
        :param condition_value1: The initial value of the condition, e.g. `False`
        :param condition_value2: The value of the condition after changing it, e.g. `True`
        :param significance_threshold: At which probability threshold to consider effects significant
        """
        super(NoSignificantPerformanceChange, self).__init__()
        self.condition_name = condition_name
        self.condition_value1 = condition_value1
        self.condition_value2 = condition_value2
        self.significance_threshold = significance_threshold

    def __call__(self, source, aggregate_target):
        """
        :param source: Per-trial behaviors (_not_ aggregate performance measures) in the `presentation` dimension.
        :param aggregate_target: Target assembly, will be ignored.
        :return: A :class:`~brainscore.metrics.Score` of 1 if the candidate_behaviors did not significantly change;
            0 otherwise
        """
        # test if the source change is significant
        difference_significant = is_significantly_different(DataAssembly(source), between=self.condition_name)
        score = not difference_significant
        score = Score([score], coords={'aggregation': ['center']}, dims=['aggregation'])
        return score


def is_significantly_different(assembly, between, significance_threshold=0.05):
    """
    ANOVA between conditions
    :param assembly: the assembly with the values to compare
    :param between: condition to compare between, e.g. "is_face_selective"
    :param significance_threshold: p-value threshold, e.g. 0.05
    :return:
    """
    # convert assembly into dataframe
    data = assembly.to_pandas().reset_index()
    data = data.rename(columns={0: 'values'})
    value0 = data['values'][0]
    data['values'] = [int(value == value0) for value in data['values']]  # convert to int for ANOVA
    anova = pg.anova(data=data, dv='values', between=between)
    pvalue = anova['p-unc'][0]
    significantly_different = pvalue < significance_threshold
    return significantly_different


class SignificantCorrelation(Metric):
    def __init__(self, x_coord, significance_threshold=0.05, ignore_nans=False):
        super(SignificantCorrelation, self).__init__()
        self.x_coord = x_coord
        self.significance_threshold = significance_threshold
        self.ignore_nans = ignore_nans

    def __call__(self, source, target):
        source_significant_direction = self.significant_direction(source)
        target_significant_direction = self.significant_direction(target)
        score = source_significant_direction == target_significant_direction
        return Score([score], coords={'aggregation': ['center']}, dims=['aggregation'])

    def significant_direction(self, assembly):
        """
        Tests whether and in which direction the assembly's values are significantly correlated with the
        assembly's coordinate's values (`assembly[self.x_coord]`)
        :return: +1 if the correlation is significantly positive, -1 if the correlation is significantly negative,
          False otherwise
        """
        x = assembly[self.x_coord].values
        y = assembly.values
        if self.ignore_nans:
            nan = np.isnan(x) | np.isnan(y)
            x = x[~nan]
            y = y[~nan]
        r, p = pearsonr(x, y)
        if p >= self.significance_threshold:
            return False
        # at this point, we know the correlation is significant
        if r > 0:
            return +1
        else:
            return -1
