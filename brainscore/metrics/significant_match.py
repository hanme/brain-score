import numpy as np
import pingouin as pg
from brainio.assemblies import DataAssembly
from scipy.stats import pearsonr

from brainscore.metrics import Score, Metric


class SignificantPerformanceChange(Metric):
    """
    Tests that the candidate behaviors changed in the same direction as the target.
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
        super(SignificantPerformanceChange, self).__init__()
        self.condition_name = condition_name
        self.condition_value1 = condition_value1
        self.condition_value2 = condition_value2
        self.significance_threshold = significance_threshold

    def __call__(self, source, aggregate_target):
        """
        :param source: Per-trial behaviors (_not_ aggregate performance measures) in the `presentation` dimension.
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
        accuracy1 = source.sel(**{self.condition_name: self.condition_value1}).mean('presentation')
        accuracy2 = source.sel(**{self.condition_name: self.condition_value2}).mean('presentation')
        source_difference = accuracy2 - accuracy1
        # match?
        same_direction = np.sign(source_difference) == np.sign(expected_direction)
        score = same_direction and difference_significant
        score = Score([score], coords={'aggregation': ['center']}, dims=['aggregation'])
        score.attrs['delta_accuracy'] = source_difference
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
