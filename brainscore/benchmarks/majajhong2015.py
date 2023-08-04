import brainscore
from brainio.assemblies import NeuroidAssembly, walk_coords
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore.metrics import Score
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency
from brainscore.metrics.inter_individual_stats_ceiling import InterIndividualStatisticsCeiling
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, ScaledCrossRegressedCorrelation, \
    pls_regression, pearsonr_correlation
from brainscore.metrics.spatial_correlation import SpatialCorrelationSimilarity, inv_ks_similarity
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad

VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = 50
BIBTEX = """@article {Majaj13402,
            author = {Majaj, Najib J. and Hong, Ha and Solomon, Ethan A. and DiCarlo, James J.},
            title = {Simple Learned Weighted Sums of Inferior Temporal Neuronal Firing Rates Accurately Predict Human Core Object Recognition Performance},
            volume = {35},
            number = {39},
            pages = {13402--13418},
            year = {2015},
            doi = {10.1523/JNEUROSCI.5181-14.2015},
            publisher = {Society for Neuroscience},
            abstract = {To go beyond qualitative models of the biological substrate of object recognition, we ask: can a single ventral stream neuronal linking hypothesis quantitatively account for core object recognition performance over a broad range of tasks? We measured human performance in 64 object recognition tests using thousands of challenging images that explore shape similarity and identity preserving object variation. We then used multielectrode arrays to measure neuronal population responses to those same images in visual areas V4 and inferior temporal (IT) cortex of monkeys and simulated V1 population responses. We tested leading candidate linking hypotheses and control hypotheses, each postulating how ventral stream neuronal responses underlie object recognition behavior. Specifically, for each hypothesis, we computed the predicted performance on the 64 tests and compared it with the measured pattern of human performance. All tested hypotheses based on low- and mid-level visually evoked activity (pixels, V1, and V4) were very poor predictors of the human behavioral pattern. However, simple learned weighted sums of distributed average IT firing rates exactly predicted the behavioral pattern. More elaborate linking hypotheses relying on IT trial-by-trial correlational structure, finer IT temporal codes, or ones that strictly respect the known spatial substructures of IT ({\textquotedblleft}face patches{\textquotedblright}) did not improve predictive power. Although these results do not reject those more elaborate hypotheses, they suggest a simple, sufficient quantitative model: each object recognition task is learned from the spatially distributed mean firing rates (100 ms) of \~{}60,000 IT neurons and is executed as a simple weighted sum of those firing rates.SIGNIFICANCE STATEMENT We sought to go beyond qualitative models of visual object recognition and determine whether a single neuronal linking hypothesis can quantitatively account for core object recognition behavior. To achieve this, we designed a database of images for evaluating object recognition performance. We used multielectrode arrays to characterize hundreds of neurons in the visual ventral stream of nonhuman primates and measured the object recognition performance of \&gt;100 human observers. Remarkably, we found that simple learned weighted sums of firing rates of neurons in monkey inferior temporal (IT) cortex accurately predicted human performance. Although previous work led us to expect that IT would outperform V4, we were surprised by the quantitative precision with which simple IT-based linking hypotheses accounted for human behavior.},
            issn = {0270-6474},
            URL = {https://www.jneurosci.org/content/35/39/13402},
            eprint = {https://www.jneurosci.org/content/35/39/13402.full.pdf},
            journal = {Journal of Neuroscience}}"""


def _DicarloMajajHong2015Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(average_repetitions=False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(average_repetitions=True, region=region))
    return NeuralBenchmark(identifier=f'dicarlo.MajajHong2015.{region}-{identifier_metric_suffix}', version=3,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           bibtex=BIBTEX)


def DicarloMajajHong2015V4PLS():
    return _DicarloMajajHong2015Region('V4', identifier_metric_suffix='pls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=pls_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015ITPLS():
    return _DicarloMajajHong2015Region('IT', identifier_metric_suffix='pls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=pls_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015V4Mask():
    return _DicarloMajajHong2015Region('V4', identifier_metric_suffix='mask',
                                       similarity_metric=ScaledCrossRegressedCorrelation(
                                           regression=mask_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name')),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015ITMask():
    return _DicarloMajajHong2015Region('IT', identifier_metric_suffix='mask',
                                       similarity_metric=ScaledCrossRegressedCorrelation(
                                           regression=mask_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name')),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015V4RDM():
    return _DicarloMajajHong2015Region('V4', identifier_metric_suffix='rdm',
                                       similarity_metric=RDMCrossValidated(
                                           crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                       ceiler=RDMConsistency())


def DicarloMajajHong2015ITRDM():
    return _DicarloMajajHong2015Region('IT', identifier_metric_suffix='rdm',
                                       similarity_metric=RDMCrossValidated(
                                           crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                       ceiler=RDMConsistency())


def load_assembly(average_repetitions, region, access='private'):
    assembly = brainscore.get_assembly(name=f'dicarlo.MajajHong2015.{access}')
    assembly = assembly.sel(region=region)
    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly = assembly.squeeze("time_bin")
    assembly.load()
    assembly = assembly.transpose('presentation', 'neuroid')
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly


SPATIAL_BIN_SIZE_MM = .1  # .1 mm is an arbitrary choice


class DicarloMajajHong2015ITSpatialCorrelation(BenchmarkBase):
    def __init__(self):
        """
        This benchmark compares the distribution of pairwise response correlation as a function of distance between the
            data recorded in Majaj* and Hong* et al. 2015 and a candidate model.
        """
        self._assembly = self._load_assembly()
        self._metric = SpatialCorrelationSimilarity(similarity_function=inv_ks_similarity,
                                                    bin_size_mm=SPATIAL_BIN_SIZE_MM,
                                                    num_bootstrap_samples=100_000,
                                                    num_sample_arrays=10)
        ceiler = InterIndividualStatisticsCeiling(self._metric)
        super().__init__(identifier='dicarlo.MajajHong2015.IT-spatial_correlation',
                         ceiling_func=lambda: ceiler(LazyLoad(
                             lambda: self._metric.compute_global_tissue_statistic_target(self._assembly))),
                         version=1,
                         parent='IT',
                         bibtex=BIBTEX)

    def _load_assembly(self) -> NeuroidAssembly:
        assembly = brainscore.get_assembly('dicarlo.MajajHong2015').sel(region='IT')
        print("assembly:", assembly)
        assembly = self.squeeze_time(assembly)
        assembly = self.tissue_update(assembly)
        return assembly

    def __call__(self, candidate: BrainModel) -> Score:
        """
        This computes the statistics, i.e. the pairwise response correlation of candidate and target, respectively and
        computes a score based on the ks similarity of the two resulting distributions
        :param candidate: BrainModel
        :return: average inverted ks similarity for the pairwise response correlation compared to the MajajHong assembly
        """
        candidate.start_recording(recording_target='IT', time_bins=[(70, 170)],
                                  recording_type=BrainModel.RecordingType.exact,
                                  # "we implanted each monkey with three arrays in the left cerebral hemisphere"
                                  hemisphere=BrainModel.Hemisphere.left)
        candidate_assembly = candidate.look_at(self._assembly.stimulus_set)
        candidate_assembly = self.squeeze_time(candidate_assembly)

        score = self._metric(candidate_assembly, self._assembly)
        return score

    @staticmethod
    def tissue_update(assembly):
        """
        Temporary functions: Obsolete when all saved assemblies updated such that x and y coordinates
        of each array electrode are stored in assembly.neuroid.tissue_{x,y}
        """
        if not hasattr(assembly, 'tissue_x'):
            assembly['tissue_x'] = assembly['x']
            assembly['tissue_y'] = assembly['y']
        attrs = assembly.attrs
        assembly = type(assembly)(assembly.values, coords={
            coord: (dims, values) for coord, dims, values in walk_coords(assembly)}, dims=assembly.dims)
        assembly.attrs = attrs
        return assembly

    @staticmethod
    def squeeze_time(assembly):
        if 'time_bin' in assembly.dims:
            assembly = assembly.squeeze('time_bin')
        if hasattr(assembly, "time_step"):
            assembly = assembly.squeeze("time_step")
        return assembly
